// crates/bt-data/src/cache.rs
// Author: Sourish Dey

use bt_core::{BtError, Candle, Result};
use chrono::{Duration, Utc};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;
use tracing::instrument;

fn to_bt_err(e: rusqlite::Error) -> BtError {
    BtError::Database(e.to_string())
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts INTEGER NOT NULL,
    o REAL NOT NULL,
    h REAL NOT NULL,
    l REAL NOT NULL,
    c REAL NOT NULL,
    v REAL NOT NULL,
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval ON ohlcv(symbol, interval);
CREATE INDEX IF NOT EXISTS idx_ohlcv_fetched_at ON ohlcv(fetched_at);
";

/// SQLite-backed cache for OHLCV data.
/// TTL: 5 minutes for intraday intervals, 24 hours for daily+ intervals.
pub struct Cache {
    conn: Mutex<Connection>,
}

impl Cache {
    /// Open or create cache at `path`.
    ///
    /// The parent directory is created if missing: `Connection::open` fails
    /// outright on a nonexistent directory, so a caller passing a path under a
    /// fresh install's `data/` would otherwise get an error from a cache that
    /// had no reason not to exist yet.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| BtError::Database(e.to_string()))?;
            }
        }
        let conn = Connection::open(path).map_err(to_bt_err)?;
        conn.execute_batch(SCHEMA).map_err(to_bt_err)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Get cached OHLCV data for symbol, interval and an optional time window.
    ///
    /// `start_ts`/`end_ts` bound the requested range in Unix seconds. The cache
    /// is keyed by symbol and interval, so without this filter a cache entry
    /// fetched for a long range (e.g. 1Y) would also be served for a short one
    /// (e.g. 1D), and the chart would draw the wrong number of bars. Pass
    /// `None` to accept the whole cached series.
    ///
    /// Returns None on cache miss or when the cached data is expired.
    #[instrument(skip(self))]
    pub fn get_ohlcv_in_range(
        &self,
        symbol: &str,
        interval: &str,
        start_ts: Option<i64>,
        end_ts: Option<i64>,
    ) -> Result<Option<Vec<Candle>>> {
        let ttl_secs = self.ttl_for_interval(interval);
        let cutoff = (Utc::now() - Duration::seconds(ttl_secs)).timestamp();

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT ts, o, h, l, c, v FROM ohlcv
                 WHERE symbol = ?1 AND interval = ?2 AND fetched_at > ?3
                   AND (?4 IS NULL OR ts >= ?4)
                   AND (?5 IS NULL OR ts <= ?5)
                 ORDER BY ts",
            )
            .map_err(to_bt_err)?;

        let rows = stmt
            .query_map(params![symbol, interval, cutoff, start_ts, end_ts], |row| {
                Ok(Candle {
                    t: row.get::<_, i64>(0)? as f64,
                    open: row.get(1)?,
                    high: row.get(2)?,
                    low: row.get(3)?,
                    close: row.get(4)?,
                    volume: row.get(5)?,
                })
            })
            .map_err(to_bt_err)?;

        let mut candles = Vec::new();
        for row in rows {
            candles.push(row.map_err(to_bt_err)?);
        }

        if candles.is_empty() {
            Ok(None)
        } else {
            Ok(Some(candles))
        }
    }

    /// Get cached OHLCV data for symbol and interval, ignoring any time window.
    /// Returns None if cache miss or data expired.
    #[instrument(skip(self))]
    pub fn get_ohlcv(&self, symbol: &str, interval: &str) -> Result<Option<Vec<Candle>>> {
        self.get_ohlcv_in_range(symbol, interval, None, None)
    }

    /// Store OHLCV data in cache.
    #[instrument(skip(self, candles))]
    pub fn put_ohlcv(&self, symbol: &str, interval: &str, candles: &[Candle]) -> Result<()> {
        if candles.is_empty() {
            return Ok(());
        }

        let now = Utc::now().timestamp();
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(to_bt_err)?;

        {
            let mut stmt = tx
                .prepare(
                    "INSERT OR REPLACE INTO ohlcv (symbol, interval, ts, o, h, l, c, v, fetched_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                )
                .map_err(to_bt_err)?;

            for c in candles {
                stmt.execute(params![
                    symbol, interval, c.t as i64, c.open, c.high, c.low, c.close, c.volume, now
                ])
                .map_err(to_bt_err)?;
            }
        }

        tx.commit().map_err(to_bt_err)?;
        Ok(())
    }

    /// Clear expired entries (older than max TTL).
    #[instrument(skip(self))]
    pub fn cleanup(&self) -> Result<usize> {
        let max_ttl = 86400 * 7;
        let cutoff = (Utc::now() - Duration::seconds(max_ttl)).timestamp();

        let conn = self.conn.lock().unwrap();
        let deleted = conn
            .execute("DELETE FROM ohlcv WHERE fetched_at < ?1", params![cutoff])
            .map_err(to_bt_err)?;

        Ok(deleted)
    }

    /// Get cache stats.
    #[instrument(skip(self))]
    pub fn stats(&self) -> Result<CacheStats> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM ohlcv", [], |row| row.get(0))
            .map_err(to_bt_err)?;
        let size: i64 = conn
            .query_row(
                "SELECT SUM(length(symbol) + length(interval) + 56) FROM ohlcv",
                [],
                |row| row.get(0),
            )
            .map_err(to_bt_err)
            .unwrap_or(0);

        Ok(CacheStats {
            entries: count as usize,
            approx_size_bytes: size as usize,
        })
    }

    fn ttl_for_interval(&self, interval: &str) -> i64 {
        match interval {
            "1m" | "5m" | "15m" | "30m" | "1h" => 300,
            "1d" | "1wk" | "1mo" => 86400,
            _ => 300,
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub approx_size_bytes: usize,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new("./data/cache.db").expect("Failed to create default cache")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache.db");
        let cache = Cache::new(&path).unwrap();

        let candles = vec![
            Candle::new(1_000_000.0, 100.0, 105.0, 99.0, 103.0, 1000.0),
            Candle::new(1_000_001.0, 103.0, 107.0, 102.0, 105.0, 1500.0),
        ];

        cache.put_ohlcv("TEST", "1d", &candles).unwrap();

        let retrieved = cache.get_ohlcv("TEST", "1d").unwrap().unwrap();
        assert_eq!(retrieved.len(), 2);
        assert!((retrieved[0].open - 100.0).abs() < f64::EPSILON);
        assert!((retrieved[1].close - 105.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache.db");
        let cache = Cache::new(&path).unwrap();

        let result = cache.get_ohlcv("NONEXISTENT", "1d").unwrap();
        assert!(result.is_none());
    }

    /// Regression test: the cache is keyed by (symbol, interval) only, so a
    /// long-range entry was also served for a short-range request and the chart
    /// drew the wrong number of bars. The time window must be honoured.
    #[test]
    fn test_cache_range_filter_excludes_out_of_window_bars() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache_range.db");
        let cache = Cache::new(&path).unwrap();

        let day = 86_400_i64;
        let base = 1_700_000_000_i64;
        // A year of daily bars, as a 1Y request would cache.
        let year: Vec<Candle> = (0..365)
            .map(|i| {
                Candle::new(
                    (base + i as i64 * day) as f64,
                    100.0,
                    105.0,
                    99.0,
                    103.0,
                    1000.0,
                )
            })
            .collect();
        cache.put_ohlcv("RANGE", "1d", &year).unwrap();

        // A 1D request at the very end of that year must not get all 365 bars.
        let one_day_start = base + 360 * day;
        let one_day_end = one_day_start + day;
        let intraday = cache
            .get_ohlcv_in_range("RANGE", "1d", Some(one_day_start), Some(one_day_end))
            .unwrap()
            .unwrap();
        assert!(
            intraday.len() < 10,
            "1D request returned {} bars, expected only the in-window ones",
            intraday.len()
        );
        for c in &intraday {
            assert!(
                c.t >= one_day_start as f64 && c.t <= one_day_end as f64,
                "bar {} escaped the requested window",
                c.t
            );
        }

        // The unfiltered accessor still returns the whole cached series.
        let all = cache.get_ohlcv("RANGE", "1d").unwrap().unwrap();
        assert_eq!(all.len(), 365);

        // A window that contains no data must report a miss, not stale rows.
        let outside = cache
            .get_ohlcv_in_range("RANGE", "1d", Some(0), Some(1000))
            .unwrap();
        assert!(outside.is_none(), "window outside the data must miss");
    }

    #[test]
    fn test_cache_range_filter_keeps_overlapping_bars() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache_range2.db");
        let cache = Cache::new(&path).unwrap();

        let day = 86_400_i64;
        let base = 1_700_000_000_i64;
        let year: Vec<Candle> = (0..100)
            .map(|i| {
                Candle::new(
                    (base + i as i64 * day) as f64,
                    100.0,
                    105.0,
                    99.0,
                    103.0,
                    1000.0,
                )
            })
            .collect();
        cache.put_ohlcv("RANGE2", "1d", &year).unwrap();

        let from = base + 10 * day;
        let to = base + 19 * day;
        let window = cache
            .get_ohlcv_in_range("RANGE2", "1d", Some(from), Some(to))
            .unwrap()
            .unwrap();
        assert_eq!(window.len(), 10, "inclusive bounds should return 10 bars");
        assert_eq!(window[0].t, from as f64);
        assert_eq!(window[9].t, to as f64);
    }

    #[test]
    fn test_cache_expiry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache.db");
        let cache = Cache::new(&path).unwrap();

        let candles = vec![Candle::new(1_000_000.0, 100.0, 105.0, 99.0, 103.0, 1000.0)];

        let old_time = (Utc::now() - Duration::seconds(400)).timestamp();
        cache
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO ohlcv VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params!["TEST", "1m", 1_000_000, 100.0, 105.0, 99.0, 103.0, 1000.0, old_time],
            )
            .unwrap();

        let result = cache.get_ohlcv("TEST", "1m").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_daily_cache_not_expired() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache.db");
        let cache = Cache::new(&path).unwrap();

        let candles = vec![Candle::new(1_000_000.0, 100.0, 105.0, 99.0, 103.0, 1000.0)];

        let old_time = (Utc::now() - Duration::seconds(12 * 3600)).timestamp();
        cache
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO ohlcv VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params!["TEST", "1d", 1_000_000, 100.0, 105.0, 99.0, 103.0, 1000.0, old_time],
            )
            .unwrap();

        let result = cache.get_ohlcv("TEST", "1d").unwrap();
        assert!(result.is_some());
    }
}
