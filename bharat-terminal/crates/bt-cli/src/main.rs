//! bt-cli: Bharat Terminal command-line renderer v2. Made by Sourish Dey.
//!
//! Generates all 25 visualizations to a local output directory using
//! real market data (via bt-data) with synthetic fallback.
//! Supports live mode with periodic refresh.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use tokio::time::sleep;

use bt_core::{synthetic_correlated_returns, synthetic_ohlcv, APP_NAME, AUTHOR, TAGLINE};
use bt_data::{DataService, Interval, COMPANY_LIST, DEFAULT_COMPANY};
use bt_viz::palette::Theme as VizTheme;
use bt_viz::{
    acf_pacf, adx, bb_width, candlestick, candlestick_3d, candlestick_bollinger, candlestick_ma,
    candlestick_macd, candlestick_rsi, correlation_heatmap, cumulative_delta, drawdown,
    efficient_frontier, heikin_ashi, multi_indicator, nifty_treemap, price_volume_scatter, renko,
    seasonality_polar, sector_performance, sector_treemap, sensex_heatmap, vol_smile,
    volume_profile, yield_curve,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ThemeArg {
    Dark,
    Light,
}

impl From<ThemeArg> for VizTheme {
    fn from(t: ThemeArg) -> Self {
        match t {
            ThemeArg::Dark => VizTheme::Dark,
            ThemeArg::Light => VizTheme::Light,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RangeArg {
    D1,
    W1,
    M1,
    M3,
    M6,
    Y1,
    Y5,
}

impl RangeArg {
    fn to_days(&self) -> i64 {
        match self {
            RangeArg::D1 => 1,
            RangeArg::W1 => 7,
            RangeArg::M1 => 30,
            RangeArg::M3 => 90,
            RangeArg::M6 => 180,
            RangeArg::Y1 => 365,
            RangeArg::Y5 => 1825,
        }
    }

    fn to_interval(&self) -> Interval {
        match self {
            RangeArg::D1 => Interval::Min5,
            RangeArg::W1 => Interval::Min15,
            RangeArg::M1 => Interval::Hour1,
            RangeArg::M3 => Interval::Day1,
            RangeArg::M6 => Interval::Day1,
            RangeArg::Y1 => Interval::Day1,
            RangeArg::Y5 => Interval::Week1,
        }
    }

    fn to_yahoo_range(&self) -> &'static str {
        match self {
            RangeArg::D1 => "1d",
            RangeArg::W1 => "5d",
            RangeArg::M1 => "1mo",
            RangeArg::M3 => "3mo",
            RangeArg::M6 => "6mo",
            RangeArg::Y1 => "1y",
            RangeArg::Y5 => "5y",
        }
    }
}

/// BHARAT TERMINAL v4 — Bloomberg power. Zero cost. Made in India.
#[derive(Debug, Parser)]
#[command(
    name = "bt-cli",
    version,
    about = "BHARAT TERMINAL v4 — render visualizations with real market data."
)]
struct Cli {
    /// Symbol to fetch (e.g., RELIANCE.NS, AAPL, BTC-USD). Default: RELIANCE.NS
    #[arg(long, default_value = "RELIANCE.NS")]
    symbol: String,

    /// Time range: 1d, 1w, 1m, 3m, 6m, 1y, 5y
    #[arg(long, value_enum, default_value_t = RangeArg::Y1)]
    range: RangeArg,

    /// Output directory for rendered PNGs.
    #[arg(long, default_value = "output")]
    out_dir: PathBuf,

    /// Color theme to render with.
    #[arg(long, value_enum, default_value_t = ThemeArg::Dark)]
    theme: ThemeArg,

    /// RNG seed for synthetic fallback data.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Enable live mode: re-fetch and re-render every 30 seconds.
    #[arg(long, default_value_t = false)]
    live: bool,

    /// List all supported companies and exit.
    #[arg(long, default_value_t = false)]
    list_companies: bool,

    /// Subcommand. When omitted, the classic render flow runs unchanged.
    #[command(subcommand)]
    command: Option<Commands>,
}

/// v4.0 subcommands. `render` is the historical default and stays the
/// no-subcommand path, so existing scripts keep working byte for byte.
#[derive(Debug, Clone, clap::Subcommand)]
enum Commands {
    /// Run a forecaster over fetched closes and print the path to stdout.
    Forecast {
        /// Symbol to forecast (e.g., RELIANCE.NS).
        #[arg(long, default_value = "RELIANCE.NS")]
        symbol: String,
        /// Engine: chronos, dlinear, nhits or auto.
        #[arg(long, default_value = "auto")]
        engine: String,
        /// Bars to project.
        #[arg(long, default_value_t = 20)]
        horizon: usize,
    },
    /// Manage the shared watchlist (same prefs.json the app reads).
    Watchlist {
        #[command(subcommand)]
        action: WatchlistAction,
    },
    /// Backtest an SMA-cross strategy over fetched closes.
    Backtest {
        /// Symbol to backtest (e.g., RELIANCE.NS).
        #[arg(long, default_value = "RELIANCE.NS")]
        symbol: String,
        /// Fast SMA window in bars.
        #[arg(long, default_value_t = 10)]
        fast: usize,
        /// Slow SMA window in bars.
        #[arg(long, default_value_t = 30)]
        slow: usize,
    },
    /// Screen cached symbols with technical filters.
    Screen {
        /// Filter expression, e.g. "rsi<30 AND change_5d>2".
        #[arg(long, default_value = "rsi<30")]
        filter: String,
        /// Output CSV path.
        #[arg(long, default_value = "screen_results.csv")]
        output: PathBuf,
    },
    /// Export portfolio holdings with cached prices to CSV.
    ExportPortfolio {
        /// portfolio.json path (default: exe-dir data/portfolio.json).
        #[arg(long)]
        portfolio: Option<PathBuf>,
        /// Output CSV path.
        #[arg(long, default_value = "portfolio_export.csv")]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, clap::Subcommand)]
enum WatchlistAction {
    /// Add a ticker (no-op when already starred or the list is full).
    Add {
        /// Ticker, e.g. RELIANCE.NS.
        symbol: String,
    },
    /// Remove a ticker (no-op when absent).
    Remove {
        /// Ticker, e.g. RELIANCE.NS.
        symbol: String,
    },
    /// Print starred tickers, one per line.
    List,
}

fn banner() {
    println!("================================================================");
    println!(" {APP_NAME} v4");
    println!(" {TAGLINE}");
    println!(" Made by {AUTHOR}");
    println!("================================================================");
}

fn list_companies() {
    println!("\nSupported Companies ({}):\n", COMPANY_LIST.len());
    println!("{:<35} {:<15} {}", "Name", "Ticker", "Exchange");
    println!("{}", "-".repeat(70));
    for (name, ticker, exchange) in COMPANY_LIST {
        println!("{:<35} {:<15} {}", name, ticker, exchange);
    }
    println!("\nTotal: {} companies", COMPANY_LIST.len());
}

async fn fetch_series(
    service: &DataService,
    symbol: &str,
    range: RangeArg,
) -> bt_core::OhlcvSeries {
    use chrono::{Duration, Utc};

    let end = Utc::now();
    let start = end - Duration::days(range.to_days());
    let interval = range.to_interval();

    match service.fetch_ohlcv(symbol, interval, start, end).await {
        Ok(series) => {
            println!(
                "  ✓ Fetched {} candles for {} ({} days, {:?})",
                series.candles.len(),
                symbol,
                range.to_days(),
                interval
            );
            series
        }
        Err(e) => {
            eprintln!(
                "  ✗ Failed to fetch {}: {}. Using synthetic fallback.",
                symbol, e
            );
            let days = (range.to_days() as f64
                / match interval {
                    Interval::Min1 => 1.0 / (24.0 * 60.0),
                    Interval::Min5 => 5.0 / (24.0 * 60.0),
                    Interval::Min15 => 15.0 / (24.0 * 60.0),
                    Interval::Min30 => 30.0 / (24.0 * 60.0),
                    Interval::Hour1 => 1.0 / 24.0,
                    Interval::Day1 => 1.0,
                    Interval::Week1 => 7.0,
                    Interval::Month1 => 30.0,
                }) as usize;
            synthetic_ohlcv(symbol, days.max(50), 42, 100.0)
        }
    }
}

async fn render_all(
    service: &DataService,
    symbol: &str,
    range: RangeArg,
    theme: VizTheme,
    seed: u64,
    out_dir: &PathBuf,
    iteration: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    use chrono::Utc;

    println!("\n[{}] Fetching data for {}...", iteration, symbol);
    let series = fetch_series(service, symbol, range).await;

    let timestamp = Utc::now().format("%H%M%S").to_string();
    let prefix = if iteration > 1 {
        format!("{:03}_", iteration)
    } else {
        String::new()
    };
    let mut rendered = 0usize;

    // 1. Candlestick + Volume
    {
        let cfg = candlestick::CandlestickConfig::new()
            .title("Candlestick + Volume")
            .theme(theme);
        let path = out_dir.join(format!("{}01_candlestick_{}.png", prefix, timestamp));
        candlestick::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[1/25] Candlestick + Volume         -> {}", path.display());
        rendered += 1;
    }

    // 2. Candlestick + MA
    {
        let cfg = candlestick_ma::CandlestickMAConfig::new()
            .title("Candlestick + MA (20,50,200)")
            .theme(theme);
        let path = out_dir.join(format!("{}02_candlestick_ma_{}.png", prefix, timestamp));
        candlestick_ma::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[2/25] Candlestick + MA             -> {}", path.display());
        rendered += 1;
    }

    // 3. Candlestick + Bollinger
    {
        let cfg = candlestick_bollinger::CandlestickBollingerConfig::new()
            .title("Candlestick + Bollinger Bands")
            .theme(theme);
        let path = out_dir.join(format!(
            "{}03_candlestick_bollinger_{}.png",
            prefix, timestamp
        ));
        candlestick_bollinger::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[3/25] Candlestick + Bollinger       -> {}", path.display());
        rendered += 1;
    }

    // 4. Candlestick + RSI
    {
        let cfg = candlestick_rsi::CandlestickRSIConfig::new()
            .title("Candlestick + RSI(14)")
            .theme(theme);
        let path = out_dir.join(format!("{}04_candlestick_rsi_{}.png", prefix, timestamp));
        candlestick_rsi::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[4/25] Candlestick + RSI            -> {}", path.display());
        rendered += 1;
    }

    // 5. Candlestick + MACD
    {
        let cfg = candlestick_macd::CandlestickMACDConfig::new()
            .title("Candlestick + MACD")
            .theme(theme);
        let path = out_dir.join(format!("{}05_candlestick_macd_{}.png", prefix, timestamp));
        candlestick_macd::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[5/25] Candlestick + MACD           -> {}", path.display());
        rendered += 1;
    }

    // 6. Heikin-Ashi
    {
        let cfg = heikin_ashi::HeikinAshiConfig::new()
            .title("Heikin-Ashi")
            .theme(theme);
        let path = out_dir.join(format!("{}06_heikin_ashi_{}.png", prefix, timestamp));
        heikin_ashi::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[6/25] Heikin-Ashi                  -> {}", path.display());
        rendered += 1;
    }

    // 7. Renko
    {
        let cfg = renko::RenkoConfig::new()
            .title("Renko")
            .theme(theme)
            .brick_size(10.0);
        let path = out_dir.join(format!("{}07_renko_{}.png", prefix, timestamp));
        renko::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[7/25] Renko                        -> {}", path.display());
        rendered += 1;
    }

    // 8. Volume Profile
    {
        let cfg = volume_profile::VolumeProfileConfig::new()
            .title("Volume Profile")
            .theme(theme);
        let path = out_dir.join(format!("{}08_volume_profile_{}.png", prefix, timestamp));
        volume_profile::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[8/25] Volume Profile               -> {}", path.display());
        rendered += 1;
    }

    // 9. Multi-Indicator Dashboard
    {
        let cfg = multi_indicator::MultiIndicatorConfig::new()
            .title("Multi-Indicator Dashboard")
            .theme(theme);
        let path = out_dir.join(format!("{}09_multi_indicator_{}.png", prefix, timestamp));
        multi_indicator::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[9/25] Multi-Indicator Dashboard     -> {}", path.display());
        rendered += 1;
    }

    // 10. Bollinger Band Width
    {
        let cfg = bb_width::BBWidthConfig::new()
            .title("Bollinger Band Width")
            .theme(theme);
        let path = out_dir.join(format!("{}10_bb_width_{}.png", prefix, timestamp));
        bb_width::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[10/25] Bollinger Band Width         -> {}", path.display());
        rendered += 1;
    }

    // 11. ADX / DI+ / DI-
    {
        let cfg = adx::ADXConfig::new().title("ADX / DI+ / DI-").theme(theme);
        let path = out_dir.join(format!("{}11_adx_{}.png", prefix, timestamp));
        adx::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[11/25] ADX / DI+ / DI-              -> {}", path.display());
        rendered += 1;
    }

    // 12. Nifty 50 Treemap (sample data)
    {
        let cfg = nifty_treemap::NiftyTreemapConfig::new()
            .title("Nifty 50 Treemap")
            .theme(theme);
        let path = out_dir.join(format!("{}12_nifty_treemap_{}.png", prefix, timestamp));
        nifty_treemap::render_sample_png(&cfg, path.to_str().unwrap())?;
        println!("[12/25] Nifty 50 Treemap            -> {}", path.display());
        rendered += 1;
    }

    // 13. Sensex Heatmap (sample data)
    {
        let cfg = sensex_heatmap::SensexHeatmapConfig::new()
            .title("Sensex Heatmap")
            .theme(theme);
        let path = out_dir.join(format!("{}13_sensex_heatmap_{}.png", prefix, timestamp));
        sensex_heatmap::render_sample_png(&cfg, path.to_str().unwrap())?;
        println!("[13/25] Sensex Heatmap              -> {}", path.display());
        rendered += 1;
    }

    // 14. Sector Performance (sample data)
    {
        let cfg = sector_performance::SectorPerformanceConfig::new()
            .title("Sector Performance")
            .theme(theme);
        let path = out_dir.join(format!("{}14_sector_performance_{}.png", prefix, timestamp));
        sector_performance::render_sample_png(&cfg, path.to_str().unwrap())?;
        println!("[14/25] Sector Performance           -> {}", path.display());
        rendered += 1;
    }

    // 15. Candlestick 3D
    {
        let cfg = candlestick_3d::Candlestick3DConfig::new()
            .title("Candlestick 3D")
            .theme(theme);
        let path = out_dir.join(format!("{}15_candlestick_3d_{}.png", prefix, timestamp));
        candlestick_3d::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[15/25] Candlestick 3D              -> {}", path.display());
        rendered += 1;
    }

    // 16. Price-Volume Scatter
    {
        let cfg = price_volume_scatter::PriceVolumeScatterConfig::new()
            .title("Price-Volume Scatter")
            .theme(theme);
        let path = out_dir.join(format!(
            "{}16_price_volume_scatter_{}.png",
            prefix, timestamp
        ));
        price_volume_scatter::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[16/25] Price-Volume Scatter        -> {}", path.display());
        rendered += 1;
    }

    // 17. Cumulative Delta
    {
        let cfg = cumulative_delta::CumulativeDeltaConfig::new().theme(theme);
        let path = out_dir.join(format!("{}17_cumulative_delta_{}.png", prefix, timestamp));
        cumulative_delta::render_png(&series, &cfg, path.to_str().unwrap())?;
        println!("[17/25] Cumulative Delta             -> {}", path.display());
        rendered += 1;
    }

    // 18. Drawdown
    {
        let cfg = drawdown::DrawdownConfig::new().theme(theme);
        let path = out_dir.join(format!("{}18_drawdown_{}.png", prefix, timestamp));
        drawdown::render_png(&series.closes(), &cfg, path.to_str().unwrap())?;
        println!("[18/25] Drawdown Underwater          -> {}", path.display());
        rendered += 1;
    }

    // 19. Correlation Heatmap (multi-symbol)
    {
        let symbols = [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "ITC.NS",
        ];
        let mut all_data = Vec::new();
        for sym in symbols {
            let s = fetch_series(service, sym, range).await;
            all_data.push((sym.to_string(), s.returns()));
        }
        let cfg = correlation_heatmap::CorrelationHeatmapConfig::new().theme(theme);
        let path = out_dir.join(format!(
            "{}19_correlation_heatmap_{}.png",
            prefix, timestamp
        ));
        correlation_heatmap::render_png(&all_data, &cfg, path.to_str().unwrap())?;
        println!("[19/25] Correlation Heatmap          -> {}", path.display());
        rendered += 1;
    }

    // 20. Volatility Smile (synthetic)
    {
        let curves = vec![
            vol_smile::synthetic_smile("7D", 0.28, 0.10, 14),
            vol_smile::synthetic_smile("30D", 0.22, 0.07, 14),
            vol_smile::synthetic_smile("90D", 0.19, 0.05, 14),
        ];
        let cfg = vol_smile::VolSmileConfig::new().theme(theme);
        let path = out_dir.join(format!("{}20_vol_smile_{}.png", prefix, timestamp));
        vol_smile::render_png(&curves, &cfg, path.to_str().unwrap())?;
        println!("[20/25] Volatility Smile / Skew      -> {}", path.display());
        rendered += 1;
    }

    // 21. Efficient Frontier (synthetic)
    {
        let expected_returns = vec![0.09, 0.14, 0.11, 0.16, 0.07, 0.10];
        let cov: Vec<Vec<f64>> = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        if i == j {
                            0.03 + i as f64 * 0.006
                        } else {
                            0.006
                        }
                    })
                    .collect()
            })
            .collect();
        let portfolios =
            efficient_frontier::simulate_portfolios(&expected_returns, &cov, 0.065, 2000, seed + 4);
        let cfg = efficient_frontier::EfficientFrontierConfig::new().theme(theme);
        let path = out_dir.join(format!("{}21_efficient_frontier_{}.png", prefix, timestamp));
        efficient_frontier::render_png(&portfolios, &cfg, path.to_str().unwrap())?;
        println!("[21/25] Efficient Frontier           -> {}", path.display());
        rendered += 1;
    }

    // 22. ACF/PACF
    {
        let returns = series.returns();
        let cfg = acf_pacf::AcfPacfConfig::new().theme(theme).max_lag(25);
        let path = out_dir.join(format!("{}22_acf_pacf_{}.png", prefix, timestamp));
        acf_pacf::render_png(&returns, &cfg, path.to_str().unwrap())?;
        println!("[22/25] ACF / PACF                 -> {}", path.display());
        rendered += 1;
    }

    // 23. Sector Treemap (sample)
    {
        let nodes = vec![
            sector_treemap::TreemapNode::new("Reliance", 1_800_000.0, 1.2),
            sector_treemap::TreemapNode::new("TCS", 1_400_000.0, -0.8),
            sector_treemap::TreemapNode::new("HDFC Bank", 1_100_000.0, 0.5),
            sector_treemap::TreemapNode::new("Infosys", 700_000.0, -1.5),
            sector_treemap::TreemapNode::new("ICICI Bank", 650_000.0, 2.1),
            sector_treemap::TreemapNode::new("ITC", 500_000.0, 0.1),
            sector_treemap::TreemapNode::new("L&T", 420_000.0, 0.9),
            sector_treemap::TreemapNode::new("Bharti Airtel", 610_000.0, -0.3),
        ];
        let cfg = sector_treemap::TreemapConfig::new()
            .title("BMAP -- NIFTY Sector Map")
            .theme(theme);
        let path = out_dir.join(format!("{}23_sector_treemap_{}.png", prefix, timestamp));
        sector_treemap::render_png(&nodes, &cfg, path.to_str().unwrap())?;
        println!("[23/25] Sector Treemap (BMAP)        -> {}", path.display());
        rendered += 1;
    }

    // 24. Yield Curve Family (synthetic)
    {
        let tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0];
        let curves = vec![
            yield_curve::synthetic_curve("2026-06-01", 6.8, -1.2, 0.4, &tenors),
            yield_curve::synthetic_curve("2026-07-15", 6.6, -1.0, 0.5, &tenors),
            yield_curve::synthetic_curve("2026-09-20", 6.5, -0.8, 0.6, &tenors),
        ];
        let cfg = yield_curve::YieldCurveConfig::new()
            .title("GOVT -- India Sovereign Yield Curve")
            .theme(theme);
        let path = out_dir.join(format!("{}24_yield_curve_{}.png", prefix, timestamp));
        yield_curve::render_png(&curves, &cfg, path.to_str().unwrap())?;
        println!("[24/25] Yield Curve Family           -> {}", path.display());
        rendered += 1;
    }

    // 25. Seasonality Polar Heatmap
    {
        let grid = seasonality_polar::synthetic_seasonality(seed + 6);
        let cfg = seasonality_polar::SeasonalityConfig::new().theme(theme);
        let path = out_dir.join(format!("{}25_seasonality_polar_{}.png", prefix, timestamp));
        seasonality_polar::render_png(&grid, &cfg, path.to_str().unwrap())?;
        println!("[25/25] Seasonality Polar Heatmap    -> {}", path.display());
        rendered += 1;
    }

    Ok(rendered)
}

/// Forecast `horizon` bars for `symbol` and print `engine, value` lines.
///
/// Fetches a year of daily bars (enough history for every engine) and runs the
/// same fallback chain the app uses, so the CLI and the Forecast tab agree.
async fn run_forecast(
    symbol: &str,
    engine: &str,
    horizon: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use bt_analytics::forecast::{Engine, Forecaster};

    if horizon == 0 {
        return Err("horizon must be at least 1".into());
    }
    let preferred = match engine.to_lowercase().as_str() {
        "chronos" => Engine::Chronos,
        "dlinear" => Engine::DLinear,
        "nhits" => Engine::NHits,
        "auto" => Engine::Auto,
        other => return Err(format!("unknown engine {other:?}: use chronos, dlinear, nhits or auto").into()),
    };
    let service = DataService::new()?;
    let series = fetch_series(&service, symbol, RangeArg::Y1).await;
    let closes = series.closes();
    let forecaster = Forecaster::with_default_paths();
    let (values, name) = forecaster
        .predict_with_preference(preferred, &closes, horizon)
        .map_err(|e| format!("forecast failed: {e}"))?;
    println!("engine: {name}");
    println!("symbol: {symbol}  bars: {}  horizon: {horizon}", closes.len());
    for (i, v) in values.iter().enumerate() {
        println!("+{}: {:.2}", i + 1, v);
    }
    Ok(())
}

/// prefs.json path shared with the desktop app (exe-dir `data/`).
fn app_prefs_path() -> std::path::PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join("data").join("prefs.json")
}

fn read_watchlist() -> Vec<String> {
    std::fs::read_to_string(app_prefs_path())
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|v| v.get("watchlist").cloned())
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn write_watchlist(list: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = app_prefs_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut doc: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    doc["watchlist"] = serde_json::to_value(list)?;
    std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}

/// Manage the watchlist the desktop app reads from the same file.
fn run_watchlist(action: WatchlistAction) -> Result<(), Box<dyn std::error::Error>> {
    const MAX: usize = 50;
    let mut list = read_watchlist();
    match action {
        WatchlistAction::Add { symbol } => {
            let s = symbol.trim().to_uppercase();
            if s.is_empty() {
                return Err("empty symbol".into());
            }
            if list.iter().any(|w| w == &s) {
                println!("{s} is already starred");
            } else if list.len() >= MAX {
                return Err(format!("watchlist is full ({MAX} symbols)").into());
            } else {
                list.push(s.clone());
                write_watchlist(&list)?;
                println!("starred {s} ({} of {MAX})", list.len());
            }
        }
        WatchlistAction::Remove { symbol } => {
            let s = symbol.trim().to_uppercase();
            let before = list.len();
            list.retain(|w| w != &s);
            if list.len() == before {
                println!("{s} was not starred");
            } else {
                write_watchlist(&list)?;
                println!("unstarred {s}");
            }
        }
        WatchlistAction::List => {
            if list.is_empty() {
                println!("watchlist is empty — star symbols in the app or with `watchlist add`");
            }
            for s in &list {
                println!("{s}");
            }
        }
    }
    Ok(())
}

/// Backtest SMA-cross over a year of daily closes and print the tape.
async fn run_backtest(
    symbol: &str,
    fast: usize,
    slow: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = DataService::new()?;
    let series = fetch_series(&service, symbol, RangeArg::Y1).await;
    let closes = series.closes();
    let r = bt_analytics::backtest_sma_cross(&closes, fast, slow)
        .ok_or("backtest needs at least 2 closes")?;
    println!("strategy: {}", r.strategy);
    println!("symbol: {symbol}  bars: {}", closes.len());
    println!("total return: {:+.2}%", r.total_return * 100.0);
    println!("buy-and-hold: {:+.2}%", r.buy_hold_return * 100.0);
    println!("sharpe (ann.): {:.2}", r.sharpe);
    println!("max drawdown:  {:.2}%", r.max_drawdown * 100.0);
    println!("exposure:      {:.0}% of bars", r.exposure * 100.0);
    Ok(())
}

/// Screen cached symbols with a technical filter expression and write matches
/// to CSV. Cache-only by design: screening 500 symbols over the network would
/// take minutes and hammer the providers; the report states its coverage so a
/// thin cache reads as thin.
async fn run_screen(
    filter: &str,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use bt_analytics::screener as scr;
    let filters =
        scr::parse_filters(filter).map_err(|e| format!("bad filter: {e}"))?;
    let cache = bt_data::cache::Cache::new(bt_data::default_cache_path())?;
    let mut rows = Vec::new();
    let mut screened = 0usize;
    for (_, ticker, _) in bt_data::COMPANY_LIST {
        if ticker.contains('-') || ticker.contains('=') || ticker.starts_with('^') {
            continue;
        }
        let mut found = None;
        for iv in ["1d", "1wk", "1mo"] {
            if let Ok(Some(candles)) = cache.get_ohlcv(ticker, iv) {
                if candles.len() >= 20 {
                    found = Some(candles);
                    break;
                }
            }
        }
        let Some(candles) = found else { continue };
        screened += 1;
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        rows.push(scr::compute_technicals(ticker, &closes, &volumes));
    }
    let hits = scr::apply_filters(&rows, &filters);
    let mut csv = String::from("symbol,price,rsi,macd_hist,change_5d,change_20d,volume_ratio,sma50_ratio,sma200_ratio,bb_pos,high52_dist,low52_dist\n");
    for r in &hits {
        let f = |k: &str| {
            r.get(k)
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default()
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.symbol,
            f("price"),
            f("rsi"),
            f("macd_hist"),
            f("change_5d"),
            f("change_20d"),
            f("volume_ratio"),
            f("sma50_ratio"),
            f("sma200_ratio"),
            f("bb_pos"),
            f("high52_dist"),
            f("low52_dist"),
        ));
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, csv)?;
    println!(
        "screened {screened} cached symbols, {} match -> {}",
        hits.len(),
        output.display()
    );
    Ok(())
}

/// Export portfolio holdings with cached prices to CSV.
///
/// `--format` is deliberately CSV-only: the xlsx writer is a v4.2 dependency
/// and this command refuses to pretend otherwise.
fn run_export_portfolio(
    portfolio: Option<&std::path::Path>,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use bt_analytics::portfolio as pf;
    use std::collections::HashMap;

    let path = portfolio.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let base = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        base.join("data").join("portfolio.json")
    });
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let holdings: Vec<pf::Holding> = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    pf::validate(&holdings).map_err(|e| format!("invalid holdings: {e}"))?;

    let cache = bt_data::cache::Cache::new(bt_data::default_cache_path())?;
    let mut prices = HashMap::new();
    for h in &holdings {
        for iv in ["1d", "1wk", "1mo"] {
            if let Ok(Some(candles)) = cache.get_ohlcv(&h.symbol, iv) {
                if let Some(last) = candles.last() {
                    prices.insert(h.symbol.clone(), last.close);
                    break;
                }
            }
        }
    }
    let mut csv = String::from("symbol,qty,avg_price,last_price,market_value,pnl,pnl_pct\n");
    for h in &holdings {
        let px = prices.get(&h.symbol).copied().unwrap_or(f64::NAN);
        let (mv, pnl, pct) = if px.is_finite() {
            (
                format!("{:.2}", h.market_value(px)),
                format!("{:+.2}", h.pnl(px)),
                format!("{:+.2}", h.pnl_pct(px)),
            )
        } else {
            ("".to_string(), "".to_string(), "".to_string())
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            h.symbol,
            h.qty,
            h.avg_price,
            if px.is_finite() {
                format!("{px:.2}")
            } else {
                "n/a".to_string()
            },
            mv,
            pnl,
            pct
        ));
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, csv)?;
    println!(
        "exported {} holdings ({} priced) -> {}",
        holdings.len(),
        prices.len(),
        output.display()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let theme: VizTheme = cli.theme.into();

    banner();

    if cli.list_companies {
        list_companies();
        return Ok(());
    }

    match cli.command {
        Some(Commands::Forecast {
            symbol,
            engine,
            horizon,
        }) => return run_forecast(&symbol, &engine, horizon).await,
        Some(Commands::Watchlist { action }) => return run_watchlist(action),
        Some(Commands::Backtest { symbol, fast, slow }) => {
            return run_backtest(&symbol, fast, slow).await
        }
        Some(Commands::Screen { filter, output }) => return run_screen(&filter, &output).await,
        Some(Commands::ExportPortfolio { portfolio, output }) => {
            return run_export_portfolio(portfolio.as_deref(), &output)
        }
        None => {}
    }

    std::fs::create_dir_all(&cli.out_dir)?;
    println!("Rendering into: {}", cli.out_dir.display());
    println!(
        "Symbol: {} | Range: {:?} | Theme: {:?} | Live: {}\n",
        cli.symbol, cli.range, cli.theme, cli.live
    );

    let service = DataService::new()?;
    let start = Instant::now();
    let mut total_rendered = 0usize;
    let mut iteration = 1usize;

    loop {
        let iter_start = Instant::now();
        let rendered = render_all(
            &service,
            &cli.symbol,
            cli.range,
            theme,
            cli.seed,
            &cli.out_dir,
            iteration,
        )
        .await?;
        total_rendered += rendered;
        let iter_elapsed = iter_start.elapsed();

        println!(
            "\n--- Iteration {} completed in {:.2?} ({} charts) ---",
            iteration, iter_elapsed, rendered
        );

        if !cli.live {
            break;
        }

        // Wait 30 seconds
        println!("Waiting 30 seconds for next refresh... (Ctrl+C to stop)");
        sleep(Duration::from_secs(30)).await;
        iteration += 1;
    }

    let elapsed = start.elapsed();
    println!("\n================================================================");
    println!(
        " Total rendered: {} charts in {:.2?}",
        total_rendered, elapsed
    );
    println!(" {APP_NAME} v4 -- Made by {AUTHOR}");
    println!("================================================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_constants_are_present() {
        assert!(!APP_NAME.is_empty());
        assert!(!AUTHOR.is_empty());
        assert!(!TAGLINE.is_empty());
    }

    #[test]
    fn test_range_args() {
        assert_eq!(RangeArg::D1.to_days(), 1);
        assert_eq!(RangeArg::Y1.to_days(), 365);
        assert_eq!(RangeArg::Y5.to_yahoo_range(), "5y");
    }
}
