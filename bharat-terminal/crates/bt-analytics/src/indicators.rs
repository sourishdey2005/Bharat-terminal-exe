// crates/bt-analytics/src/indicators.rs
// Author: Sourish Dey

//! Technical indicator computations over OHLCV series.
//! All functions are pure, deterministic, and panic-free.

use bt_core::{Candle, OhlcvSeries};
use std::f64;

/// Simple Moving Average.
pub fn sma(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period {
        return vec![f64::NAN; series.candles.len()];
    }

    let closes: Vec<f64> = series.candles.iter().map(|c| c.close).collect();
    let mut result = vec![f64::NAN; closes.len()];

    let mut sum: f64 = closes[..period].iter().sum();
    result[period - 1] = sum / period as f64;

    for i in period..closes.len() {
        sum += closes[i] - closes[i - period];
        result[i] = sum / period as f64;
    }

    result
}

/// Exponential Moving Average.
pub fn ema(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.is_empty() {
        return vec![f64::NAN; series.candles.len()];
    }

    let closes: Vec<f64> = series.candles.iter().map(|c| c.close).collect();
    let mut result = vec![f64::NAN; closes.len()];

    let alpha = 2.0 / (period as f64 + 1.0);

    // Start with SMA for first period
    if closes.len() >= period {
        let sum: f64 = closes[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..closes.len() {
            result[i] = alpha * closes[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// Relative Strength Index (RSI) using Wilder's smoothing.
pub fn rsi(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period + 1 {
        return vec![f64::NAN; series.candles.len()];
    }

    let closes: Vec<f64> = series.candles.iter().map(|c| c.close).collect();
    let mut result = vec![f64::NAN; closes.len()];

    let mut gains = Vec::with_capacity(closes.len() - 1);
    let mut losses = Vec::with_capacity(closes.len() - 1);

    for i in 1..closes.len() {
        let diff = closes[i] - closes[i - 1];
        if diff >= 0.0 {
            gains.push(diff);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-diff);
        }
    }

    // Initial average gain/loss (simple average)
    let avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        result[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        result[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Wilder's smoothing
    let mut ag = avg_gain;
    let mut al = avg_loss;

    for i in (period + 1)..closes.len() {
        ag = (ag * (period - 1) as f64 + gains[i - 1]) / period as f64;
        al = (al * (period - 1) as f64 + losses[i - 1]) / period as f64;

        if al == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = ag / al;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    result
}

/// MACD: returns (macd_line, signal_line, histogram)
pub fn macd(series: &OhlcvSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema12 = ema(series, 12);
    let ema26 = ema(series, 26);

    let n = series.candles.len();
    let mut macd_line = vec![f64::NAN; n];
    let mut signal_line = vec![f64::NAN; n];
    let mut histogram = vec![f64::NAN; n];

    for i in 0..n {
        if !ema12[i].is_nan() && !ema26[i].is_nan() {
            macd_line[i] = ema12[i] - ema26[i];
        }
    }

    // Signal line is EMA(9) of MACD line
    let alpha = 2.0 / 10.0;
    let mut signal_started = false;

    for i in 0..n {
        if !macd_line[i].is_nan() {
            if !signal_started {
                signal_line[i] = macd_line[i];
                signal_started = true;
            } else {
                signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i - 1];
            }
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    (macd_line, signal_line, histogram)
}

/// Bollinger Bands: returns (middle, upper, lower)
pub fn bollinger(series: &OhlcvSeries, period: usize, k: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = sma(series, period);
    let n = series.candles.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    let closes: Vec<f64> = series.candles.iter().map(|c| c.close).collect();

    for i in period - 1..n {
        if middle[i].is_nan() {
            continue;
        }
        let window = &closes[i + 1 - period..=i];
        let mean = middle[i];
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();
        upper[i] = mean + k * std_dev;
        lower[i] = mean - k * std_dev;
    }

    (middle, upper, lower)
}

/// Average True Range (ATR) using Wilder's smoothing.
pub fn atr(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period + 1 {
        return vec![f64::NAN; series.candles.len()];
    }

    let n = series.candles.len();
    let mut tr = vec![0.0; n];
    let mut result = vec![f64::NAN; n];

    // True Range
    tr[0] = series.candles[0].high - series.candles[0].low;
    for i in 1..n {
        let h = series.candles[i].high;
        let l = series.candles[i].low;
        let pc = series.candles[i - 1].close;
        tr[i] = (h - l).max((h - pc).abs()).max((l - pc).abs());
    }

    // Initial ATR (SMA of first period TRs)
    let sum: f64 = tr[1..=period].iter().sum();
    result[period] = sum / period as f64;

    // Wilder's smoothing
    for i in (period + 1)..n {
        result[i] = (result[i - 1] * (period - 1) as f64 + tr[i]) / period as f64;
    }

    result
}

/// Volume Weighted Average Price (VWAP).
/// Uses typical price * volume / cumulative volume.
pub fn vwap(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut result = vec![f64::NAN; n];
    let mut cum_pv = 0.0;
    let mut cum_vol = 0.0;

    for i in 0..n {
        let c = &series.candles[i];
        let typical = (c.high + c.low + c.close) / 3.0;
        cum_pv += typical * c.volume;
        cum_vol += c.volume;
        if cum_vol > 0.0 {
            result[i] = cum_pv / cum_vol;
        }
    }

    result
}

/// On-Balance Volume (OBV).
pub fn obv(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut result = vec![0.0; n];

    if n == 0 {
        return result;
    }

    result[0] = series.candles[0].volume;

    for i in 1..n {
        let vol = series.candles[i].volume;
        if series.candles[i].close > series.candles[i - 1].close {
            result[i] = result[i - 1] + vol;
        } else if series.candles[i].close < series.candles[i - 1].close {
            result[i] = result[i - 1] - vol;
        } else {
            result[i] = result[i - 1];
        }
    }

    result
}

/// Stochastic Oscillator: returns (%K, %D)
pub fn stochastic(series: &OhlcvSeries, k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
    if k_period == 0 || series.candles.len() < k_period {
        let n = series.candles.len();
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let n = series.candles.len();
    let mut k = vec![f64::NAN; n];

    for i in (k_period - 1)..n {
        let window = &series.candles[i + 1 - k_period..=i];
        let highest_high = window
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = window.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let close = series.candles[i].close;

        if highest_high > lowest_low {
            k[i] = 100.0 * (close - lowest_low) / (highest_high - lowest_low);
        } else {
            k[i] = 50.0;
        }
    }

    // %D is SMA of %K
    let mut d = vec![f64::NAN; n];
    if d_period > 0 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in (k_period - 1)..n {
            if k[i].is_nan() {
                continue;
            }
            sum += k[i];
            count += 1;
            if count == d_period {
                d[i] = sum / d_period as f64;
            } else if count > d_period {
                sum -= k[i - d_period];
                d[i] = sum / d_period as f64;
            }
        }
    }

    (k, d)
}

/// Average Directional Index (ADX) with +DI and -DI.
/// Returns (adx, plus_di, minus_di)
pub fn adx(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if period == 0 || series.candles.len() < period + 1 {
        let n = series.candles.len();
        return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let n = series.candles.len();
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    // Calculate +DM, -DM, TR
    tr[0] = series.candles[0].high - series.candles[0].low;
    for i in 1..n {
        let h = series.candles[i].high;
        let l = series.candles[i].low;
        let ph = series.candles[i - 1].high;
        let pl = series.candles[i - 1].low;
        let pc = series.candles[i - 1].close;

        let up_move = h - ph;
        let down_move = pl - l;

        plus_dm[i] = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        minus_dm[i] = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        tr[i] = (h - l).max((h - pc).abs()).max((l - pc).abs());
    }

    // Wilder's smoothing
    let smooth = |vals: &[f64]| -> Vec<f64> {
        let mut out = vec![f64::NAN; vals.len()];
        if vals.len() <= period {
            return out;
        }
        let sum: f64 = vals[1..=period].iter().sum();
        out[period] = sum / period as f64;
        for i in (period + 1)..vals.len() {
            out[i] = (out[i - 1] * (period - 1) as f64 + vals[i]) / period as f64;
        }
        out
    };

    // Wilder's smoothing over DM/TR. These seed from index 1 because their
    // leading entries are real (0.0) rather than NaN.
    let atr_smooth = smooth(&tr);
    let plus_di_smooth = smooth(&plus_dm);
    let minus_di_smooth = smooth(&minus_dm);

    let mut plus_di = vec![f64::NAN; n];
    let mut minus_di = vec![f64::NAN; n];
    let mut dx = vec![f64::NAN; n];

    for i in period..n {
        if atr_smooth[i] > 0.0 {
            plus_di[i] = 100.0 * plus_di_smooth[i] / atr_smooth[i];
            minus_di[i] = 100.0 * minus_di_smooth[i] / atr_smooth[i];
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 0.0 {
                dx[i] = 100.0 * (plus_di[i] - minus_di[i]).abs() / di_sum;
            }
        }
    }

    // ADX is a Wilder-smoothed DX, but the smoothing has to start where DX
    // actually becomes defined. Reusing `smooth` here would seed from
    // `dx[1..=period]`, and every one of those leading entries is still NaN, so
    // the seed, and therefore the entire ADX line, would be NaN. Start the
    // window at the first defined DX instead.
    let first = period;
    let mut adx = vec![f64::NAN; n];
    if n > first + period - 1 {
        let seed: f64 = dx[first..=first + period - 1].iter().sum();
        adx[first + period - 1] = seed / period as f64;
        for i in (first + period)..n {
            adx[i] = (adx[i - 1] * (period - 1) as f64 + dx[i]) / period as f64;
        }
    }

    (adx, plus_di, minus_di)
}

/// Commodity Channel Index (CCI).
pub fn cci(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period {
        return vec![f64::NAN; series.candles.len()];
    }

    let n = series.candles.len();
    let mut result = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let window = &series.candles[i + 1 - period..=i];
        let typical_prices: Vec<f64> = window
            .iter()
            .map(|c| (c.high + c.low + c.close) / 3.0)
            .collect();
        let mean = typical_prices.iter().sum::<f64>() / period as f64;
        let mean_dev = typical_prices
            .iter()
            .map(|tp| (tp - mean).abs())
            .sum::<f64>()
            / period as f64;

        if mean_dev > 0.0 {
            let tp =
                (series.candles[i].high + series.candles[i].low + series.candles[i].close) / 3.0;
            result[i] = (tp - mean) / (0.015 * mean_dev);
        }
    }

    result
}

/// Williams %R.
pub fn williams_r(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period {
        return vec![f64::NAN; series.candles.len()];
    }

    let n = series.candles.len();
    let mut result = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let window = &series.candles[i + 1 - period..=i];
        let highest_high = window
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = window.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let close = series.candles[i].close;

        if highest_high > lowest_low {
            result[i] = -100.0 * (highest_high - close) / (highest_high - lowest_low);
        } else {
            result[i] = -50.0;
        }
    }

    result
}

/// Rate of Change (ROC).
pub fn roc(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() <= period {
        return vec![f64::NAN; series.candles.len()];
    }

    let n = series.candles.len();
    let mut result = vec![f64::NAN; n];

    for i in period..n {
        let prev = series.candles[i - period].close;
        let curr = series.candles[i].close;
        if prev != 0.0 {
            result[i] = 100.0 * (curr - prev) / prev;
        }
    }

    result
}

/// Chaikin Money Flow (CMF).
pub fn cmf(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    if period == 0 || series.candles.len() < period {
        return vec![f64::NAN; series.candles.len()];
    }

    let n = series.candles.len();
    let mut result = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let window = &series.candles[i + 1 - period..=i];
        let mut mfv_sum = 0.0;
        let mut vol_sum = 0.0;

        for c in window {
            let mfm = if c.high != c.low {
                ((c.close - c.low) - (c.high - c.close)) / (c.high - c.low)
            } else {
                0.0
            };
            let mfv = mfm * c.volume;
            mfv_sum += mfv;
            vol_sum += c.volume;
        }

        if vol_sum > 0.0 {
            result[i] = mfv_sum / vol_sum;
        }
    }

    result
}

/// Keltner Channels: returns (middle, upper, lower)
pub fn keltner(
    series: &OhlcvSeries,
    period: usize,
    atr_mult: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = ema(series, period);
    let atr_vals = atr(series, period);

    let n = series.candles.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in 0..n {
        if !middle[i].is_nan() && !atr_vals[i].is_nan() {
            upper[i] = middle[i] + atr_mult * atr_vals[i];
            lower[i] = middle[i] - atr_mult * atr_vals[i];
        }
    }

    (middle, upper, lower)
}

/// Donchian Channels: returns (upper, middle, lower)
pub fn donchian(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if period == 0 || series.candles.len() < period {
        let n = series.candles.len();
        return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let n = series.candles.len();
    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let window = &series.candles[i + 1 - period..=i];
        let hi = window
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = window.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        upper[i] = hi;
        lower[i] = lo;
        middle[i] = (hi + lo) / 2.0;
    }

    (upper, middle, lower)
}

/// Parabolic SAR.
pub fn parabolic_sar(
    series: &OhlcvSeries,
    af_start: f64,
    af_increment: f64,
    af_max: f64,
) -> Vec<f64> {
    let n = series.candles.len();
    let mut sar = vec![f64::NAN; n];

    if n < 2 {
        return sar;
    }

    let mut is_uptrend = series.candles[1].close > series.candles[0].close;
    let mut ep = if is_uptrend {
        series.candles[1].high
    } else {
        series.candles[1].low
    };
    let mut af = af_start;

    sar[0] = if is_uptrend {
        series.candles[0].low
    } else {
        series.candles[0].high
    };
    sar[1] = sar[0];

    for i in 2..n {
        let prev_sar = sar[i - 1];
        let candle = &series.candles[i];

        sar[i] = prev_sar + af * (ep - prev_sar);

        if is_uptrend {
            sar[i] = sar[i]
                .min(series.candles[i - 1].low)
                .min(series.candles[i - 2].low);
            if candle.low <= sar[i] {
                is_uptrend = false;
                sar[i] = ep;
                ep = candle.low;
                af = af_start;
            } else {
                if candle.high > ep {
                    ep = candle.high;
                    af = (af + af_increment).min(af_max);
                }
            }
        } else {
            sar[i] = sar[i]
                .max(series.candles[i - 1].high)
                .max(series.candles[i - 2].high);
            if candle.high >= sar[i] {
                is_uptrend = true;
                sar[i] = ep;
                ep = candle.high;
                af = af_start;
            } else {
                if candle.low < ep {
                    ep = candle.low;
                    af = (af + af_increment).min(af_max);
                }
            }
        }
    }

    sar
}

/// Heikin-Ashi transformation.
/// Returns (ha_open, ha_high, ha_low, ha_close)
pub fn heikin_ashi(series: &OhlcvSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut ha_open = vec![0.0; n];
    let mut ha_high = vec![0.0; n];
    let mut ha_low = vec![0.0; n];
    let mut ha_close = vec![0.0; n];

    if n == 0 {
        return (ha_open, ha_high, ha_low, ha_close);
    }

    // First candle
    ha_open[0] = (series.candles[0].open + series.candles[0].close) / 2.0;
    ha_close[0] = (series.candles[0].open
        + series.candles[0].high
        + series.candles[0].low
        + series.candles[0].close)
        / 4.0;
    ha_high[0] = series.candles[0].high.max(ha_open[0]).max(ha_close[0]);
    ha_low[0] = series.candles[0].low.min(ha_open[0]).min(ha_close[0]);

    for i in 1..n {
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0;
        ha_close[i] = (series.candles[i].open
            + series.candles[i].high
            + series.candles[i].low
            + series.candles[i].close)
            / 4.0;
        ha_high[i] = series.candles[i].high.max(ha_open[i]).max(ha_close[i]);
        ha_low[i] = series.candles[i].low.min(ha_open[i]).min(ha_close[i]);
    }

    (ha_open, ha_high, ha_low, ha_close)
}

/// Renko brick calculation (fixed brick size).
/// Returns vector of (price, direction) where direction is +1 for up, -1 for down.
pub fn renko(series: &OhlcvSeries, brick_size: f64) -> Vec<(f64, i8)> {
    if brick_size <= 0.0 || series.candles.is_empty() {
        return Vec::new();
    }

    let mut bricks = Vec::new();
    let mut current_price = series.candles[0].close;
    let mut direction = 0i8; // 0 = undefined, 1 = up, -1 = down

    for candle in &series.candles {
        let close = candle.close;

        if direction == 0 {
            if (close - current_price).abs() >= brick_size {
                direction = if close > current_price { 1 } else { -1 };
            }
        }

        while (close - current_price).abs() >= brick_size && direction != 0 {
            current_price += direction as f64 * brick_size;
            bricks.push((current_price, direction));

            // Check for reversal
            let next_price = current_price + direction as f64 * brick_size * 2.0;
            if direction == 1 && close <= next_price - brick_size * 2.0 {
                direction = -1;
            } else if direction == -1 && close >= next_price + brick_size * 2.0 {
                direction = 1;
            }
        }
    }

    bricks
}

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::OhlcvSeries;

    fn test_series() -> OhlcvSeries {
        let candles = vec![
            Candle::new(0.0, 100.0, 102.0, 99.0, 101.0, 1000.0),
            Candle::new(1.0, 101.0, 103.0, 100.0, 102.0, 1100.0),
            Candle::new(2.0, 102.0, 104.0, 101.0, 103.0, 1200.0),
            Candle::new(3.0, 103.0, 105.0, 102.0, 104.0, 1300.0),
            Candle::new(4.0, 104.0, 106.0, 103.0, 105.0, 1400.0),
            Candle::new(5.0, 105.0, 107.0, 104.0, 106.0, 1500.0),
            Candle::new(6.0, 106.0, 108.0, 105.0, 107.0, 1600.0),
            Candle::new(7.0, 107.0, 109.0, 106.0, 108.0, 1700.0),
            Candle::new(8.0, 108.0, 110.0, 107.0, 109.0, 1800.0),
            Candle::new(9.0, 109.0, 111.0, 108.0, 110.0, 1900.0),
        ];
        OhlcvSeries::new("TEST", candles)
    }

    #[test]
    fn test_sma() {
        let s = test_series();
        let res = sma(&s, 3);
        assert_eq!(res.len(), 10);
        assert!(res[0].is_nan());
        assert!(res[1].is_nan());
        assert!((res[2] - 102.0).abs() < 0.01); // (101+102+103)/3
        assert!((res[9] - 109.0).abs() < 0.01); // (108+109+110)/3
    }

    #[test]
    fn test_ema() {
        let s = test_series();
        let res = ema(&s, 3);
        assert_eq!(res.len(), 10);
        assert!(!res[2].is_nan());
        assert!(!res[9].is_nan());
    }

    #[test]
    fn test_rsi() {
        let s = test_series();
        let res = rsi(&s, 3);
        assert_eq!(res.len(), 10);
        // RSI should be between 0 and 100
        for v in &res[3..] {
            assert!(*v >= 0.0 && *v <= 100.0);
        }
    }

    #[test]
    fn test_macd() {
        let s = test_series();
        let (macd_line, signal, hist) = macd(&s);
        assert_eq!(macd_line.len(), 10);
        assert_eq!(signal.len(), 10);
        assert_eq!(hist.len(), 10);
    }

    #[test]
    fn test_bollinger() {
        let s = test_series();
        let (mid, upper, lower) = bollinger(&s, 3, 2.0);
        assert_eq!(mid.len(), 10);
        for i in 2..10 {
            assert!(upper[i] >= mid[i]);
            assert!(lower[i] <= mid[i]);
        }
    }

    #[test]
    fn test_atr() {
        let s = test_series();
        let res = atr(&s, 3);
        assert_eq!(res.len(), 10);
        for v in &res[3..] {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_vwap() {
        let s = test_series();
        let res = vwap(&s);
        assert_eq!(res.len(), 10);
        for v in &res {
            assert!(!v.is_nan());
            assert!(*v > 0.0);
        }
    }

    #[test]
    fn test_obv() {
        let s = test_series();
        let res = obv(&s);
        assert_eq!(res.len(), 10);
        assert_eq!(res[0], 1000.0);
        // All closes are rising, so OBV should keep increasing
        for i in 1..10 {
            assert!(res[i] > res[i - 1]);
        }
    }

    #[test]
    fn test_stochastic() {
        let s = test_series();
        let (k, d) = stochastic(&s, 3, 3);
        assert_eq!(k.len(), 10);
        assert_eq!(d.len(), 10);
        for v in &k[2..] {
            assert!(*v >= 0.0 && *v <= 100.0);
        }
    }

    #[test]
    fn test_adx() {
        let s = test_series();
        let (adx, plus_di, minus_di) = adx(&s, 3);
        assert_eq!(adx.len(), 10);
        for v in &adx[3..] {
            if !v.is_nan() {
                assert!(*v >= 0.0 && *v <= 100.0);
            }
        }
    }

    #[test]
    fn test_cci() {
        let s = test_series();
        let res = cci(&s, 3);
        assert_eq!(res.len(), 10);
    }

    #[test]
    fn test_williams_r() {
        let s = test_series();
        let res = williams_r(&s, 3);
        assert_eq!(res.len(), 10);
        for v in &res[2..] {
            assert!(*v >= -100.0 && *v <= 0.0);
        }
    }

    #[test]
    fn test_roc() {
        let s = test_series();
        let res = roc(&s, 3);
        assert_eq!(res.len(), 10);
        assert!(!res[3].is_nan());
    }

    #[test]
    fn test_cmf() {
        let s = test_series();
        let res = cmf(&s, 3);
        assert_eq!(res.len(), 10);
    }

    #[test]
    fn test_keltner() {
        let s = test_series();
        let (mid, upper, lower) = keltner(&s, 3, 2.0);
        assert_eq!(mid.len(), 10);
    }

    #[test]
    fn test_donchian() {
        let s = test_series();
        let (upper, middle, lower) = donchian(&s, 3);
        assert_eq!(upper.len(), 10);
        for i in 2..10 {
            assert!(upper[i] >= middle[i]);
            assert!(lower[i] <= middle[i]);
        }
    }

    #[test]
    fn test_parabolic_sar() {
        let s = test_series();
        let res = parabolic_sar(&s, 0.02, 0.02, 0.2);
        assert_eq!(res.len(), 10);
    }

    #[test]
    fn test_heikin_ashi() {
        let s = test_series();
        let (o, h, l, c) = heikin_ashi(&s);
        assert_eq!(o.len(), 10);
        for i in 0..10 {
            assert!(h[i] >= o[i] && h[i] >= c[i]);
            assert!(l[i] <= o[i] && l[i] <= c[i]);
        }
    }

    #[test]
    fn test_renko() {
        let s = test_series();
        let bricks = renko(&s, 1.0);
        assert!(!bricks.is_empty());
        for (_, dir) in &bricks {
            assert!(*dir == 1 || *dir == -1);
        }
    }
}
