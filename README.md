```
══════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   ██████╗ ██╗  ██╗ █████╗ ██████╗  █████╗ ████████╗                        ║
║   ██╔══██╗██║  ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝                        ║
║   ██████╔╝███████║███████║██████╔╝███████║   ██║                           ║
║   ██╔══██╗██╔══██║██╔══██║██╔══██╗██╔══██║   ██║                           ║
║   ██████╔╝██║  ██║██║  ██║██║  ██║██║  ██║   ██║                           ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                           ║
║                                                                            ║
║   T E R M I N A L   v 3 . 0 . 0                                            ║
║                                                                            ║
║   Bloomberg power. Zero cost. Made in India.                               ║
║                                                                            ║
══════════════════════════════════════════════════════════════════════════════╝
```

**Author:** Sourish Dey  
**Version:** 3.0.0  
**License:** MIT  
**Platform:** Windows 10/11 (64-bit)  
**Rust Edition:** 2021

---

## 📥 Download

| File | Description | Size |
|------|-------------|------|
| [BharatTerminal-v3.0.0.exe](releases/BharatTerminal-v3.0.0.exe) | Desktop GUI (standalone) | ~8 MB |
| [BharatTerminal-v3.0.0-cli.exe](releases/BharatTerminal-v3.0.0-cli.exe) | Command-line tool | ~7 MB |

> **System Requirements:** Windows 10/11 (64-bit) · 4 GB RAM · 500 MB disk space

---

## 📸 Screenshot

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BHARAT TERMINAL v3  │ RELIANCE.NS ▼ │ 1D 1W 1M 3M 6M 1Y 5Y │ Live │ 🌙 │ 🔄 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Price Action │ Order Flow │ Indicators │ Risk │ Volatility │ India │ ...   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │              📊  INTERACTIVE CHART CANVAS  📊                       │   │
│   │                                                                     │   │
│   │     Candlestick · Heikin-Ashi · Renko · Kagi · Point & Figure      │   │
│   │     Volume Profile · Footprint · Cumulative Delta · Market Profile │   │
│   │     RSI · MACD · Bollinger · Ichimoku · VWAP · ADX · CCI ...       │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ● Live │ Source: Yahoo (RELIANCE.NS) │ Last: 14:32:05 │ Latency: 0ms      │
└─────────────────────────────────────────────────────────────────────────────┘
```

> *Screenshot placeholder — run the app to see the full Bloomberg-style interface.*

---

## 🏷️ Feature Badges

| | | | |
|:--|:--|:--|:--|
| 📈 163 Interactive Tabs | 📊 200+ Visualizations | 🇮🇳 India-Specific | 🔴 Live Mode |
| 📉 Technical Analysis | 📋 F&O Chain Analytics | 💰 G-Sec & Money Market | 🏦 RBI Policy |
| 🌍 Global Markets | ₿ Crypto | 📊 Portfolio Risk | 🔬 Quantitative |
| 📰 News & Research | 🏦 Banking Indicators | 📋 IPO Pipeline | 💹 GST/Budget |
| 🔒 Privacy-First | 💾 SQLite Cache | 🎨 Dark/Light Theme | 🆓 100% Free |

---

## 📋 Table of Contents

1. [Feature Categories (Bloomberg-Style)](#-feature-categories-bloomberg-style)
2. [Data Sources](#-data-sources)
3. [India-Specific Features](#-india-specific-features)
4. [Installation](#-installation)
5. [Project Structure](#-project-structure)
6. [Privacy & Security](#-privacy--security)
7. [License](#-license)

---

## 📊 Feature Categories (Bloomberg-Style)

### 1. Core Price Charting & Technical Formats

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **GP** | Graph Price | Primary charting hub with overlays, split/dividend adjustments |
| **G** | Custom Technical Chart Engine | Multi-pane technical chart suites |
| **GPO** | Graph Price OHLC | OHLC bar charts for volatility discovery |
| **GPC** | Graph Price Candlestick | Japanese candlestick engine |
| **GIP** | Intraday Price Graph | Tick-by-tick, 1-minute, up to 240 days |
| **HP** | Historical Price & Volume | Tabular and visual historical pricing |
| **HS** | Historical Spread/Performance | Side-by-side or normalized ratio charts |
| **COMP** | Comparative Return | Normalized total return from base date |

#### Candlestick Rendering

The `GP — Candlestick` tab renders professional Japanese candles:

| Element | Behaviour |
|:--------|:----------|
| Wick | Thin high-to-low line behind each body |
| Body | Filled open-to-close rectangle, centred and never collapsed |
| Bullish | Green (`#00FF88`) when `close > open` |
| Bearish | Red (`#FF3B3B`) when `close < open` |
| Doji | `open == close` bodies expand to a visible minimum height |
| Trend arrows | Green ▲ below bullish bars, red ▼ above bearish bars (toggleable) |
| Hover | Tooltip with `O / H / L / C / V` snapped to the nearest bar |
| Price axis | Right-hand side, precision chosen from the price magnitude |
| X-axis | Auto-formatted: clock time intraday, `DD Mon` daily, `Mon YYYY` longer |
| Volume | Separate pane sharing the exact same x-range as the price pane |
| Legend | `Last` price plus `O / H / L / C` and total volume above the chart |
| Current price | Dashed horizontal guide at the last close |

**`1M` uses daily bars, so it reads like `3M`.** An earlier build mapped `1M`
to hourly candles, and a month of hourly data is full of overnight and weekend
gaps — the chart rendered as a scatter of thin floating candles instead of a
continuous series. `1M` now uses daily bars, giving the same density and
readability as `3M`, while `1D` and `1W` keep intraday bars where the gaps are
genuinely part of the picture.

**Bar width is derived from the data.** The median gap between consecutive
timestamps sets the body width, so 1-minute, 5-minute, daily and weekly series
all render at the correct density instead of overlapping. Bars are never wider
than the gap that separates them.

**Double-click to zoom in, right-click to zoom out.** Both step one level
(2× in, ½× out) and centre on the pointer, so repeated clicks tighten or widen
the view smoothly instead of snapping. The header shows the current factor
(e.g. `4.0x`) with **🔍−**, **🔍+** and **Reset** buttons for pointer-free
control. Zoom persists across the 30-second live refresh and resets when you
switch symbol or timeframe.

**Drag to pan — on every time-series tab.** Once zoomed in, dragging the chart
moves the view horizontally through time and vertically through price. Panning
is clamped so the view can never be dragged off the data, and a zoom always
snaps onto real candles even if you double-click empty space. Every
candlestick-style tab (`GP — Candlestick`, `GP (HA)`, `C3D`, `CMA`, `CBB`,
`CRSI`, `CMACD`) shares this behaviour through one common gesture handler, so
zooming one tab does not leave the others stuck at a stale zoom level.

**Pinch to zoom, two fingers to move (touch).** On a touchscreen, spread two
fingers to zoom in and pinch them together to zoom out, around the point
between your fingers. Dragging two fingers pans the view horizontally and
vertically at the same time, with the same clamping as a mouse drag. A
horizontal-only pinch zooms time, a vertical-only pinch zooms price, and a
diagonal pinch zooms both. Because a pinch reports a small factor every frame,
zoom is continuous rather than the fixed 2× step used by a double-click, so
the view scales smoothly instead of jumping between extremes.

**Scroll to zoom, drag or swipe to pan (mouse and trackpad).** The wheel works
from the full-range view, so it is how a zoomed window is entered with a
mouse: wheel-up zooms in around the pointer, wheel-down zooms out, and
horizontal scroll (shift+wheel, trackpad swipe) pans through time. Trackpad
pinch and ctrl+wheel arrive as a proportional zoom and scale both axes around
the pointer. Dragging pans horizontally and vertically once zoomed in; the
volume pane answers horizontal drags and scrolls too, so no part of the
canvas feels dead.

**Long custom windows fetch the whole window.** Yahoo's chart endpoint takes
*either* a preset range *or* explicit start/end timestamps -- sending both
makes the preset win and the timestamps vanish, which capped every window
over 5 years at 5 years of data (a 2017-2026 request charted only ~2021 on).
Explicit windows are now requested with timestamps alone, so a 9-year custom
range returns all ~470 weekly bars.

**X-axis labels follow the selected timeframe:**

| Range | Label | Example |
|-------|:------|:--------|
| `1D` | time + date | `09:30 15 Jan` |
| `1W`, `1M` | date + month | `15 Jan` |
| `3M`, `6M`, `1Y`, `5Y` | month + year | `Jan 2024` |
| `Custom` | chosen from the window length | `15 Jan` or `Jan 2024` |

Switching timeframe always re-requests the matching window, even if a fetch is
already running, so the chart can never show one range's data under another
range's label.

**Cached data must actually cover the request.** A cache entry is only reused
when it spans the whole requested window. Without this check a cached 6-month
series satisfied a 1-year query — the range filter passed because those bars
fell inside the year — and `6M` and `1Y` rendered the *same* chart under two
different labels. A few days of slack is allowed so weekends and holidays do
not force a refetch.

**Custom date range.** Alongside `1D`, `1W`, `1M`, `3M`, `6M`, `1Y` and `5Y`,
the header has a **Custom** button that reveals two `YYYY-MM-DD` fields and an
**Apply** button. The window is validated on apply: the dates may be given in
either order, but the span must be at least one day and at most ten years, and
an invalid entry reports why instead of silently falling back to a preset. The
bar interval and the x-axis label style are both chosen from the requested
span, so a three-day window uses 5-minute bars with clock labels while a
two-year window uses daily bars with month labels. The chosen dates are
persisted and restored on the next launch; if they fail to parse on startup the
app falls back to the `1Y` preset rather than starting with a broken range.

**Trend arrows auto-hide above 90 bars**, where a per-bar marker is visual
noise. The header shows `(hidden: N bars)` so the state is never silent; zoom
in to see them.

**The chart fills the window.** The price and volume panes are sized from the
real space left between this tab's own header rows and the bottom of the
window, after reserving room for the x-axis labels and the status bar. The two
panes therefore land exactly on the bottom edge at any window size or DPI
scale, instead of leaving a blank ribbon underneath or pushing the volume pane
off the bottom.

The same renderer backs the `GP (HA)`, `C3D`, `CMA`, `CBB`, `CRSI` and `CMACD`
candle tabs, and all of them resolve their visible range from the same
`ZoomState`, so a previously viewed time range cannot leave them zoomed out.

### Data Source Fallback Chain

| Order | Source | Covers |
|------:|--------|--------|
| 1 | SQLite cache | Previously fetched bars, **only when they span the whole request** |
| 2 | Yahoo Finance | Global equities, indices, ETFs, FX, crypto |
| 3 | Coinbase | `*-USD` crypto spot markets |
| 4 | **NSE bhavcopy** | Indian equities, official daily settlement files |

The **bhavcopy** source downloads the exchange's own end-of-day ZIP from
`archives.nseindia.com` — no API key, no session, no cookie — and parses the
authoritative settlement record. One file contains every listed scrip (about
2,600 for NSE), so a Yahoo outage still yields real prices rather than
synthetic filler. BSE's equivalent `EQ_ISIN_DDMMYY.zip` layout is also parsed.

The fallback is restricted to **daily and coarser** requests. A settlement file
carries one end-of-day bar per scrip, so serving it for a 5-minute request
would return a single daily candle under an intraday label. Sub-daily ranges
now report the failure instead of charting a misleading one-bar series. The
walker also collects **every** trading day in the requested window rather than
stopping at the first day that happens to contain the scrip — a one-bar
fallback charts as a flat line, which is worse than reporting the outage.

```rust
use bt_data::bhavcopy::BhavcopyProvider;
let p = BhavcopyProvider::new()?;
let series = p.fetch_nse_symbol("RELIANCE.NS", date).await?;
```



#### Alternative Bar Styles

| Style | Description |
|:------|:------------|
| Heikin-Ashi | Averaged candlestick smoothing |
| Renko | Brick-based trend filtering |
| Point & Figure | X/O reversal columns |
| Kagi | Thick/thin line reversals |
| Three-Line Break | Breakout-based line charts |
| EquiVolume | Volume-weighted price bars |
| Hollow Candles | Hollow/filled bullish/bearish |
| Range Bars | Fixed-range price bars |
| Dollar Bars | Fixed-dollar-volume bars |
| Tick Bars | Fixed-trade-count bars |
| Imbalance Bars | Order-flow imbalance detection |
| Volume Bars | Fixed-volume aggregation |
| ZigZag | Swing high/low detection |
| Auto S/R | Automatic support/resistance |

#### Technical Overlays

| Overlay | Description |
|:--------|:------------|
| BOLL | Bollinger Bands |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| ADX | Average Directional Index |
| Ichimoku | Ichimoku Cloud |
| Keltner | Keltner Channels |
| Donchian | Donchian Channels |
| ATR | Average True Range |
| Stochastic | Stochastic Oscillator |
| Stoch RSI | Stochastic RSI |
| OBV | On-Balance Volume |
| VWAP | Volume-Weighted Average Price |
| CCI | Commodity Channel Index |
| Williams %R | Williams Percent Range |
| ROC | Rate of Change |
| CMF | Chaikin Money Flow |
| Parabolic SAR | Parabolic Stop and Reverse |
| TRIX | Triple Exponential Average |
| Vortex | Vortex Indicator |
| Aroon | Aroon Indicator |
| Gann | Gann Angles |
| Pivot | Pivot Points |
| Supertrend | Supertrend Indicator |
| MFI | Money Flow Index |
| Williams | Williams %R |
| Bollinger %B | Bollinger Percent B |
| Bollinger Bandwalk | Bandwalk Detection |
| Bollinger Squeeze | Squeeze Detection |
| Bollinger Breakout | Breakout Signals |
| Bollinger Double | Double Bottom/Top |
| Bollinger Mean Reversion | Mean Reversion |
| Bollinger Momentum | Momentum Analysis |
| Bollinger Trend | Trend Analysis |
| Bollinger Volatility | Volatility Analysis |

#### Pattern Recognition

| Pattern | Description |
|:--------|:------------|
| Hammer | Bullish reversal |
| Shooting Star | Bearish reversal |
| Doji | Indecision |
| Engulfing | Bullish/Bearish engulfing |
| Morning Star | Bullish reversal |
| Evening Star | Bearish reversal |

---

### 2. Market Microstructure, Volume & Breadth

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **VAP** | Volume-at-Price | POC, Value Areas, volume shelves |
| **TPO** | Time-Price Opportunity | Market Profile with 30-min brackets |
| **CVD** | Cumulative Volume Delta | Net buying/selling flow |
| **DOM/MDM** | Depth of Market | Order-book ladders |
| **IMAP** | Index Map/Treemaps | Market-cap-sized treemaps |
| **MOV/IMOV** | Index Movers | Stock contribution to index moves |
| **MRR/GRR** | Member Ranked Returns | Ranked gainers/decliners |
| **WEI/WB** | World Equity/Bond Monitors | Multi-panel global dashboards |

#### Order Flow Analytics

| Module | Description |
|:-------|:------------|
| Volume Profile | Horizontal volume histogram |
| Footprint | Bid/ask volume per bar |
| Order Book Heatmap | Depth visualization |
| Cumulative Delta | Buy/sell pressure |
| Market Profile | TPO bracket chart |
| Volume Clock | Time-based volume |
| Tick Tape | Real-time trade feed |
| Delta Divergence | Price vs delta divergence |

---

### 3. Fixed Income, Yield Curves & Structured Finance

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **GC** | Graph Curves | Multi-tenor yield curves |
| **GC3D** | 3D Yield Curve Surface | Term-structure mesh |
| **FWCV/CURV** | Forward Curves | Spot, par, forward rates |
| **YAS** | Yield and Spread Analysis | Duration, convexity, OAS |
| **SPA** | Structured Product Analytics | CMO, ABS, CMBS tranches |
| **WALG** | Weighted Average Life | Principal paydown projections |
| **YT** | Yield Table & Graph | Multi-scenario yield projections |
| **VALL/GALL** | Value All | Dealer bid/ask comparison |
| **DDIS** | Debt Distribution | Maturity-wall bar charts |
| **CAST** | Capital Structure Tree | Debt stack visualization |
| **WIRP** | World Interest Rate Probability | Central bank rate forecasts |

---

### 4. Valuation, Fundamentals & Relative Value

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **GF/GPF** | Graph Fundamentals | Income statement, balance sheet, cash flow |
| **GE** | Valuation Multiple Bands | P/E, EV/EBITDA, P/B corridors |
| **RV** | Relative Valuation | Peer group scatter plots |
| **BETA** | Beta Calculation | Regression vs benchmark |
| **ECO/ECST** | Economic Time Series | Consensus forecasts, actual prints |
| **FXFM/FXFC** | FX Forecasts | Option implied probability distributions |

---

### 5. Derivatives, Volatility & Options

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **VOLS/HIVG** | Volatility Surface & Smile | 2D slices + 3D surfaces |
| **VDAT/VIX** | Volatility Analytics & Cones | Historical vs implied cones |
| **OVME/OVML** | Option Valuation | Multi-leg strategy payoff graphs |

#### Greeks Visualization

| Greek | Description |
|:------|:------------|
| Delta | Price sensitivity |
| Gamma | Delta acceleration |
| Vega | Volatility sensitivity |
| Theta | Time decay |
| Rho | Rate sensitivity |

#### Options Analytics

| Module | Description |
|:-------|:------------|
| Open Interest & Volume Profile | Call vs Put stacked bars |
| IV Surface | 3D implied volatility mesh |
| Term Structure | IV across expiries |
| Vol Cone | Historical vs implied cones |
| Option Payoff | Multi-leg strategy P&L |
| Skew Evolution | IV skew over time |
| Gamma Exposure | Market gamma positioning |
| Put/Call Ratio | Sentiment indicator |
| IV Rank | IV percentile ranking |
| Sharpe Surface | Risk-adjusted return surface |

---

### 6. Quantitative Analysis & Portfolio Risk

| Ticker | Module | Description |
|:------:|:------:|:------------|
| **PORT** | Portfolio & Risk Analytics | Multi-asset portfolio analysis |
| **CMPT/EF** | Efficient Frontier | Markowitz optimization |
| **CORR** | Correlation Matrix | Multi-asset heatmap |

#### Risk Analytics

| Module | Description |
|:-------|:------------|
| Brinson-Fachler Attribution | Waterfall charts |
| Factor Risk Exposure | Value, Growth, Momentum, Size, Quality |
| Drawdown/Underwater Charts | Peak-to-trough declines |
| VaR | Historical and parametric distributions |
| Q-Q Plots | Fat-tail risk assessment |
| ACF/PACF | Serial correlation analysis |
| Monte Carlo | Simulation cloud |
| Rolling Sharpe | Rolling risk-adjusted returns |
| Rolling Sortino | Rolling downside risk |
| Beta/Alpha | CAPM regression |
| Rolling MaxDD | Rolling maximum drawdown |
| VaR Backtest | VaR model validation |
| Hurst | Hurst Exponent |
| Wavelet | Wavelet Analysis |
| Kalman | Kalman Filter |
| Markov Regime | Regime Detection |
| Copula 3D | 3D Copula Visualization |
| Return Dist | Return Distribution |
| Rolling Moments | Rolling Skew/Kurtosis |

---

## 🌐 Data Sources

### Yahoo Finance (Free, No API Key)

Stocks, ETFs, indices, crypto, FX. Direct HTTP with retry logic.

```rust
// crates/bt-data/src/yahoo.rs
use bt_data::{YahooProvider, Interval};
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = YahooProvider::new();
    
    // Fetch OHLCV data
    let end = Utc::now();
    let start = end - chrono::Duration::days(365);
    let series = provider.fetch_ohlcv("RELIANCE.NS", Interval::Day1, start, end).await?;
    
    println!("Fetched {} candles for {}", series.candles.len(), series.symbol);
    
    // Search symbols
    let results = provider.search_symbols("RELIANCE").await?;
    for r in results {
        println!("{} — {} ({})", r.symbol, r.name, r.exchange);
    }
    
    // Fetch company profile
    let profile = provider.fetch_company_profile("AAPL").await?;
    println!("{}: {} | Cap: ${:.0}B", 
        profile.name, profile.sector, profile.market_cap / 1e9);
    
    Ok(())
}
```

### NSE India (Free, Cookie-Based)

F&O chains, Bhavcopy, indices. Cookie-based session management.

```rust
// crates/bt-data/src/india.rs
use bt_data::india::*;

fn main() {
    // F&O Options Chain
    let spot = 24_842.0;
    let chain = sample_options_chain(spot);
    for entry in &chain {
        println!("Strike {} | Call IV: {:.2}% | Put IV: {:.2}% | Call OI: {} | Put OI: {}",
            entry.strike, entry.call_iv * 100.0, entry.put_iv * 100.0,
            entry.call_oi, entry.put_oi);
    }
    
    // IV Surface
    let iv_surface = sample_iv_surface();
    println!("Tenors: {:?} | Strikes: {:?}", iv_surface.tenors, iv_surface.strikes);
    
    // G-Sec Yield Curve
    let gsec = sample_gsec_curve();
    for (t, y) in gsec.tenors.iter().zip(&gsec.yields) {
        println!("{:.2}Y: {:.2}%", t, y);
    }
    
    // Money Market
    let mm = sample_money_market();
    println!("MIBOR 3M: {:.2}% | TREPS: {:.2}% | T-Bill 364D: {:.2}%",
        mm.mibor_3m, mm.treps, mm.t_bill_364);
    
    // RBI Policy
    let rbi = sample_rbi_policy();
    println!("Repo: {:.2}% | CRR: {:.2}% | SLR: {:.2}% | Stance: {}",
        rbi.repo_rate, rbi.crr, rbi.slr, rbi.stance);
    
    // Macro Indicators
    for ind in sample_macro_indicators() {
        println!("{}: {} {} (prev: {})", ind.name, ind.value, ind.unit, ind.prev);
    }
    
    // Commodities
    for c in sample_commodities() {
        println!("{}: ₹{:.2}/{} ({:+.2}%)", c.name, c.price, c.unit, c.change_pct);
    }
    
    // USD/INR Forward Curve
    let usd = sample_usdinr_forward();
    for (t, fwd) in usd.tenors.iter().zip(&usd.forward_points) {
        println!("{}D: {} paise", t, fwd);
    }
    
    // Mutual Funds
    for mf in sample_mf_schemes() {
        println!("{}: NAV ₹{:.2} | AUM ₹{:.0}cr | Sharpe: {:.2}",
            mf.name, mf.nav, mf.aum_cr, mf.sharpe);
    }
    
    // FPI/FII Flows
    for flow in sample_fpi_fii_flows() {
        println!("{}: FPI ₹{:.0}cr | DII ₹{:.0}cr",
            flow.date, flow.fpi_equity, flow.dii);
    }
    
    // Credit Ratings
    for cr in sample_credit_ratings() {
        println!("{}: {} {} ({})", cr.issuer, cr.rating, cr.outlook, cr.action);
    }
    
    // Banking Indicators
    for bi in sample_banking_indicators() {
        println!("{}: {} {}", bi.name, bi.value, bi.unit);
    }
    
    // Corporate Actions
    for ca in sample_corp_actions() {
        println!("{}: {} — {} (Ex: {})", ca.symbol, ca.action, ca.detail, ca.ex_date);
    }
    
    // IPO Pipeline
    for ipo in sample_ipo_pipeline() {
        println!("{}: ₹{:.0}cr | {} | GMP: {:.1}%",
            ipo.company, ipo.issue_size_cr, ipo.status, ipo.gmp);
    }
    
    // India Portfolio with Tax
    for h in sample_india_portfolio() {
        println!("{}: STCG ₹{:.0} | LTCG ₹{:.0} | Tax ₹{:.0}",
            h.symbol, h.stcg, h.ltcg, h.tax_liability);
    }
}
```

### Financial Modeling Prep (Free Tier: 250 req/day)

Fundamentals, financial statements, ratios.

```rust
// FMP API — free tier: 250 requests/day
// Base URL: https://financialmodelingprep.com/api/v3
// Get your free key at: https://financialmodelingprep.com/developer/docs

use reqwest::Client;
use serde::Deserialize;

#[derive(Deserialize)]
struct FMPIncomeStatement {
    symbol: String,
    date: String,
    revenue: f64,
    net_income: f64,
    eps: f64,
    ebitda: f64,
}

async fn fetch_fmp_income(api_key: &str, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let url = format!(
        "https://financialmodelingprep.com/api/v3/income-statement/{}?limit=4&apikey={}",
        symbol, api_key
    );
    let resp = client.get(&url).send().await?;
    let statements: Vec<FMPIncomeStatement> = resp.json().await?;
    for s in &statements {
        println!("{}: Revenue ${:.2}B | EPS ${:.2}", s.date, s.revenue / 1e9, s.eps);
    }
    Ok(())
}
```

### Coinbase (Free)

Crypto spot markets.

```rust
// crates/bt-data/src/coinbase.rs
use bt_data::CoinbaseProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = CoinbaseProvider::new();
    
    // Fetch crypto OHLCV
    let series = provider.fetch_ohlcv("BTC-USD", Interval::Day1).await?;
    println!("BTC: {} candles, last price ${:.2}", 
        series.candles.len(), 
        series.candles.last().unwrap().close);
    
    Ok(())
}
```

### RBI (Free)

Reference rates, monetary policy.

```rust
// RBI data — free public endpoints
// https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
// https://dbie.rbi.org.in/DBIE/dbie.rbi?site=home

// RBI reference rates (daily)
// https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=52867

// Monetary policy statements
// https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=52868
```

### Data Source Summary

| Source | Coverage | Auth | Rate Limit |
|:-------|:---------|:-----|:-----------|
| Yahoo Finance | Stocks, ETFs, Indices, Crypto, FX | None | ~2000/hr |
| NSE India | F&O chains, Bhavcopy, Indices | Cookie-based | Session |
| Financial Modeling Prep | Fundamentals, Statements | API Key | 250/day (free) |
| Coinbase | Crypto spot markets | None | Public |
| RBI | Reference rates, Policy | None | Public |
| SQLite Cache | Local cache with TTL | None | None |

---

## 🇮🇳 India-Specific Features

| Category | Features |
|:---------|:---------|
| **Equity Markets** | NSE/BSE equity data, 78+ companies, real-time quotes |
| **F&O Analytics** | Options chain with IV, OI, Greeks, IV surface, OI heatmap |
| **G-Sec** | Government securities yield curve, multi-segment (SDL, Corporate) |
| **Money Market** | MIBOR, TREPS, T-Bills, CP/CD rates, CBLO, MSF |
| **RBI Policy** | Repo rate, CRR, SLR, stance, inflation target, meeting calendar |
| **Macro Indicators** | CPI, WPI, IIP, GDP, PMI, Core Sector, Trade Deficit, CAD |
| **Commodities** | MCX (Gold, Silver, Crude, Metals), NCDEX (Agri) |
| **USD/INR** | Spot, forward points, forward outright curve |
| **Mutual Funds** | NAV, AUM, category analytics, Sharpe, Beta, Alpha |
| **Corporate Filings** | Corporate actions calendar (dividends, splits, bonus, rights, buyback) |
| **FPI/FII Tracking** | Daily FPI/FII/DII flow data, equity and debt segments |
| **Credit Ratings** | CRISIL, ICRA, CARE, India Ratings — upgrades, downgrades, outlooks |
| **Banking Indicators** | NPA ratios, CRAR, CD ratio, NIM, credit/deposit growth, PCR |
| **IPO Pipeline** | Live/upcoming/listed IPOs with GMP, price bands, issue size |
| **GST/Budget** | Monthly GST collections, Union Budget metrics, fiscal deficit |
| **India Portfolio** | Holdings with LTCG/STCG tax calculation, dividend income |
| **Sector Research** | P/E, P/B, ROE, EPS growth, momentum, outlook by sector |
| **Market Breadth** | Advances/declines, 52w highs/lows, above 50/200 DMA |
| **News & Regulatory** | SEBI/RBI updates, market news with sentiment |
| **Algo Feed** | NSE/BSE multicast feed status, latency, uptime |
| **AI Research** | AI-curated research summaries from top brokers |
| **Dashboard Widgets** | Nifty, Sensex, Bank Nifty, USD/INR, Gold, India VIX, 10Y G-Sec |

---

## 🚀 Installation

### Option 1: Installer (Recommended)

1. Download `BharatTerminal-v3.0.0.msi` from the [releases](bharat-terminal/releases/) folder
2. Double-click it — installs per-user to `%LOCALAPPDATA%\Programs\BharatTerminal`, **no admin prompt**
3. Launch **Bharat Terminal** from the Start Menu (the icon, taskbar and window carry the brand mark)

To uninstall, use Settings → Apps → Bharat Terminal → Uninstall. Your local
cache and preferences survive uninstalling; delete the install folder's
`data/` subfolder if you want those gone too.

### Option 2: Portable (Download .exe)

1. Download `BharatTerminal-v3.0.0.exe` from the [releases](bharat-terminal/releases/) folder
2. Double-click to run — **no installation required**
3. The app creates `./data/` on first run for cache and preferences

### Option 3: Build from Source

```cmd
git clone https://github.com/sourishdey2005/Bharat-terminal-exe.git
cd Bharat-terminal-exe
cargo build --release --workspace
```

Output binaries:
- `target/release/bt-app.exe` — Desktop GUI
- `target/release/bt-cli.exe` — Command-line tool

### Option 4: CLI Usage

```cmd
:: List all supported companies
BharatTerminal-v3.0.0-cli.exe --list-companies

:: Fetch and render 25 visualizations for a symbol
BharatTerminal-v3.0.0-cli.exe --symbol RELIANCE.NS --range 1y --live

:: Custom output directory and theme
BharatTerminal-v3.0.0-cli.exe --symbol AAPL --range 6mo --out-dir output_aapl --theme light

:: Crypto
BharatTerminal-v3.0.0-cli.exe --symbol BTC-USD --range 3m --theme dark
```

#### CLI Options

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--symbol` | Ticker symbol (e.g., RELIANCE.NS, AAPL, BTC-USD) | RELIANCE.NS |
| `--range` | Time range: 1d, 1w, 1m, 3m, 6m, 1y, 5y | 1y |
| `--out-dir` | Output directory for PNGs | output |
| `--theme` | Color theme: dark, light | dark |
| `--seed` | RNG seed for synthetic fallback | 42 |
| `--live` | Auto-refresh every 30 seconds | false |
| `--list-companies` | List all supported companies and exit | false |

---

## 📁 Project Structure

```
BharatTerminal/
├── crates/
│   ├── bt-core/           # Shared types, errors, synthetic data generators
│   ├── bt-data/           # Market data providers
│   │   ├── yahoo.rs      #   Yahoo Finance (stocks, ETFs, indices, crypto)
│   │   ├── india.rs       #   NSE India (F&O, G-Sec, RBI, macro, commodities)
│   │   ├── coinbase.rs    #   Crypto spot markets
│   │   ├── provider.rs    #   DataProvider trait
│   │   ├── symbol.rs      #   Company list (78+ companies)
│   │   └── cache.rs       #   SQLite cache with TTL
│   ├── bt-analytics/      # Technical indicators and risk metrics
│   │   ├── indicators.rs  #   SMA, EMA, RSI, MACD, Bollinger, ADX, ATR, etc.
│   │   └── risk.rs        #   VaR, Sharpe, Sortino, drawdown, beta, etc.
│   ├── bt-viz/            # 200+ visualization modules (203 Rust files)
│   │   ├── candlestick.rs #   Candlestick charts
│   │   ├── candle_*.rs    #   30+ candlestick overlay variants
│   │   ├── bollinger_*.rs #   10 Bollinger Band strategies
│   │   ├── comparison_*.rs #   Multi-stock comparison charts
│   │   ├── india_*.rs     #   India-specific visualizations
│   │   ├── footprint.rs   #   Footprint charts
│   │   ├── vol_smile.rs   #   Volatility smile
│   │   ├── efficient_frontier.rs # Markowitz optimization
│   │   ├── correlation_heatmap.rs # Correlation matrix
│   │   ├── drawdown.rs    #   Underwater charts
│   │   ├── acf_pacf.rs    #   Serial correlation
│   │   ├── seasonality_polar.rs # Seasonality heatmap
│   │   ├── yield_curve.rs #   Yield curve family
│   │   ├── sector_treemap.rs # Market treemap
│   │   ├── nifty_treemap.rs  # Nifty 50 treemap
│   │   ├── sensex_heatmap.rs # Sensex heatmap
│   │   ├── fo_chain.rs    #   F&O options chain
│   │   ├── fii_dii_flow.rs #  FII/DII flow
│   │   ├── india_macro.rs #   India macro dashboard
│   │   ├── india_fx.rs    #   USD/INR analysis
│   │   ├── india_equity.rs #  India equity analytics
│   │   ├── banking_indicators.rs # Banking system
│   │   ├── corporate_actions.rs  # Corp actions calendar
│   │   ├── commodity_dashboard.rs # MCX/NCDEX
│   │   ├── econ_calendar.rs    # Economic calendar
│   │   ├── earnings_calendar.rs # Earnings calendar
│   │   ├── ai_research.rs      # AI research feed
│   │   ├── algo_feed.rs       # Algo feed status
│   │   ├── bse_heatmap.rs     # BSE heatmap
│   │   ├── india_mobility.rs  # India mobility
│   │   ├── copula_3d.rs       # 3D copula
│   │   └── ... (200+ more)
│   ├── bt-cli/            # Command-line interface (25 chart renderer)
│   └── bt-app/            # Desktop GUI (egui/eframe, 163 tabs)
├── releases/              # Downloadable executables
├── docs/                  # Documentation
│   ├── COMPANIES.md       #   Supported companies list
│   ├── INSTALL.md         #   Installation guide
│   ├── VISUALIZATIONS.md  #   Visualization catalog
│   └── screenshots/       #   Screenshot images
├── assets/                # Fonts and icons
│   └── fonts/             #   DejaVu Sans (Regular, Bold, Mono)
├── data/                  # Runtime data (created on first run)
│   ├── cache.db           #   SQLite cache
│   └── prefs.json         #   User preferences
├── Cargo.toml             # Workspace manifest
├── Cargo.lock             # Dependency lock file
├── lib.rs                 # Library root
└── LICENSE                # MIT License
```

### Crate Overview

| Crate | Purpose | Key Dependencies |
|:------|:--------|:-----------------|
| `bt-core` | Shared types, errors, synthetic data | chrono, serde |
| `bt-data` | Market data providers | reqwest, tokio, rusqlite |
| `bt-analytics` | Technical indicators, risk metrics | — |
| `bt-viz` | 200+ visualization modules | plotters, image |
| `bt-cli` | Command-line interface | clap, tokio |
| `bt-app` | Desktop GUI | egui, eframe, egui_plot |

### Module Count

| Metric | Count |
|:-------|:------:|
| Rust source files (bt-viz) | 203 |
| GUI tabs | 163 |
| CLI visualizations | 25 |
| Technical indicators | 30+ |
| Candlestick overlay variants | 30+ |
| Bollinger Band strategies | 10 |
| India-specific modules | 30+ |
| Total crates | 6 |

---

## 🔮 Price Forecasting (Forecast Tab)

The **Forecast** tab under *Advanced* charts the trailing closes against a
predicted continuation, using a three-engine fallback chain. The app never
crashes or hangs when a model is missing — each engine reports availability
up front, failures degrade to the next engine, and the panel always says
which engine actually ran.

| Order | Engine | Needs | Context / Horizon |
|------:|--------|-------|:-----------------:|
| 1 | IBM Granite TTM R2 | `models/ttm-q8.gguf` + `models/config.json` + `zsfm` CLI on `PATH` | 512 / 96 |
| 2 | NanoForecast v0.5 | `models/nanoforecast.onnx` (+ ONNX Runtime for `ort` to dlopen) | 512 / 48 |
| 3 | Auto bench | nothing — pure Rust | any / any |
| 4 | ARIMA(1,1,1) | nothing | any / any |
| 5 | Exp. smoothing (0.3) | nothing | any / any |
| 6 | Moving average (5) | nothing | any / any |

Granite runs through the `zsfm` CLI (`zsfm ttm infer` reading
`{"context": [...], "horizon": N}` on stdin), resolved from `PATH` with a
`~/.cargo/bin` fallback and a 60 s timeout, so a wedged subprocess can never
freeze the app. A live run measures ~0.2 s for 64 points. Each engine paints
its own line colour (Granite green, NanoForecast amber, statistical blue),
so switching engines visibly changes the chart.

Alongside forecasting, the **WatchSignal LSTM** (`models/stock_signal_lstm_v1_seed42.onnx`)
classifies the trailing 30 bars into a single BUY/HOLD/SELL signal with a
confidence readout, shown in the status bar and on the Forecast tab. The
model's true output is one `(sell, hold, buy)` triple (measured, not
assumed), calibrated with the shipped temperature (1.07) when the scores
are logits rather than probabilities. Its 55-feature layout is provisional
(the training order was never published), and both the tab and the code say
so — treat live signals as experimental until the training layout is
confirmed.

The **Prefer** dropdown on the panel selects the starting engine; anything
missing or failing falls down the chain automatically, and the result line
names the engine that actually ran (with a fallback note when it differs
from the preference). `Auto` runs the statistical auto-selector, which
tries ARIMA, then smoothing, then moving average and reports the winner.

Notes worth knowing:

- **Model files live in `models/`** next to the executable (same rule as
  `prefs.json`/`cache.db`), falling back to `./models` for `cargo run`.
  The Granite GGUF (~1 MB) and its `config.json` can be fetched from
  Hugging Face, and the MSI installer ships both, so an installed app
  resolves its weights out of the box — only the `zsfm` CLI remains
  user-supplied. The NanoForecast repo currently ships only
  `model.safetensors`, so its ONNX has to be exported before that engine
  can load — until then the statistical bench carries the forecast,
  honestly labelled.
- **ONNX Runtime is pinned, not probed:** `ort` is pointed at an exact 1.28.0
  `onnxruntime.dll` next to the executable (shipped by the installer,
  `native/` for developers) because blindly loading the OS-resolved DLL
  can pull an incompatible inbox build and crash natively instead of
  erroring. No DLL, no inference — a plain error the chain handles.
- **2 GB RAM discipline:** the ONNX session uses one intra-op and one
  inter-op thread, Level1 graph optimization only, and
  `with_memory_pattern(false)` so the arena allocator cannot pin large
  blocks. The status bar's **RAM readout** (this process's RSS, refreshed
  once a second) makes that verifiable while you work.
- **Horizon slider (5–96 bars)** with a Run button; the first visit
  auto-runs once data is present. Neural engines return up to their fixed
  output window; ARIMA returns exactly the requested horizon.

```rust
// crates/bt-analytics/src/forecast/mod.rs
use bt_analytics::{Forecaster, models_dir};

let f = Forecaster::with_default_paths(); // probes models/ for both files
println!("engine: {}", f.model_name());
let (values, engine) = f.predict_with_engine(&closes, 20)?;
println!("{} forecast {} points", engine, values.len());
```

## 📂 Where Settings and Cache Live

Both are stored in a `data/` folder **next to the executable**, not in the
working directory:

```
<app folder>/
  BharatTerminal-v3.0.0.exe
  data/
    prefs.json    last symbol, range, theme, custom dates
    cache.db      cached OHLCV bars
```

A relative `./data` path resolved against whatever folder you launched from, so
the same build wrote its settings in different places depending on how it was
started — and could not write at all when launched from a read-only location
such as the Desktop. Anchoring to the executable keeps one build's data with
that build.

## 🔒 Privacy & Security

| Feature | Status |
|:--------|:-------|
| Data collection | ❌ **None** — no user data collected |
| Telemetry | ❌ **None** — no tracking or analytics |
| Local storage | ✅ All data stored locally in `./data/` |
| Cache | ✅ SQLite with TTL (5-min intraday, 24h daily) |
| Preferences | ✅ Local JSON file (`./data/prefs.json`) |
| Network | ✅ Direct API calls only — no intermediary servers |
| Error handling | ✅ Graceful fallback to synthetic data — never crashes |

> **Your data stays on your machine. Bharat Terminal does not phone home.**

---

## 📄 License

MIT License

Copyright © 2026 Sourish Dey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Made with ❤️ by Sourish Dey**

**Bharat Terminal v3 — Bloomberg power. Zero cost. Made in India.**
