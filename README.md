# VCP Scanner — NSE India (Standalone)

Detects Mark Minervini's **Volatility Contraction Pattern** (VCP) across NSE-listed stocks.

## Detection Method

**Two-Pass Architecture** — zero missed patterns:

1. **ATR Compression Pre-filter** — eliminates ~70% of stocks in milliseconds (no VCP = no compression)
2. **Rolling Time-Window Contraction Finder** — tries 2T–6T splits with equal & progressive windows, keeps the best-scoring structure

Runs on **daily bars** (not weekly — a 3-week VCP only has 3 weekly bars, not enough to detect 2–4 contractions).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download NSE data first
python main.py --download

# Run live VCP scan
python main.py

# Save signals to database
python main.py --save

# Debug one stock (shows full pattern diagnosis)
python main.py --debug RELIANCE.NS

# Backtest one stock
python main.py --backtest RELIANCE.NS

# Backtest full NSE universe + generate Excel report
python main.py --backtest-all

# Quick test (first 100 symbols)
python main.py --backtest-all --limit 100

# Regenerate Excel report for a saved run
python main.py --report bt_vcp_2026-06-27_abc12345

# List all saved backtest runs
python main.py --list-runs
```

## Configuration

Edit `config.py` to adjust:
- `ACCOUNT_SIZE` — your account size for position sizing
- `RISK_PER_TRADE_PCT` — % of account to risk per trade (default 1%)
- `VCP_MIN_QUALITY_SCORE` — minimum score to emit a signal (default 60)
- `VCP_ATR_COMPRESSION_MAX` — ATR compression threshold (default 0.60 for NSE)
- `VCP_MAX_LAST_WIDTH_PCT` — maximum final contraction width (default 15% for NSE)

Or use environment variables:
```bash
export ACCOUNT_SIZE=2000000
export RISK_PCT=0.5
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
```

## File Structure

```
vcp_scanner/
├── main.py                  ← Entry point (run this)
├── vcp.py                   ← VCP detector (two-pass engine)
├── vcp_scanner.py           ← Live signal runner
├── backtest.py              ← Walk-forward backtester
├── backtest_vcp_report.py   ← Excel report generator (7 sheets)
├── database.py              ← SQLite persistence
├── config.py                ← All settings
├── downloader.py            ← yfinance data download
├── indicators.py            ← ATR, RSI, ADX, RS Rating, SEPA
├── universe.py              ← NSE symbol list
├── logger_utils.py          ← Logging
├── requirements.txt
├── data/
│   ├── daily/               ← Cached price data (Parquet)
│   └── signals/             ← SQLite database
├── reports/                 ← Excel backtest reports saved here
└── logs/                    ← Log files
```

## VCP Quality Score (0–100)

| Factor | Weight | Notes |
|--------|--------|-------|
| Trend template (9 criteria) | 22 pts | Minervini's Stage 2 |
| T-count (3T=8, 4T=10) | 10 pts | More contractions = stronger |
| Width tightening (monotonic) | 18 pts | The defining VCP property |
| Final contraction tightness | 12 pts | ≤3%=12, ≤6%=9, ≤9%=5 |
| Higher lows across sequence | 10 pts | Accumulation signal |
| Volume slope negative | 8 pts | Supply drying up |
| VDU day present | 6 pts | 52-week-low volume day |
| RS rating | 8 pts | ≥90=8, ≥85=6, ≥70=3 |
| Breakout volume surge | 6 pts | 2×=6, 1.5×=5, 1.3×=4 |

## Backtest Report Sheets

1. **Verdict** — pass/fail against objective criteria
2. **Equity Curve** — cumulative R-multiple growth
3. **Score Band Analysis** — does the score predict outcomes?
4. **T-Count Analysis** — does 3T beat 2T? (VCP-specific)
5. **Contraction Analysis** — tighter final = better? (VCP-specific)
6. **Yearly Performance** — regime dependency check
7. **Trade Log** — every trade with all VCP fields

## NSE-Specific Parameters (vs Minervini US values)

| Parameter | US Value | NSE Value | Reason |
|-----------|----------|-----------|--------|
| ATR compression threshold | ≤0.50 | ≤0.60 | Higher baseline volatility |
| Max final contraction | ≤10% | ≤15% | Mid/small-caps wider |
| Buy zone above pivot | 3% | 5% | Wider bid-ask spreads |
| Breakout volume ratio | 1.4× | 1.3× | Lower average liquidity |

Reference: Mark Minervini, *Trade Like a Stock Market Wizard* (2013)
