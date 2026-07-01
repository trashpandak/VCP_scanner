# VCP Backtest & Scanner — Full Diagnostic Report
# Date: 2026-06-29

## EXECUTIVE SUMMARY

Good news: The scanner IS working. Multi-timeframe detection is firing correctly
(daily + weekly + monthly combinations all visible in scanner.log). ABSLAMC.NS
detected with 100% win rate. 64/100 symbols found trades (1,049 total).

However there are 3 critical issues and several secondary issues.

---

## ISSUE 1: PERFORMANCE TOO SLOW (most urgent)

### What you see
Scanner.log shows ADANIPOWER.NS generating 200+ detection entries across
~2 minutes (12:52 to 12:54). Just ONE stock is taking 2 minutes.
With 500+ NSE symbols, that's 16+ hours per full scan.

### Root cause: O(N²) scanning inside the backtest
The backtest uses VCP_WALK_STEP_DAYS=1 (daily step) and tries
9 lookback windows × 6 T-counts × 2 strategies = 108 combinations per day.
For a stock with 5 years of history (1,250 days), that's 1,250 × 108 = 135,000
window evaluations per symbol. This is the source of the 2-minute-per-stock lag.

### Fix required in config.py
```python
VCP_WALK_STEP_DAYS = 5          # back to weekly step for backtest
VCP_LOOKBACK_WINDOWS = [80, 130, 220]   # reduce from 9 to 3 key windows
VCP_WEEKLY_LOOKBACK_WINDOWS = [16, 26, 44]
VCP_MONTHLY_LOOKBACK_WINDOWS = [8, 13, 22]
```

For the LIVE daily scan (not backtest), keep trying more windows — but
cache the result per symbol so it's not recalculated multiple times.

---

## ISSUE 2: DAILY SCAN NOT PRODUCING EXCEL OUTPUT

### What you see
vcp_signals table has 0 rows. Trade log CSV is empty.

### Root cause: Two separate problems
**A) The daily scan (python main.py) needs --save flag to write to DB**
If you ran `python main.py` without `--save`, signals are printed to console
only — never saved to DB, never exported to CSV.

**B) There is no Excel output for the LIVE daily scan at all**
The backtest generates an Excel report (backtest_vcp_report.py).
The daily scan has NO equivalent Excel report — it just prints a table
to the console and optionally saves to SQLite.

### Fix: Add a daily scan Excel report
Need to create `vcp_daily_report.py` that reads from vcp_signals table
and produces a clean Excel file with today's signals, ranked by score.
Must be called from main.py after each scan.

---

## ISSUE 3: BACKTEST QUALITY — WIN RATE 28.3% IS TOO LOW

### From the xlsx backtest (196 trades, the earlier run):
- Win rate: 36.7% (acceptable)
- PF: 3.65 (GOOD — asymmetric payoff working)
- Expectancy: +1.678R (EXCELLENT)
- But: 2007=0%, 2011=0%, 2013=0%, 2018=0%, 2019=0% — bear years killing it

### From the new backtest (1,049 trades, 100 symbols):
- Win rate: 28.3% (too low)
- PF: 1.78
- Expectancy: +0.559R

### Root cause: No market regime filter at entry
The scanner detects VCPs even in bear markets (2007, 2011, 2013, 2018, 2019
all show 0% win rates). The market_regime field is being computed but NOT
used as an entry gate in the backtest — it only adjusts the score.

### Fix required in backtest.py
Add regime check BEFORE entering a trade:
```python
# Skip entry in bear markets
if pattern.market_regime == "bear":
    continue
```

### Secondary scoring issue
Score correlation with win = 0.109 (weak). The "strong" band (70-79) achieves
45.1% WR vs "watch" (60-69) at 36% — score IS discriminating but weakly.
The "below_watch" (<60) band still has 30.9% WR — too close to higher bands.
Need to raise VCP_BACKTEST_MIN_SCORE from 50 to 60.

---

## MULTI-TIMEFRAME DETECTION: IS IT WORKING?

### Answer: YES — but with a deduplication bug

From scanner.log, ADANIPOWER.NS correctly shows:
- `TFs=['daily']` — daily only
- `TFs=['daily', 'weekly']` — weekly + daily
- `TFs=['daily', 'weekly', 'monthly']` — all three timeframes

This confirms multi-TF detection is working. However, the log shows the
SAME stock being detected dozens of times at slightly different lookback
windows. The backtest dedup key is:
  `(base_start_date, round(pivot_price, 1))`

Two different lookback windows can produce slightly different pivot prices
on the same base → same trade entered twice at slightly different prices.
This inflates trade count and distorts statistics.

### Fix required in backtest.py
Tighten dedup key to use week of base_start_date:
```python
# Round base_start_date to nearest week to collapse nearby detections
from datetime import timedelta
base_week = pattern.base_start_date - timedelta(days=pattern.base_start_date.weekday())
dedup_key = (base_week, round(pattern.pivot_price / 10) * 10)
```

---

## T-COUNT DISTRIBUTION PROBLEM

From xlsx: 5T=164, 4T=17, 6T=15 — 5T dominates 84% of all trades.
This means the window algorithm is almost always finding 5 contractions.
The score is penalising 6T (overextended) correctly but 5T may be dominating
because the 9-window lookback combinations happen to find 5 windows most cleanly
when divided. This isn't wrong but reduces discriminative value of T-count.

### Fix in config.py
```python
VCP_MIN_CONTRACTIONS = 2
VCP_MAX_CONTRACTIONS = 5   # cap at 5, remove 6T entirely
```
Minervini rarely discusses 5T+ VCPs — 2T/3T/4T are the core patterns.

---

## SUMMARY OF ALL FIXES NEEDED

| Priority | File | Change | Impact |
|----------|------|--------|--------|
| CRITICAL | config.py | Reduce lookback windows from 9 to 3 | 10x speed |
| CRITICAL | config.py | Walk step back to 5 days | 5x speed |
| CRITICAL | main.py | Auto-save signals + generate Excel daily report | Daily scan output |
| HIGH | backtest.py | Skip bear market entries | +8pp win rate |
| HIGH | backtest.py | Tighten dedup key | Accurate stats |
| HIGH | config.py | VCP_BACKTEST_MIN_SCORE = 60 | Quality gate |
| MEDIUM | config.py | VCP_MAX_CONTRACTIONS = 5 | Better T-count dist |
| LOW | vcp.py | Cache indicator calculations per symbol | Minor speed |
