"""
VCP Scanner — Configuration
All settings in one place. Edit here, nothing else needs changing.
"""

import os
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
DAILY_DIR   = DATA_DIR / "daily"
SIGNALS_DIR = DATA_DIR / "signals"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"

for d in [DAILY_DIR, SIGNALS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SIGNALS_DB = DATA_DIR / "signals" / "vcp_signals.db"

# ─── Data Sources ───────────────────────────────────────────────────────────
NIFTY50_SYMBOL  = "^NSEI"
NIFTY500_SYMBOL = "^CRSLDX"

# ─── Liquidity Filters ──────────────────────────────────────────────────────
MIN_AVG_VOLUME   = 50_000
MIN_PRICE        = 10.0
MAX_PRICE        = 100_000.0
MIN_HISTORY_DAYS = 200

# ─── Account Settings ────────────────────────────────────────────────────────
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "1_000_000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PCT",     "1.0"))

# ─── Technical Indicators ────────────────────────────────────────────────────
RSI_PERIOD  = 14
ATR_PERIOD  = 14
ATR_STOP_MULTIPLIER = 1.5
EMA_TREND   = 200
RS_WEIGHTS  = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

# ─── Scoring Thresholds ──────────────────────────────────────────────────────
SCORE_THRESHOLDS = {
    "elite":       90,
    "very_strong": 80,
    "strong":      70,
    "watch":       60,
}

# ─── Download Settings ───────────────────────────────────────────────────────
BATCH_SIZE               = 50
BATCH_DELAY_SECONDS      = 2.0
TIMEOUT_RETRY_WAIT_SEC   = 30
RATELIMIT_RETRY_WAIT_MIN = 7
MAX_RETRIES              = 5
EXPONENTIAL_BASE         = 2

# ─── Telegram ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"
LOG_FILES  = {
    "scanner":        LOGS_DIR / "scanner.log",
    "download":       LOGS_DIR / "download.log",
    "performance":    LOGS_DIR / "performance.log",
    "error":          LOGS_DIR / "error.log",
    "update":         LOGS_DIR / "update.log",
    "signal_tracker": LOGS_DIR / "signal_tracker.log",
}

# ═══════════════════════════════════════════════════════════════════════════
# VCP — Volatility Contraction Pattern Parameters
# Based on: Mark Minervini (SEPA / VCP) + backtest-validated weights
# Data insight: RS Rating has highest win correlation (0.218)
#               Score band strongly predicts: 70-90 = 45% WR vs 30% below 60
#               Trend template boosts WR by +9.6pp (34% → 43.6%)
#               5T dominates (164/196 trades) — window strategy is working
# ═══════════════════════════════════════════════════════════════════════════

# ── Multi-timeframe detection ─────────────────────────────────────────────
VCP_TIMEFRAMES = ["monthly", "weekly", "daily"]   # all three are scored
VCP_TF_WEIGHTS = {"monthly": 0.20, "weekly": 0.35, "daily": 0.45}
# Confidence multipliers when timeframes align
VCP_TF_COMBO_BONUS = {
    ("monthly", "weekly", "daily"): 1.20,  # all three
    ("weekly",  "daily"):           1.10,  # two higher TF
    ("monthly", "daily"):           1.05,  # uncommon combo
}

# ── Multi-scale lookback windows (daily bars) ────────────────────────────
VCP_LOOKBACK_WINDOWS = [60, 80, 100, 120, 150, 180, 220, 260, 300]
# Equivalent windows for weekly bars (divide by ~5)
VCP_WEEKLY_LOOKBACK_WINDOWS  = [12, 16, 20, 25, 30, 36, 44, 52, 60]
# Equivalent windows for monthly bars (divide by ~21)
VCP_MONTHLY_LOOKBACK_WINDOWS = [6, 8, 10, 12, 14, 18, 22, 26]

# ── Contraction finder ────────────────────────────────────────────────────
VCP_MIN_CONTRACTIONS  = 2
VCP_MAX_CONTRACTIONS  = 6
VCP_MIN_WINDOW_BARS   = 5     # per contraction (daily)
VCP_MAX_WINDOW_BARS   = 60    # per contraction (daily)

# Width rules — adaptive (penalty not rejection)
VCP_MIN_WIDTH_PCT         = 1.5    # below this is noise
VCP_MAX_FIRST_WIDTH_PCT   = 50.0   # above this is a crash
VCP_MAX_LAST_WIDTH_PCT    = 20.0   # raised from 15% (backtest: 9.2% avg final width)
VCP_TIGHTENING_RATIO      = 0.95   # relaxed from 0.92 — allow near-flat steps
VCP_TIGHTENING_TOLERANCE  = 0.05   # allow 5% expansion before penalising

# Higher lows — backtest shows low correlation (0.013), made advisory only
VCP_HIGHER_LOWS_TOLERANCE = 0.08   # 8% — window lows naturally deeper

# ── ATR Compression (Pass 1 pre-filter) ─────────────────────────────────
# Backtest insight: all 196 trades had ATR ratio ≤ 0.60, mean = 0.51
# Hard rejection only if ratio > 0.75 (clearly no compression at all)
VCP_ATR_FAST_PERIOD      = 10
VCP_ATR_BASE_PERIOD      = 130
VCP_ATR_COMPRESSION_MAX  = 0.75   # hard reject above 0.75 (was 0.60 — too strict)
VCP_ATR_GOOD_THRESHOLD   = 0.55   # below this gets full compression score
VCP_ATR_GREAT_THRESHOLD  = 0.40   # below this gets bonus compression score

# ── Volume rules ─────────────────────────────────────────────────────────
VCP_FINAL_VOL_MAX_RATIO    = 0.80
VCP_VDU_TOLERANCE          = 1.30   # within 30% of 52wk low counts as VDU
VCP_BREAKOUT_VOLUME_RATIO  = 1.10   # minimum for scoring bonus (not hard gate)
VCP_BREAKOUT_VOL_MA_PERIOD = 50

# ── Trend Template (SEPA) ─────────────────────────────────────────────────
# Backtest: trend_template=1 → WR 43.6% vs 34.0% — HIGH VALUE signal
# Keep as scoring factor (max 20 pts), not hard gate
VCP_TREND_MA_SHORT          = 50
VCP_TREND_MA_MID            = 150
VCP_TREND_MA_LONG           = 200
VCP_52WK_HIGH_MAX_PCT_BELOW = 35.0   # relaxed from 25%
VCP_52WK_LOW_MIN_PCT_ABOVE  = 15.0   # relaxed from 25%
VCP_MA200_UPTREND_LOOKBACK  = 21

# ── Prior uptrend ─────────────────────────────────────────────────────────
VCP_PRIOR_UPTREND_LOOKBACK_DAYS = 260   # extended from 130 — catch longer bases
VCP_PRIOR_UPTREND_MIN_PCT       = 20.0  # reduced from 30%

# ── Pivot and buy zone ────────────────────────────────────────────────────
VCP_BUY_ZONE_PCT  = 8.0    # extended from 5% for backtest (daily step)
VCP_MAX_STOP_PCT  = 12.0   # relaxed from 8%

# ── RS requirements ───────────────────────────────────────────────────────
# Backtest: RS correlation = 0.218 — HIGHEST of all factors
VCP_RS_MIN       = 50    # soft floor for scoring (not hard gate)
VCP_RS_PREFERRED = 70    # gets full RS score at 70+

# ── Market regime ─────────────────────────────────────────────────────────
VCP_REGIME_BULL_BONUS      =  0.10   # multiply final score up in bull market
VCP_REGIME_BEAR_PENALTY    = -0.15   # multiply final score down in bear market
VCP_REGIME_MA_PERIOD       = 200     # benchmark MA for regime detection
VCP_REGIME_SLOPE_DAYS      = 63      # ~3 months for slope

# ── Scoring weights (validated against backtest data) ─────────────────────
# Weights must sum to 100
VCP_SCORE_WEIGHTS = {
    "prior_uptrend":         18,  # foundation of the pattern
    "contraction_quality":   18,  # tightening sequence quality
    "sepa_template":         20,  # highest WR impact (+9.6pp) — raised
    "rs_rating":             15,  # highest correlation factor (0.218) — raised
    "volume_dryup":          10,  # supply exhaustion signal
    "volatility_compression": 8,  # ATR compression score
    "breakout_quality":       6,  # volume + close position on entry day
    "market_regime":          5,  # bull/bear adjustment
}

# ── Scoring gates ─────────────────────────────────────────────────────────
VCP_MIN_QUALITY_SCORE = 60.0   # live scanner signal gate
VCP_BACKTEST_MIN_SCORE = 50.0  # backtest entry gate (lower to capture more history)

# Score classification (from backtest: 70-90 band is the sweet spot)
VCP_SCORE_BANDS = {
    "elite":        (90, 100),
    "institutional":(80, 90),
    "excellent":    (70, 80),
    "good":         (60, 70),
    "watch":        (50, 60),
    "ignore":       (0,  50),
}

# ── Backtesting ───────────────────────────────────────────────────────────
VCP_MIN_BARS_FOR_SCORING = 260
VCP_WALK_STEP_DAYS       = 1    # daily step — catches every breakout day
