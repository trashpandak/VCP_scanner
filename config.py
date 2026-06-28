"""
VCP Scanner — Configuration
All settings in one place. Edit here, nothing else needs changing.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
DAILY_DIR   = DATA_DIR / "daily"
SIGNALS_DIR = DATA_DIR / "signals"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"

for d in [DAILY_DIR, SIGNALS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SIGNALS_DB = DATA_DIR / "signals" / "vcp_signals.db"

# ─── Data Sources ─────────────────────────────────────────────────────────────
NIFTY50_SYMBOL  = "^NSEI"
NIFTY500_SYMBOL = "^CRSLDX"

# ─── Liquidity Filters ────────────────────────────────────────────────────────
MIN_AVG_VOLUME   = 50_000
MIN_PRICE        = 10.0
MAX_PRICE        = 100_000.0
MIN_HISTORY_DAYS = 200

# ─── Account Settings ─────────────────────────────────────────────────────────
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "1_000_000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PCT",     "1.0"))

# ─── Technical Indicators ────────────────────────────────────────────────────
RSI_PERIOD  = 14
ATR_PERIOD  = 14
ATR_STOP_MULTIPLIER = 1.5
EMA_TREND   = 200
RS_WEIGHTS  = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

# ─── Scoring Thresholds ───────────────────────────────────────────────────────
SCORE_THRESHOLDS = {
    "elite":       90,
    "very_strong": 80,
    "strong":      70,
    "watch":       60,
}

# ─── Download Settings ────────────────────────────────────────────────────────
BATCH_SIZE               = 50
BATCH_DELAY_SECONDS      = 2.0
TIMEOUT_RETRY_WAIT_SEC   = 30
RATELIMIT_RETRY_WAIT_MIN = 7
MAX_RETRIES              = 5
EXPONENTIAL_BASE         = 2

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"
LOG_FILES  = {
    "scanner":     LOGS_DIR / "scanner.log",
    "download":    LOGS_DIR / "download.log",
    "performance": LOGS_DIR / "performance.log",
    "error":       LOGS_DIR / "error.log",
    "update":      LOGS_DIR / "update.log",
    "signal_tracker": LOGS_DIR / "signal_tracker.log",
}

# ─── VCP (Volatility Contraction Pattern) ────────────────────────────────────
# Mark Minervini — "Trade Like a Stock Market Wizard" (2013)
# Detection: ATR compression pre-filter → rolling time-window finder
# Timeframe: DAILY bars

# Trend Template — Minervini's Stage 2 criteria
VCP_TREND_MA_SHORT          = 50
VCP_TREND_MA_MID            = 150
VCP_TREND_MA_LONG           = 200
VCP_52WK_HIGH_MAX_PCT_BELOW = 25.0
VCP_52WK_LOW_MIN_PCT_ABOVE  = 25.0
VCP_MA200_UPTREND_LOOKBACK  = 21

# Prior uptrend
VCP_PRIOR_UPTREND_LOOKBACK_DAYS = 130
VCP_PRIOR_UPTREND_MIN_PCT       = 30.0

# ATR compression pre-filter (Pass 1)
VCP_ATR_FAST_PERIOD     = 10
VCP_ATR_BASE_PERIOD     = 130
VCP_ATR_COMPRESSION_MAX = 0.60   # NSE: 0.60 (US Minervini value: 0.50)

# Rolling window contraction finder (Pass 2)
VCP_LOOKBACK_BARS    = 130
VCP_MIN_WINDOW_BARS  = 5
VCP_MAX_WINDOW_BARS  = 40
VCP_MIN_CONTRACTIONS = 2
VCP_MAX_CONTRACTIONS = 6

# Contraction quality rules
VCP_MIN_WIDTH_PCT         = 2.0
VCP_MAX_FIRST_WIDTH_PCT   = 40.0
VCP_MAX_LAST_WIDTH_PCT    = 15.0   # NSE: 15% (US Minervini: 10%)
VCP_TIGHTENING_RATIO      = 0.92
VCP_HIGHER_LOWS_TOLERANCE = 0.02

# Volume rules
VCP_FINAL_VOL_MAX_RATIO    = 0.75
VCP_VDU_TOLERANCE          = 1.20
VCP_BREAKOUT_VOLUME_RATIO  = 1.30   # NSE: 1.30 (US: 1.40)
VCP_BREAKOUT_VOL_MA_PERIOD = 50

# Pivot / buy zone / stop
VCP_BUY_ZONE_PCT  = 5.0    # NSE: 5% (US: 3%)
VCP_MAX_STOP_PCT  = 8.0

# RS / SEPA
VCP_RS_MIN       = 70
VCP_RS_PREFERRED = 85

# Backtesting
VCP_MIN_BARS_FOR_SCORING = 260
VCP_WALK_STEP_DAYS       = 5

# Live scanner gate
VCP_MIN_QUALITY_SCORE = 60.0
