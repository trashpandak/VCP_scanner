"""
VCP Scanner — Database Layer
SQLite persistence for VCP signals and backtest results.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from config import SIGNALS_DB
from logger_utils import get_logger

log = get_logger("scanner")
DB_PATH = SIGNALS_DB

DDL = """
CREATE TABLE IF NOT EXISTS vcp_signals (
    signal_id              TEXT PRIMARY KEY,
    symbol                 TEXT NOT NULL,
    sector                 TEXT,
    scan_date              TEXT NOT NULL,
    status                 TEXT DEFAULT 'Watching',
    quality_score          REAL,
    t_count                INTEGER,
    pivot_price            REAL,
    buy_zone_high          REAL,
    current_price          REAL,
    stop_loss              REAL,
    target1                REAL,
    target2                REAL,
    risk_per_share         REAL,
    position_size          INTEGER,
    capital_required       REAL,
    rr_ratio               REAL,
    atr                    REAL,
    rs_rating              REAL,
    sepa_score             REAL,
    is_breaking_out        INTEGER DEFAULT 0,
    breakout_volume_ratio  REAL,
    trend_template_ok      INTEGER DEFAULT 0,
    higher_lows_ok         INTEGER DEFAULT 0,
    vdu_day_present        INTEGER DEFAULT 0,
    atr_compression_ratio  REAL,
    final_contraction_width REAL,
    base_duration_weeks    INTEGER,
    prior_uptrend_pct      REAL,
    created_at             TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id              TEXT PRIMARY KEY,
    run_date            TEXT,
    pattern_type        TEXT DEFAULT 'vcp',
    symbols_tested      INTEGER,
    symbols_with_trades INTEGER,
    total_trades        INTEGER,
    win_rate            REAL,
    profit_factor       REAL,
    expectancy          REAL,
    avg_cagr            REAL,
    avg_drawdown        REAL,
    avg_sharpe          REAL,
    avg_hold_days       REAL,
    notes               TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_symbol_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    pattern_type    TEXT DEFAULT 'vcp',
    trades          INTEGER,
    win_rate        REAL,
    profit_factor   REAL,
    cagr_pct        REAL,
    max_drawdown    REAL,
    sharpe          REAL,
    avg_hold        REAL
);

CREATE TABLE IF NOT EXISTS backtest_trade_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    symbol          TEXT,
    pattern_type    TEXT DEFAULT 'vcp',
    entry_date      TEXT,
    exit_date       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    stop_loss       REAL,
    target1         REAL,
    target2         REAL,
    outcome         TEXT,
    rr_realised     REAL,
    hold_days       INTEGER,
    composite_score REAL,
    score_band      TEXT,
    rs_rating       REAL,
    rsi_at_entry    REAL,
    adx_at_entry    REAL,
    sepa_score      REAL,
    t_count         INTEGER,
    window_strategy TEXT,
    base_start_date TEXT,
    base_duration_weeks INTEGER,
    pivot_price     REAL,
    pivot_low       REAL,
    atr_compression_ratio REAL,
    contraction_widths    TEXT,
    contraction_lows      TEXT,
    contraction_durations TEXT,
    contraction_volumes   TEXT,
    final_contraction_width_pct REAL,
    contractions_tightening INTEGER,
    contractions_shortening INTEGER,
    higher_lows_ok          INTEGER,
    volume_slope_negative   INTEGER,
    vdu_day_present         INTEGER,
    breakout_volume_ratio   REAL,
    rs_new_high             INTEGER,
    stop_is_wide            INTEGER,
    trend_template_ok       INTEGER,
    price_above_ma50        INTEGER,
    price_above_ma150       INTEGER,
    price_above_ma200       INTEGER,
    ma150_above_ma200       INTEGER,
    ma200_uptrend           INTEGER,
    ma50_above_ma150        INTEGER,
    pct_above_52wk_low      REAL,
    pct_below_52wk_high     REAL,
    prior_uptrend_pct       REAL,
    market_regime           TEXT,
    active_timeframes       TEXT,
    tf_combo_bonus          REAL,
    vol_compression_score   REAL,
    volume_trend_score      REAL,
    accumulation_score      REAL,
    sepa_score              REAL
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(DDL)
        # Auto-migrate: add columns introduced in v2 if they don't exist yet
        _migrate_add_columns(con)
    log.debug("VCP database initialised at %s", DB_PATH)


def _migrate_add_columns(con) -> None:
    """Add any new columns that don't exist yet (safe to run every startup)."""
    new_cols = {
        "backtest_trade_log": [
            ("market_regime",         "TEXT"),
            ("active_timeframes",     "TEXT"),
            ("tf_combo_bonus",        "REAL"),
            ("vol_compression_score", "REAL"),
            ("volume_trend_score",    "REAL"),
            ("accumulation_score",    "REAL"),
            ("sepa_score",            "REAL"),
        ],
    }
    for table, cols in new_cols.items():
        try:
            existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            for col_name, col_type in cols:
                if col_name not in existing:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    log.debug("Migrated: added %s.%s", table, col_name)
        except Exception as e:
            log.warning("Migration warning for %s: %s", table, e)


def save_vcp_signal(sig) -> None:
    sql = """
        INSERT OR REPLACE INTO vcp_signals (
            signal_id, symbol, sector, scan_date, status, quality_score,
            t_count, pivot_price, buy_zone_high, current_price, stop_loss,
            target1, target2, risk_per_share, position_size, capital_required,
            rr_ratio, atr, rs_rating, sepa_score, is_breaking_out,
            breakout_volume_ratio, trend_template_ok, higher_lows_ok,
            vdu_day_present, atr_compression_ratio, final_contraction_width,
            base_duration_weeks, prior_uptrend_pct
        ) VALUES (
            :signal_id, :symbol, :sector, :scan_date, :status, :quality_score,
            :t_count, :pivot_price, :buy_zone_high, :current_price, :stop_loss,
            :target1, :target2, :risk_per_share, :position_size, :capital_required,
            :rr_ratio, :atr, :rs_rating, :sepa_score, :is_breaking_out,
            :breakout_volume_ratio, :trend_template_ok, :higher_lows_ok,
            :vdu_day_present, :atr_compression_ratio, :final_contraction_width,
            :base_duration_weeks, :prior_uptrend_pct
        )
    """
    from dataclasses import asdict
    row = {
        "signal_id":              sig.signal_id,
        "symbol":                 sig.symbol,
        "sector":                 sig.sector,
        "scan_date":              sig.scan_date.isoformat(),
        "status":                 sig.status,
        "quality_score":          sig.quality_score,
        "t_count":                sig.t_count,
        "pivot_price":            sig.pivot_price,
        "buy_zone_high":          sig.buy_zone_high,
        "current_price":          sig.current_price,
        "stop_loss":              sig.stop_loss,
        "target1":                sig.target1,
        "target2":                sig.target2,
        "risk_per_share":         sig.risk_per_share,
        "position_size":          sig.position_size,
        "capital_required":       sig.capital_required,
        "rr_ratio":               sig.rr_ratio,
        "atr":                    sig.atr,
        "rs_rating":              sig.rs_rating,
        "sepa_score":             sig.sepa_score,
        "is_breaking_out":        int(sig.is_breaking_out),
        "breakout_volume_ratio":  sig.breakout_volume_ratio,
        "trend_template_ok":      int(sig.trend_template_ok),
        "higher_lows_ok":         int(sig.higher_lows_ok),
        "vdu_day_present":        int(sig.vdu_day_present),
        "atr_compression_ratio":  sig.atr_compression_ratio,
        "final_contraction_width": sig.final_contraction_width,
        "base_duration_weeks":    sig.base_duration_weeks,
        "prior_uptrend_pct":      sig.prior_uptrend_pct,
    }
    with _conn() as con:
        con.execute(sql, row)


def save_backtest_run(summary: dict) -> None:
    cols = list(summary.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_str = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO backtest_runs ({col_str}) VALUES ({placeholders})"
    with _conn() as con:
        con.execute(sql, summary)


def save_backtest_symbol_summary(run_id: str, summaries: list[dict]) -> None:
    if not summaries:
        return
    rows = [{"run_id": run_id, **s} for s in summaries]
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_str = ", ".join(cols)
    sql = f"INSERT INTO backtest_symbol_summary ({col_str}) VALUES ({placeholders})"
    with _conn() as con:
        con.executemany(sql, rows)


def save_backtest_trade_log(run_id: str, trades: list[dict]) -> None:
    if not trades:
        return
    rows = [{"run_id": run_id, **t} for t in trades]
    # Use first row keys to build the INSERT
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_str = ", ".join(cols)
    sql = f"INSERT INTO backtest_trade_log ({col_str}) VALUES ({placeholders})"
    with _conn() as con:
        con.executemany(sql, rows)


def get_backtest_runs_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM backtest_runs ORDER BY created_at DESC", con)


def get_backtest_symbol_summary_df(run_id: str) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM backtest_symbol_summary WHERE run_id = ?",
            con, params=(run_id,)
        )


def get_backtest_trade_log_df(run_id: str) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM backtest_trade_log WHERE run_id = ?",
            con, params=(run_id,)
        )


def get_vcp_signals_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM vcp_signals ORDER BY scan_date DESC, quality_score DESC",
            con
        )
