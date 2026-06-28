"""
VCP Scanner — Backtest Engine
Walk-forward historical simulation for the VCP pattern.
"""

from __future__ import annotations

import gc
import uuid
from datetime import date
from typing import Optional

import pandas as pd

from config import (
    ATR_PERIOD, RS_WEIGHTS, RISK_PER_TRADE_PCT,
    SCORE_THRESHOLDS,
    VCP_MIN_BARS_FOR_SCORING, VCP_WALK_STEP_DAYS, VCP_MAX_STOP_PCT,
)
from database import (
    save_backtest_run, save_backtest_symbol_summary, save_backtest_trade_log,
)
from downloader import load_daily
from indicators import atr as calc_atr, rsi as calc_rsi, adx as calc_adx, rs_rating as calc_rs
from vcp import detect_vcp
from logger_utils import get_logger

log = get_logger("performance")

RISK_FREE_RATE   = 0.065   # RBI repo rate proxy
VCP_MIN_BARS_BT  = VCP_MIN_BARS_FOR_SCORING
VCP_WALK_STEP_BT = VCP_WALK_STEP_DAYS


def backtest_vcp_symbol(
    symbol:    str,
    benchmark: Optional[pd.DataFrame] = None,
) -> tuple[Optional[dict], list[dict]]:
    """
    Walk-forward VCP backtest for one symbol.
    Returns (summary_dict, trade_log_list).

    Walk step: 5 trading days (weekly cadence).
    Entry:  actual close of the confirmed breakout bar.
    Stop:   just below pivot_low (final contraction intraday low).
    T1:     entry × 1.20 (+20%)
    T2:     entry × 1.40 (+40%)
    Dedup:  (base_start_date, round(pivot_price, 1))
    """
    daily = load_daily(symbol)
    if daily is None or len(daily) < VCP_MIN_BARS_BT:
        return None, []

    trade_log:   list[dict] = []
    seen_pivots: set[tuple] = set()
    n = len(daily)

    for idx in range(VCP_MIN_BARS_BT, n, VCP_WALK_STEP_BT):
        as_of_date  = daily.index[idx]
        daily_slice = daily.iloc[:idx + 1]

        try:
            pattern = detect_vcp(symbol, daily_slice, benchmark=benchmark)
        except Exception as e:
            log.debug("VCP detection error %s at %s: %s", symbol, as_of_date, e)
            continue

        if pattern is None or not pattern.is_valid or not pattern.is_breaking_out:
            continue

        dedup_key = (pattern.base_start_date, round(pattern.pivot_price, 1))
        if dedup_key in seen_pivots:
            continue
        seen_pivots.add(dedup_key)

        # Point-in-time entry
        entry_date  = daily_slice.index[-1]
        entry_price = float(daily_slice["Close"].iloc[-1])
        if entry_price < pattern.pivot_price or entry_price > pattern.buy_zone_high:
            entry_price = pattern.pivot_price

        stop    = round(pattern.pivot_low * 0.99, 2)
        target1 = round(entry_price * 1.20, 2)
        target2 = round(entry_price * 1.40, 2)

        if entry_price <= stop:
            continue

        risk_per_share   = entry_price - stop
        natural_stop_pct = (entry_price - stop) / entry_price * 100
        stop_is_wide     = natural_stop_pct > VCP_MAX_STOP_PCT

        # Point-in-time indicators
        try:
            atr_val = float(calc_atr(
                daily_slice["High"], daily_slice["Low"], daily_slice["Close"], ATR_PERIOD
            ).iloc[-1])
            rsi_val = float(calc_rsi(daily_slice["Close"]).iloc[-1])
            adx_val = float(calc_adx(
                daily_slice["High"], daily_slice["Low"], daily_slice["Close"], 14
            )["ADX"].iloc[-1])
            if benchmark is not None:
                bench_c = benchmark["Close"].reindex(
                    daily_slice["Close"].index, method="ffill"
                ).dropna()
                rs_val = calc_rs(daily_slice["Close"], bench_c, RS_WEIGHTS)
            else:
                rs_val = 50.0
        except Exception as e:
            log.debug("VCP indicator error %s at %s: %s", symbol, as_of_date, e)
            continue

        quality_score = pattern.quality_score
        score_band = (
            "elite"       if quality_score >= SCORE_THRESHOLDS["elite"]       else
            "very_strong" if quality_score >= SCORE_THRESHOLDS["very_strong"] else
            "strong"      if quality_score >= SCORE_THRESHOLDS["strong"]      else
            "watch"       if quality_score >= SCORE_THRESHOLDS["watch"]       else
            "below_watch"
        )

        # Simulate forward day-by-day
        future     = daily[daily.index > entry_date]
        outcome    = "open_at_end"
        exit_price = float(future["Close"].iloc[-1]) if len(future) else entry_price
        exit_date  = future.index[-1] if len(future) else entry_date
        hold_days  = len(future)

        for bar_date, bar in future.iterrows():
            if bar["Low"] <= stop:
                outcome, exit_price, exit_date = "stopped_out", stop, bar_date
                hold_days = len(future.loc[:bar_date])
                break
            if bar["High"] >= target2:
                outcome, exit_price, exit_date = "target2_hit", target2, bar_date
                hold_days = len(future.loc[:bar_date])
                break
            if bar["High"] >= target1:
                outcome = "target1_hit_holding"

        rr_realised = round((exit_price - entry_price) / risk_per_share, 3)

        trade_log.append({
            "symbol":               symbol,
            "pattern_type":         "vcp",
            "entry_date":           entry_date.date().isoformat(),
            "exit_date":            (
                exit_date.date().isoformat()
                if hasattr(exit_date, "date") else str(exit_date)
            ),
            "entry_price":          round(entry_price, 2),
            "exit_price":           round(exit_price, 2),
            "stop_loss":            round(stop, 2),
            "target1":              round(target1, 2),
            "target2":              round(target2, 2),
            "outcome":              "target1_hit" if outcome == "target1_hit_holding" else outcome,
            "rr_realised":          rr_realised,
            "hold_days":            int(hold_days),
            "composite_score":      round(quality_score, 1),
            "score_band":           score_band,
            "rs_rating":            round(rs_val, 1),
            "rsi_at_entry":         round(rsi_val, 1),
            "adx_at_entry":         round(adx_val, 1),
            "sepa_score":           0.0,
            "t_count":              pattern.t_count,
            "window_strategy":      pattern.window_strategy_used,
            "base_start_date":      (
                pattern.base_start_date.isoformat()
                if pattern.base_start_date else ""
            ),
            "base_duration_weeks":  pattern.base_duration_weeks,
            "pivot_price":          round(pattern.pivot_price, 2),
            "pivot_low":            round(pattern.pivot_low, 2),
            "atr_compression_ratio": round(pattern.atr_compression_ratio, 3),
            "contraction_widths":   str([c.width_pct for c in pattern.contractions]),
            "contraction_lows":     str([c.low       for c in pattern.contractions]),
            "contraction_durations": str([c.duration_bars for c in pattern.contractions]),
            "contraction_volumes":  str([round(c.volume_ratio, 2) for c in pattern.contractions]),
            "final_contraction_width_pct": (
                pattern.contractions[-1].width_pct if pattern.contractions else 0.0
            ),
            "contractions_tightening":  int(pattern.contractions_tightening),
            "contractions_shortening":  int(pattern.contractions_shortening),
            "higher_lows_ok":           int(pattern.higher_lows_ok),
            "volume_slope_negative":    int(pattern.volume_slope_negative),
            "vdu_day_present":          int(pattern.vdu_day_present),
            "breakout_volume_ratio":    round(pattern.breakout_volume_ratio, 2),
            "rs_new_high":              int(pattern.rs_new_high),
            "stop_is_wide":             int(stop_is_wide),
            "trend_template_ok":    int(pattern.trend_template_ok),
            "price_above_ma50":     int(pattern.price_above_ma50),
            "price_above_ma150":    int(pattern.price_above_ma150),
            "price_above_ma200":    int(pattern.price_above_ma200),
            "ma150_above_ma200":    int(pattern.ma150_above_ma200),
            "ma200_uptrend":        int(pattern.ma200_uptrend),
            "ma50_above_ma150":     int(pattern.ma50_above_ma150),
            "pct_above_52wk_low":   pattern.pct_above_52wk_low,
            "pct_below_52wk_high":  pattern.pct_below_52wk_high,
            "prior_uptrend_pct":    pattern.prior_uptrend_pct,
        })

    if not trade_log:
        return None, []

    rr_list = [t["rr_realised"] for t in trade_log]
    wins    = [r for r in rr_list if r > 0]
    losses  = [r for r in rr_list if r <= 0]
    win_r   = len(wins) / len(rr_list)
    gp      = sum(wins)
    gl      = abs(sum(losses))
    pf      = min(gp / gl, 999.0) if gl > 0 else 999.0

    risk_fraction = RISK_PER_TRADE_PCT / 100.0
    equity   = pd.Series([1.0] + [r * risk_fraction for r in rr_list]).add(1).cumprod()
    years    = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr     = (equity.iloc[-1] ** (1 / max(years, 0.1)) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)
    rr_s     = pd.Series(rr_list)
    std      = rr_s.std() if len(rr_s) > 1 else None
    sharpe   = (
        float((rr_s.mean() - RISK_FREE_RATE / 252) / std)
        if std and std > 1e-6 else 0.0
    )

    summary = {
        "symbol":        symbol,
        "pattern_type":  "vcp",
        "trades":        len(rr_list),
        "win_rate":      round(win_r * 100, 1),
        "profit_factor": round(pf, 2),
        "cagr_pct":      round(cagr, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "avg_hold":      round(sum(t["hold_days"] for t in trade_log) / len(trade_log), 1),
    }
    return summary, trade_log


def backtest_universe_vcp(symbols: list[str], notes: str = "") -> dict:
    """
    Run backtest_vcp_symbol() across all symbols.
    Saves results to DB and returns a run summary dict.
    """
    run_id = f"bt_vcp_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    log.info("VCP universe backtest START — run_id=%s  symbols=%d", run_id, len(symbols))

    bench = load_daily("^NSEI")
    if bench is None:
        log.warning("No ^NSEI benchmark — RS defaults to 50.0")

    per_symbol: list[dict] = []
    all_trades: list[dict] = []
    pending_sym:    list[dict] = []
    pending_trades: list[dict] = []
    FLUSH_EVERY = 300

    for i, sym in enumerate(symbols, 1):
        if i % 100 == 0:
            log.info("  VCP backtest: %d/%d  trades=%d", i, len(symbols), len(all_trades))
        try:
            summary, trades = backtest_vcp_symbol(sym, benchmark=bench)
            if summary:
                per_symbol.append(summary)
                all_trades.extend(trades)
                pending_sym.append(summary)
                pending_trades.extend(trades)
        except Exception as e:
            log.debug("VCP backtest error %s: %s", sym, e)

        if i % FLUSH_EVERY == 0 or i == len(symbols):
            if pending_sym or pending_trades:
                try:
                    save_backtest_symbol_summary(run_id, pending_sym)
                    save_backtest_trade_log(run_id, pending_trades)
                    log.debug("Flushed %d summaries / %d trades at %d/%d",
                              len(pending_sym), len(pending_trades), i, len(symbols))
                except Exception as e:
                    log.error("Flush failed at %d: %s", i, e)
                pending_sym    = []
                pending_trades = []
            gc.collect()

    if not per_symbol:
        log.warning("VCP backtest: zero trades across %d symbols", len(symbols))
        s = {
            "run_id": run_id, "run_date": date.today().isoformat(),
            "pattern_type": "vcp",
            "symbols_tested": len(symbols), "symbols_with_trades": 0,
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_cagr": 0.0, "avg_drawdown": 0.0,
            "avg_sharpe": 0.0, "avg_hold_days": 0.0, "notes": notes,
        }
        save_backtest_run(s)
        return s

    df       = pd.DataFrame(per_symbol)
    trades_df = pd.DataFrame(all_trades)
    rr        = trades_df["rr_realised"]

    summary = {
        "run_id":              run_id,
        "run_date":            date.today().isoformat(),
        "pattern_type":        "vcp",
        "symbols_tested":      len(symbols),
        "symbols_with_trades": len(per_symbol),
        "total_trades":        int(df["trades"].sum()),
        "win_rate":            round(float((rr > 0).mean() * 100), 2),
        "profit_factor":       round(
            min(rr[rr > 0].sum() / rr[rr < 0].abs().sum(), 999.0)
            if rr[rr < 0].abs().sum() > 0 else 999.0, 2
        ),
        "expectancy":          round(float(rr.mean()), 3),
        "avg_cagr":            round(float(df["cagr_pct"].mean()), 2),
        "avg_drawdown":        round(float(df["max_drawdown"].mean()), 2),
        "avg_sharpe":          round(
            float(df["sharpe"].dropna().mean()) if df["sharpe"].notna().any() else 0.0, 3
        ),
        "avg_hold_days":       round(float(df["avg_hold"].mean()), 1),
        "notes":               notes,
    }

    save_backtest_run(summary)
    log.info(
        "VCP backtest DONE — run_id=%s  trades=%d  WR=%.1f%%  PF=%.2f  E=%.3fR",
        run_id, summary["total_trades"], summary["win_rate"],
        summary["profit_factor"], summary["expectancy"],
    )
    return summary
