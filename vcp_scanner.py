"""
NSE Darvas Box Scanner - VCP Signal Scanner
============================================
Runs the VCP detector across the full NSE universe and generates
ranked, actionable signals with:
  • RS Rating (cross-sectional percentile vs Nifty 500)
  • SEPA score (Minervini Stage 2 template)
  • ATR-based position sizing and stop loss
  • Trend template breakdown (9 individual criteria)
  • VCP-specific quality scoring
  • Near-pivot, watching, and breaking-out status

Mirrors cup_handle_scanner.py exactly in structure, naming, and
database/Telegram output contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from config import (
    ACCOUNT_SIZE, ATR_PERIOD, ATR_STOP_MULTIPLIER, EMA_TREND,
    MIN_AVG_VOLUME, RISK_PER_TRADE_PCT, RS_WEIGHTS,
    SCORE_THRESHOLDS, VCP_MIN_QUALITY_SCORE, VCP_WATCHLIST_MIN_SCORE, VCP_RS_MIN,
    VCP_BUY_ZONE_PCT,
)
from vcp import VCPPattern, detect_vcp
from downloader import load_daily, resample_weekly
from indicators import (
    atr as calc_atr, ema, rs_rating as calc_rs,
    sepa_score as calc_sepa, trend_label,
)
from logger_utils import get_logger

log = get_logger("scanner")


@dataclass
class VCPSignal:
    # ── Identity ──────────────────────────────────────────────────────────────
    symbol:     str
    sector:     str
    scan_date:  date

    # ── VCP geometry ──────────────────────────────────────────────────────────
    t_count:                 int
    base_start_date:         date
    base_duration_weeks:     int
    contraction_widths:      list     # [c.width_pct for c in pattern.contractions]
    contraction_lows:        list     # [c.low       for c in pattern.contractions]
    final_contraction_width: float
    higher_lows_ok:          bool
    vdu_day_present:         bool
    volume_slope_negative:   bool
    atr_compression_ratio:   float
    window_strategy:         str

    # ── Pivot / entry ─────────────────────────────────────────────────────────
    pivot_price:             float
    buy_zone_high:           float
    current_price:           float
    stop_loss:               float
    target1:                 float    # entry × 1.20
    target2:                 float    # entry × 1.40
    risk_per_share:          float
    position_size:           int
    capital_required:        float
    rr_ratio:                float
    atr:                     float

    # ── Breakout state ────────────────────────────────────────────────────────
    is_breaking_out:         bool
    is_extended:             bool
    breakout_volume_ratio:   float
    rs_new_high:             bool

    # ── Trend template ────────────────────────────────────────────────────────
    trend_template_ok:       bool
    trend_template_details:  dict

    # ── Context ───────────────────────────────────────────────────────────────
    rs_rating:               float
    sepa_score:              float
    sepa_checks:             dict
    prior_uptrend_pct:       float
    pct_below_52wk_high:     float
    weekly_trend:            str   = "neutral"
    market_regime:           str   = "unknown"
    active_timeframes:       list  = None
    tf_combo_bonus:          float = 1.0
    vol_compression_score:   float = 0.0
    score_breakdown:         dict  = None

    # ── Scoring ────────────────────────────────────────────────────────────────
    quality_score:           float = 0.0
    signal_id:               str   = ""
    status:                  str   = "Watching"
    diagnostic_summary:      str   = ""   # one-line summary for logging

    def __post_init__(self):
        self.signal_id = f"VCP_{self.symbol}_{self.scan_date.isoformat()}"
        self.status = (
            "Breaking Out" if self.is_breaking_out else
            "Near Pivot"   if self.current_price >= self.pivot_price * 0.97 else
            "Watching"
        )


def scan_vcp(
    symbol:      str,
    daily:       pd.DataFrame,
    benchmark:   pd.DataFrame,
    sector:      str = "Unknown",
    rs_override: Optional[float] = None,
) -> Optional[VCPSignal]:
    """
    Full VCP scan for one symbol. Returns VCPSignal or None.

    *rs_override* — cross-sectional RS Rating computed by main.py's
    universe-wide Pass 1. When provided, used instead of the per-symbol
    approximation, matching how scan_symbol() (Darvas) and
    scan_cup_handle() (C&H) handle this. Without it, Darvas and VCP
    signals for the same stock on the same day could show inconsistent
    RS Ratings from different methodologies.

    NOTE on live vs backtest behaviour:
    This function does NOT require pattern.is_breaking_out — it surfaces
    "Watching" and "Near Pivot" signals for advance notice.
    backtest_vcp_symbol() DOES hard-require is_breaking_out so only
    realised, tradeable breakouts are counted in win-rate numbers.
    Do not make these consistent — their different behaviours are correct
    for their different purposes.
    """
    if daily is None or len(daily) < 200:
        return None

    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    pattern = detect_vcp(symbol, daily, benchmark=benchmark, rs_rating=float(rs or 50.0))
    if pattern is None:
        return None
    # Return signal at ANY score above watchlist threshold — caller (run_scan)
    # decides whether it goes into signals or watchlist bucket.
    # Using VCP_WATCHLIST_MIN_SCORE as the return gate, not VCP_MIN_QUALITY_SCORE.
    if pattern.quality_score < VCP_WATCHLIST_MIN_SCORE:
        log.debug("%s: VCP quality %.1f below watchlist threshold %.1f",
                  symbol, pattern.quality_score, VCP_WATCHLIST_MIN_SCORE)
        return None

    current_price = float(daily["Close"].iloc[-1])

    # Only surface signals where price is within or just below the pivot
    # (at most 15% below = still forming / approaching)
    if current_price < pattern.pivot_price * 0.85:
        log.debug("%s: price %.2f too far below pivot %.2f",
                  symbol, current_price, pattern.pivot_price)
        return None

    # 200-day EMA — used for scoring via SEPA (not a hard gate here;
    # SEPA already penalises stocks below EMA200 in the quality score).
    # Hard-gating here AND in SEPA was causing 0 daily signals in sideways markets.
    ema200 = float(ema(daily["Close"], EMA_TREND).iloc[-1])
    current_price_val = float(daily["Close"].iloc[-1])
    if current_price_val < ema200 * 0.85:   # only hard-block if >15% BELOW EMA200
        log.debug("%s: price %.2f is >15%% below 200 EMA %.2f — skipping", symbol, current_price_val, ema200)
        return None

    # RS Rating
    if rs_override is not None:
        rs = rs_override
    else:
        try:
            bench_close = benchmark["Close"].reindex(daily.index, method="ffill").dropna()
            rs = calc_rs(daily["Close"], bench_close, RS_WEIGHTS)
        except Exception:
            rs = 50.0

    # RS gate: only hard-block very weak RS (below 30). 
    # VCP_RS_MIN (50) is now used as a scoring threshold only.
    RS_HARD_GATE = 30
    if rs < RS_HARD_GATE:
        log.debug("%s: RS %.1f below hard gate %d", symbol, rs, RS_HARD_GATE)
        return None

    # SEPA
    try:
        sepa, sepa_checks = calc_sepa(daily["Close"])
    except Exception:
        sepa, sepa_checks = 0.0, {}

    # ATR-based stop loss below pivot_low
    atr_val   = float(calc_atr(daily["High"], daily["Low"], daily["Close"], ATR_PERIOD).iloc[-1])
    stop      = round(pattern.pivot_low - ATR_STOP_MULTIPLIER * atr_val, 2)
    entry     = current_price if pattern.is_breaking_out else pattern.pivot_price
    risk_ps   = entry - stop
    if risk_ps <= 0:
        return None

    target1     = round(entry * 1.20, 2)
    target2     = round(entry * 1.40, 2)
    risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    pos_size    = max(1, int(risk_amount / risk_ps))
    cap_req     = round(pos_size * entry, 2)
    rr_ratio    = round((target2 - entry) / risk_ps, 2)

    # Weekly trend label
    try:
        weekly    = resample_weekly(daily)
        w_trend   = trend_label(weekly["Close"]) if len(weekly) > 30 else "neutral"
    except Exception:
        w_trend = "neutral"

    pattern.rs_rating = round(rs, 1)

    sig = VCPSignal(
        symbol                   = symbol,
        sector                   = sector,
        scan_date                = date.today(),
        t_count                  = pattern.t_count,
        base_start_date          = pattern.base_start_date,
        base_duration_weeks      = pattern.base_duration_weeks,
        contraction_widths       = [c.width_pct for c in pattern.contractions],
        contraction_lows         = [c.low       for c in pattern.contractions],
        final_contraction_width  = pattern.contractions[-1].width_pct if pattern.contractions else 0.0,
        higher_lows_ok           = pattern.higher_lows_ok,
        vdu_day_present          = pattern.vdu_day_present,
        volume_slope_negative    = pattern.volume_slope_negative,
        atr_compression_ratio    = pattern.atr_compression_ratio,
        window_strategy          = pattern.window_strategy_used,
        pivot_price              = pattern.pivot_price,
        buy_zone_high            = pattern.buy_zone_high,
        current_price            = round(current_price, 2),
        stop_loss                = stop,
        target1                  = target1,
        target2                  = target2,
        risk_per_share           = round(risk_ps, 2),
        position_size            = pos_size,
        capital_required         = cap_req,
        rr_ratio                 = rr_ratio,
        atr                      = round(atr_val, 2),
        is_breaking_out          = pattern.is_breaking_out,
        is_extended              = pattern.is_extended,
        breakout_volume_ratio    = pattern.breakout_volume_ratio,
        rs_new_high              = pattern.rs_new_high,
        trend_template_ok        = pattern.trend_template_ok,
        trend_template_details   = pattern.trend_template_details,
        rs_rating                = round(rs, 1),
        sepa_score               = sepa,
        sepa_checks              = sepa_checks,
        prior_uptrend_pct        = pattern.prior_uptrend_pct,
        pct_below_52wk_high      = pattern.pct_below_52wk_high,
        weekly_trend             = w_trend,
        quality_score            = pattern.quality_score,
        market_regime            = pattern.market_regime,
        active_timeframes        = pattern.active_timeframes,
        tf_combo_bonus           = pattern.tf_combo_bonus,
        vol_compression_score    = pattern.vol_compression_score,
        score_breakdown          = {k: v for k, v in pattern.score_breakdown.explanations.items()} if pattern.score_breakdown else {},
        diagnostic_summary       = pattern.diagnostic.decision if pattern.diagnostic else "",
    )

    log.info(
        "VCP SIGNAL %-20s %dT quality=%5.1f rs=%4.1f "
        "final_width=%.1f%% pivot=%.2f status=%s",
        symbol, pattern.t_count, pattern.quality_score, rs,
        sig.final_contraction_width, pattern.pivot_price, sig.status,
    )
    return sig
