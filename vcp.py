"""
NSE Darvas Box Scanner - VCP (Volatility Contraction Pattern) Detector
=======================================================================
Implements Mark Minervini's Volatility Contraction Pattern as documented
in "Trade Like a Stock Market Wizard" (2013) and "Think & Trade Like a
Champion" (2017).

Detection Architecture: TWO-PASS DUAL METHOD
─────────────────────────────────────────────
PASS 1 — ATR Compression Pre-filter (fast, zero false negatives)
  Compute 10-day ATR vs 130-day baseline ATR normalised by price.
  If the ratio > VCP_ATR_COMPRESSION_MAX, the stock is NOT compressing
  enough to be in VCP territory — return None immediately.

PASS 2 — Rolling Time-Window Contraction Finder (precise)
  Divide the last VCP_LOOKBACK_BARS into N time windows (N = 2..6).
  For each window compute High-Low spread as % of midpoint.
  A VCP is detected when spreads are monotonically decreasing.
  Try both "equal" and "progressive" window splits, keep best score.

Why NOT swing-pivot detection:
  Swing-pivot detectors miss tight final contractions (no clear pivot
  bar), mid-formation bases (no right-side pivot yet), and smooth price
  action. Time-window boundaries capture all of these because the
  window's max/min is valid regardless of whether any bar qualifies as
  a "swing high" by a radius criterion.

Timeframe: DAILY bars for all contraction geometry.
  Weekly bars are only used for trend template MAs and prior uptrend.
  VCPs are 3-26 weeks (15-130 daily bars). On weekly bars a 3-week VCP
  collapses to 3 data points — not enough to detect 2-4 contractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List

import numpy as np
import pandas as pd

from config import (
    VCP_TREND_MA_SHORT, VCP_TREND_MA_MID, VCP_TREND_MA_LONG,
    VCP_52WK_HIGH_MAX_PCT_BELOW, VCP_52WK_LOW_MIN_PCT_ABOVE,
    VCP_MA200_UPTREND_LOOKBACK,
    VCP_PRIOR_UPTREND_LOOKBACK_DAYS, VCP_PRIOR_UPTREND_MIN_PCT,
    VCP_ATR_FAST_PERIOD, VCP_ATR_BASE_PERIOD, VCP_ATR_COMPRESSION_MAX,
    VCP_LOOKBACK_BARS, VCP_MIN_WINDOW_BARS, VCP_MAX_WINDOW_BARS,
    VCP_MIN_CONTRACTIONS, VCP_MAX_CONTRACTIONS,
    VCP_MIN_WIDTH_PCT, VCP_MAX_FIRST_WIDTH_PCT, VCP_MAX_LAST_WIDTH_PCT,
    VCP_TIGHTENING_RATIO, VCP_HIGHER_LOWS_TOLERANCE,
    VCP_FINAL_VOL_MAX_RATIO, VCP_VDU_TOLERANCE,
    VCP_BREAKOUT_VOLUME_RATIO, VCP_BREAKOUT_VOL_MA_PERIOD,
    VCP_BUY_ZONE_PCT, RS_WEIGHTS,
    MIN_AVG_VOLUME,
)
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VCPContraction:
    """One contraction (T) within the VCP sequence."""
    index:          int           # 1 = earliest/widest, t_count = final/tightest
    start_date:     date = None
    end_date:       date = None
    high:           float = 0.0   # highest intraday High in window
    low:            float = 0.0   # lowest intraday Low in window
    close_high:     float = 0.0   # highest CLOSE in window (used for pivot in final C)
    width_pct:      float = 0.0   # (high - low) / midpoint × 100
    duration_bars:  int   = 0     # trading days in this window
    avg_volume:     float = 0.0   # average daily volume in window
    volume_ratio:   float = 0.0   # avg_volume / C1 avg_volume


@dataclass
class VCPPattern:
    symbol: str

    # ── Trend template (9 criteria, ALL must pass for is_valid=True) ──────────
    trend_template_ok:      bool  = False
    trend_template_details: dict  = field(default_factory=dict)
    price_above_ma50:       bool  = False
    price_above_ma150:      bool  = False
    price_above_ma200:      bool  = False
    ma150_above_ma200:      bool  = False
    ma200_uptrend:          bool  = False
    ma50_above_ma150:       bool  = False
    ma50_above_ma200:       bool  = False
    pct_above_52wk_low:     float = 0.0
    pct_below_52wk_high:    float = 0.0

    # ── Prior uptrend ─────────────────────────────────────────────────────────
    prior_uptrend_pct:  float = 0.0
    prior_uptrend_ok:   bool  = False

    # ── ATR pre-filter ────────────────────────────────────────────────────────
    atr_compression_ratio: float = 1.0

    # ── Contraction sequence ──────────────────────────────────────────────────
    contractions:              List[VCPContraction] = field(default_factory=list)
    t_count:                   int   = 0
    window_strategy_used:      str   = ""
    contractions_tightening:   bool  = False
    contractions_shortening:   bool  = False
    volume_slope_negative:     bool  = False
    higher_lows_ok:            bool  = False
    vdu_day_present:           bool  = False

    # ── Pivot ─────────────────────────────────────────────────────────────────
    pivot_contraction:  Optional[VCPContraction] = None
    pivot_price:        float = 0.0   # highest CLOSE in final contraction
    pivot_low:          float = 0.0   # lowest intraday Low in final contraction
    buy_zone_high:      float = 0.0   # pivot × (1 + VCP_BUY_ZONE_PCT/100)
    buy_zone_low:       float = 0.0   # = pivot_price

    # ── Base metrics ──────────────────────────────────────────────────────────
    base_start_date:     date  = None
    base_duration_weeks: int   = 0

    # ── Breakout ──────────────────────────────────────────────────────────────
    is_breaking_out:        bool  = False
    is_extended:            bool  = False
    breakout_volume_ratio:  float = 0.0
    rs_new_high:            bool  = False
    rs_rating:              float = 0.0

    # ── Scoring & validity ────────────────────────────────────────────────────
    quality_score:     float = 0.0
    is_valid:          bool  = False
    rejection_reasons: list  = field(default_factory=list)
    soft_warnings:     list  = field(default_factory=list)


# ─── Pass 1: ATR Compression Pre-filter ───────────────────────────────────────

def _atr_compression_prefilter(daily: pd.DataFrame) -> tuple[bool, float]:
    """
    Fast volatility pre-filter. Returns (passes, compression_ratio).

    Uses percentage ATR (ATR / price) so stocks at different price levels
    are compared on the same scale. compression_ratio = recent_atr_pct /
    base_atr_pct. Passes when ratio ≤ VCP_ATR_COMPRESSION_MAX (≤ 0.60).

    Zero false negatives: every real VCP will show compressed ATR because
    volatility compression IS the defining property of the pattern.
    """
    min_bars = VCP_ATR_BASE_PERIOD + VCP_ATR_FAST_PERIOD + 5
    if len(daily) < min_bars:
        return False, 1.0

    high  = daily["High"]
    low   = daily["Low"]
    close = daily["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_fast = float(tr.rolling(VCP_ATR_FAST_PERIOD).mean().iloc[-1])
    atr_base = float(tr.rolling(VCP_ATR_BASE_PERIOD).mean().iloc[-1])

    price = float(close.iloc[-1])
    if atr_base <= 0 or price <= 0:
        return False, 1.0

    atr_pct_fast = atr_fast / price
    atr_pct_base = atr_base / price

    ratio = atr_pct_fast / atr_pct_base if atr_pct_base > 0 else 1.0
    return ratio <= VCP_ATR_COMPRESSION_MAX, round(ratio, 3)


# ─── Pass 2: Rolling Time-Window Contraction Finder ───────────────────────────

def _build_windows(
    n: int,
    n_contractions: int,
    strategy: str,
    min_bars: int,
) -> Optional[list[tuple[int, int]]]:
    """
    Build (start_idx, end_idx) pairs for each contraction window.

    "equal"       — all windows same size.
    "progressive" — windows get shorter left-to-right (first is ~2×
                    the last), matching real VCP structure where early
                    contractions take longer to form than the final one.

    Returns None if any window would be shorter than min_bars.
    """
    if strategy == "equal":
        window_size = n // n_contractions
        if window_size < min_bars:
            return None
        windows = []
        for i in range(n_contractions):
            start = i * window_size
            end   = (start + window_size - 1) if i < n_contractions - 1 else n - 1
            windows.append((start, end))
        return windows

    elif strategy == "progressive":
        ratios = [
            1.0 - 0.5 * i / max(n_contractions - 1, 1)
            for i in range(n_contractions)
        ]
        total_ratio = sum(ratios)
        base_size   = n / total_ratio
        sizes = [max(min_bars, round(base_size * r)) for r in ratios]
        sizes[-1] = n - sum(sizes[:-1])
        if sizes[-1] < min_bars:
            return None
        windows, cursor = [], 0
        for s in sizes:
            windows.append((cursor, cursor + s - 1))
            cursor += s
        return windows

    return None


def _score_window_candidate(contractions: list[dict]) -> float:
    """
    Score a candidate window split (0-100). Used to SELECT the best
    split when multiple N values and strategies are tried. Higher = more
    geometrically faithful VCP.

    This is NOT quality_score (which predicts trade outcomes). It just
    picks the best structural candidate from competing splits.
    """
    if len(contractions) < 2:
        return 0.0

    widths  = [c["width_pct"]     for c in contractions]
    volumes = [c["avg_volume"]    for c in contractions]
    durs    = [c["duration_bars"] for c in contractions]
    score   = 0.0

    # Width tightening (50 pts) — the defining VCP property
    tight_steps = sum(1 for i in range(1, len(widths)) if widths[i] < widths[i-1])
    score += (tight_steps / max(len(widths) - 1, 1)) * 50

    # Final contraction tightness (20 pts)
    fw = widths[-1]
    if fw <= 3.0:    score += 20
    elif fw <= 6.0:  score += 15
    elif fw <= 10.0: score += 9
    elif fw <= VCP_MAX_LAST_WIDTH_PCT: score += 4

    # Volume declining trend — linear regression slope (20 pts)
    if len(volumes) >= 2:
        x  = list(range(len(volumes)))
        xm = sum(x) / len(x)
        vm = sum(volumes) / len(volumes)
        cov = sum((xi - xm) * (vi - vm) for xi, vi in zip(x, volumes))
        var = sum((xi - xm) ** 2 for xi in x)
        slope = cov / var if var > 0 else 0
        if slope < 0:
            reduction = abs(slope * len(volumes)) / (vm if vm > 0 else 1)
            score += min(20.0, reduction * 40)

    # Duration shortening (10 pts)
    short_steps = sum(1 for i in range(1, len(durs)) if durs[i] <= durs[i-1])
    score += (short_steps / max(len(durs) - 1, 1)) * 10

    return round(score, 1)


def _find_contractions_by_windows(
    daily: pd.DataFrame,
) -> tuple[list[dict], str]:
    """
    PASS 2: Rolling time-window contraction finder.
    Returns (contractions_list, strategy_used).

    Tries N=2..VCP_MAX_CONTRACTIONS with both "equal" and "progressive"
    splits. Scores each candidate with _score_window_candidate().
    Returns the highest-scoring valid result in CHRONOLOGICAL ORDER
    (earliest/widest contraction first = index 1).
    """
    bars = (
        daily.iloc[-VCP_LOOKBACK_BARS:]
        if len(daily) >= VCP_LOOKBACK_BARS
        else daily
    )
    n = len(bars)

    best_contractions: list[dict] = []
    best_score:        float      = -1.0
    best_strategy:     str        = ""

    for n_c in range(VCP_MIN_CONTRACTIONS, VCP_MAX_CONTRACTIONS + 1):
        for strategy in ("equal", "progressive"):
            windows = _build_windows(n, n_c, strategy, VCP_MIN_WINDOW_BARS)
            if windows is None:
                continue

            candidate: list[dict] = []
            valid = True

            for idx, (s, e) in enumerate(windows):
                w = bars.iloc[s: e + 1]
                dur = len(w)
                if dur < VCP_MIN_WINDOW_BARS or dur > VCP_MAX_WINDOW_BARS:
                    valid = False
                    break

                high_val   = float(w["High"].max())
                low_val    = float(w["Low"].min())
                close_high = float(w["Close"].max())
                mid        = (high_val + low_val) / 2.0
                width_pct  = (high_val - low_val) / mid * 100 if mid > 0 else 0.0
                avg_vol    = float(w["Volume"].mean())

                if width_pct < VCP_MIN_WIDTH_PCT:
                    valid = False
                    break

                candidate.append({
                    "index":         idx + 1,
                    "start_date":    w.index[0].date(),
                    "end_date":      w.index[-1].date(),
                    "high":          round(high_val, 2),
                    "low":           round(low_val, 2),
                    "close_high":    round(close_high, 2),
                    "width_pct":     round(width_pct, 1),
                    "duration_bars": dur,
                    "avg_volume":    round(avg_vol, 0),
                    "volume_ratio":  0.0,
                })

            if not valid or len(candidate) < VCP_MIN_CONTRACTIONS:
                continue

            # Fill volume_ratio relative to C1
            c1_vol = candidate[0]["avg_volume"]
            for c in candidate:
                c["volume_ratio"] = round(c["avg_volume"] / c1_vol, 3) if c1_vol > 0 else 1.0

            sc = _score_window_candidate(candidate)
            if sc > best_score:
                best_score        = sc
                best_contractions = [dict(c) for c in candidate]
                best_strategy     = strategy

    return best_contractions, best_strategy


# ─── Trend Template ────────────────────────────────────────────────────────────

def _check_trend_template(result: VCPPattern, daily: pd.DataFrame) -> None:
    """
    Minervini's Trend Template: ALL 9 criteria must pass simultaneously.
    Uses SMA (not EMA) — Minervini specifies simple moving averages.
    """
    close = daily["Close"]
    needed = VCP_TREND_MA_LONG + VCP_MA200_UPTREND_LOOKBACK + 10
    if len(close) < needed:
        result.trend_template_ok = False
        result.rejection_reasons.append("Insufficient history for trend template MAs")
        return

    price     = float(close.iloc[-1])
    ma50      = float(close.rolling(VCP_TREND_MA_SHORT).mean().iloc[-1])
    ma150     = float(close.rolling(VCP_TREND_MA_MID).mean().iloc[-1])
    ma200     = float(close.rolling(VCP_TREND_MA_LONG).mean().iloc[-1])
    ma200_ago = float(
        close.rolling(VCP_TREND_MA_LONG).mean().iloc[-(VCP_MA200_UPTREND_LOOKBACK + 1)]
    )

    high_52wk = float(daily["High"].rolling(252).max().iloc[-1])
    low_52wk  = float(daily["Low"].rolling(252).min().iloc[-1])

    criteria = {
        "price_above_ma50":            price > ma50,
        "price_above_ma150":           price > ma150,
        "price_above_ma200":           price > ma200,
        "ma150_above_ma200":           ma150 > ma200,
        "ma50_above_ma150":            ma50 > ma150,
        "ma50_above_ma200":            ma50 > ma200,
        "ma200_uptrend":               ma200 > ma200_ago,
        "within_25pct_of_52wk_high":   (
            high_52wk > 0 and
            price >= high_52wk * (1 - VCP_52WK_HIGH_MAX_PCT_BELOW / 100)
        ),
        "above_25pct_of_52wk_low":     (
            low_52wk > 0 and
            (price / low_52wk - 1) * 100 >= VCP_52WK_LOW_MIN_PCT_ABOVE
        ),
    }

    result.trend_template_details = criteria
    result.trend_template_ok      = all(criteria.values())

    result.price_above_ma50   = criteria["price_above_ma50"]
    result.price_above_ma150  = criteria["price_above_ma150"]
    result.price_above_ma200  = criteria["price_above_ma200"]
    result.ma150_above_ma200  = criteria["ma150_above_ma200"]
    result.ma200_uptrend      = criteria["ma200_uptrend"]
    result.ma50_above_ma150   = criteria["ma50_above_ma150"]
    result.ma50_above_ma200   = criteria["ma50_above_ma200"]
    result.pct_above_52wk_low  = round((price / low_52wk - 1) * 100, 1) if low_52wk > 0 else 0.0
    result.pct_below_52wk_high = round((1 - price / high_52wk) * 100, 1) if high_52wk > 0 else 100.0

    if not result.trend_template_ok:
        failed = [k for k, v in criteria.items() if not v]
        result.rejection_reasons.append(
            f"Trend template failed — {len(failed)}/9 criteria not met: {failed}"
        )


# ─── Prior Uptrend ────────────────────────────────────────────────────────────

def _check_prior_uptrend(result: VCPPattern, daily: pd.DataFrame) -> None:
    """
    Stock must have advanced ≥ VCP_PRIOR_UPTREND_MIN_PCT before the base.
    Looks back VCP_PRIOR_UPTREND_LOOKBACK_DAYS and finds the prior low.
    """
    lookback = min(VCP_PRIOR_UPTREND_LOOKBACK_DAYS, len(daily) - 1)
    prior    = daily["Close"].iloc[-(lookback + 1):-1]
    if prior.empty:
        result.prior_uptrend_pct = 0.0
        result.prior_uptrend_ok  = False
        return

    prior_low    = float(prior.min())
    current_high = float(daily["Close"].iloc[-1])
    pct = (current_high / prior_low - 1) * 100 if prior_low > 0 else 0.0

    result.prior_uptrend_pct = round(pct, 1)
    result.prior_uptrend_ok  = pct >= VCP_PRIOR_UPTREND_MIN_PCT

    if not result.prior_uptrend_ok:
        result.rejection_reasons.append(
            f"Prior uptrend {pct:.1f}% < minimum {VCP_PRIOR_UPTREND_MIN_PCT:.0f}% required"
        )


# ─── Contraction Sequence Validation ─────────────────────────────────────────

def _validate_contraction_sequence(result: VCPPattern) -> None:
    """
    Validate that the winning window split satisfies VCP structural rules.
    Volume uses LINEAR REGRESSION slope — more robust than step-by-step
    comparison or ratio to C1 (which goes stale for 4T+ long bases).
    """
    cs = result.contractions
    if len(cs) < 2:
        result.contractions_tightening = False
        return

    widths  = [c.width_pct     for c in cs]
    volumes = [c.avg_volume    for c in cs]
    durs    = [c.duration_bars for c in cs]

    # Width monotonically tightening
    width_ok = all(
        cs[i].width_pct <= cs[i-1].width_pct * VCP_TIGHTENING_RATIO
        for i in range(1, len(cs))
    )
    result.contractions_tightening = width_ok
    if not width_ok:
        result.rejection_reasons.append(
            f"Contractions NOT tightening by ≥{int((1-VCP_TIGHTENING_RATIO)*100)}% each step. "
            f"Widths: {widths}"
        )

    # Duration shortening (soft)
    dur_ok = all(cs[i].duration_bars <= cs[i-1].duration_bars for i in range(1, len(cs)))
    result.contractions_shortening = dur_ok
    if not dur_ok:
        result.soft_warnings.append(
            f"SOFT: Durations not consistently shortening {durs}. "
            "Still valid if widths tighten."
        )

    # Volume declining trend via linear regression slope
    if len(volumes) >= 2:
        x  = list(range(len(volumes)))
        xm = sum(x) / len(x)
        vm = sum(volumes) / len(volumes)
        cov = sum((xi - xm) * (vi - vm) for xi, vi in zip(x, volumes))
        var = sum((xi - xm) ** 2 for xi in x)
        slope = cov / var if var > 0 else 0
        result.volume_slope_negative = slope < 0
    else:
        result.volume_slope_negative = False

    if not result.volume_slope_negative:
        result.soft_warnings.append(
            "SOFT: Volume not trending downward across contractions. "
            "Rising volume in the base = supply not fully absorbed."
        )

    # Final volume ratio check
    if cs and cs[-1].volume_ratio > VCP_FINAL_VOL_MAX_RATIO:
        result.soft_warnings.append(
            f"SOFT: Final contraction avg volume is {cs[-1].volume_ratio:.0%} of C1 "
            f"(target ≤ {VCP_FINAL_VOL_MAX_RATIO:.0%}). Volume not sufficiently dried up."
        )

    # First contraction width gate
    if cs[0].width_pct > VCP_MAX_FIRST_WIDTH_PCT:
        result.rejection_reasons.append(
            f"First contraction {cs[0].width_pct:.1f}% exceeds {VCP_MAX_FIRST_WIDTH_PCT:.0f}% max. "
            "Too wide — likely a crash recovery, not an orderly VCP base."
        )

    # Final contraction width gate
    if cs[-1].width_pct > VCP_MAX_LAST_WIDTH_PCT:
        result.rejection_reasons.append(
            f"Final contraction {cs[-1].width_pct:.1f}% exceeds {VCP_MAX_LAST_WIDTH_PCT:.0f}% max. "
            "Too wide for a valid pivot zone."
        )


# ─── Higher Lows Check ────────────────────────────────────────────────────────

def _check_higher_lows(contractions: list) -> bool:
    """
    Each contraction LOW must be at or above the previous LOW.
    Higher lows = buyers stepping in at higher prices = accumulation.
    Lower lows = distribution still ongoing = VCP is invalid.
    Allow VCP_HIGHER_LOWS_TOLERANCE (2%) for NSE data noise.
    """
    for i in range(1, len(contractions)):
        prev_low = contractions[i-1].low
        curr_low = contractions[i].low
        if curr_low < prev_low * (1 - VCP_HIGHER_LOWS_TOLERANCE):
            return False
    return True


# ─── Precise Pivot Identification ────────────────────────────────────────────

def _identify_pivot(
    daily: pd.DataFrame, final_c: VCPContraction
) -> tuple[float, float]:
    """
    Pivot = highest CLOSING price in the final contraction window.
    Closing prices define resistance — intraday spikes that closed back
    down are noise, not resistance.

    Stop reference = lowest INTRADAY LOW in the final contraction window.
    Minervini places the stop below the lowest intraday point.
    """
    start = pd.Timestamp(final_c.start_date)
    end   = pd.Timestamp(final_c.end_date)
    w     = daily[(daily.index >= start) & (daily.index <= end)]

    if w.empty:
        return final_c.high, final_c.low

    pivot    = float(w["Close"].max())
    stop_ref = float(w["Low"].min())
    return round(pivot, 2), round(stop_ref, 2)


# ─── Volume Dry-Up Day ────────────────────────────────────────────────────────

def _check_vdu_day(
    daily: pd.DataFrame, final_c: VCPContraction
) -> bool:
    """
    VDU = at least one day in the final contraction at/near 52-week-low
    volume. Minervini: "sellers exhausted — no one wants to sell here."
    Uses VCP_VDU_TOLERANCE (20%) for NSE data gaps/reporting lags.
    """
    vol_252   = daily["Volume"].iloc[-252:]
    low_vol   = float(vol_252.min())
    threshold = low_vol * VCP_VDU_TOLERANCE

    start = pd.Timestamp(final_c.start_date)
    end   = pd.Timestamp(final_c.end_date)
    w     = daily[(daily.index >= start) & (daily.index <= end)]

    return bool((w["Volume"] <= threshold).any()) if not w.empty else False


# ─── Breakout Confirmation ────────────────────────────────────────────────────

def _check_breakout(
    result: VCPPattern,
    daily: pd.DataFrame,
    benchmark: Optional[pd.DataFrame],
) -> None:
    """
    Breakout = close above pivot on volume ≥ VCP_BREAKOUT_VOLUME_RATIO
    of the 50-day average (Minervini's explicit baseline period).
    """
    if result.pivot_price <= 0:
        return

    last       = daily.iloc[-1]
    last_close = float(last["Close"])
    last_vol   = float(last["Volume"])

    avg_vol_50 = (
        float(daily["Volume"].iloc[-51:-1].mean())
        if len(daily) > 51
        else float(daily["Volume"].mean())
    )

    result.breakout_volume_ratio = (
        round(last_vol / avg_vol_50, 2) if avg_vol_50 > 0 else 0.0
    )

    above_pivot  = last_close >= result.pivot_price
    in_buy_zone  = last_close <= result.buy_zone_high
    volume_surge = result.breakout_volume_ratio >= VCP_BREAKOUT_VOLUME_RATIO

    result.is_breaking_out = above_pivot and in_buy_zone and volume_surge
    result.is_extended     = last_close > result.buy_zone_high

    if above_pivot and volume_surge and not in_buy_zone:
        result.soft_warnings.append(
            f"SOFT: {((last_close/result.pivot_price)-1)*100:.1f}% above pivot — "
            "outside buy zone. Do NOT chase."
        )
    if above_pivot and in_buy_zone and not volume_surge:
        result.soft_warnings.append(
            f"SOFT: Price in buy zone but volume only {result.breakout_volume_ratio:.2f}× "
            f"50-day average (need ≥ {VCP_BREAKOUT_VOLUME_RATIO:.2f}×). "
            "Unconfirmed breakout."
        )


# ─── RS Line New High ─────────────────────────────────────────────────────────

def _check_rs_new_high(
    result: VCPPattern,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    """
    RS line new high = strongest breakout confirmation signal Minervini mentions.
    RS line = stock / benchmark normalised. Leading RS (new high before or at
    price breakout) indicates institutional accumulation in progress.
    """
    try:
        bench   = benchmark["Close"].reindex(daily.index, method="ffill").dropna()
        stock   = daily["Close"].reindex(bench.index)
        rs_line = stock / bench * 100

        rs_now           = float(rs_line.iloc[-1])
        prior_slice      = rs_line.iloc[-273:-21] if len(rs_line) >= 273 else rs_line.iloc[:-21]
        rs_52wk_high_ago = float(prior_slice.max()) if not prior_slice.empty else rs_now

        result.rs_new_high = rs_now >= rs_52wk_high_ago
    except Exception:
        result.rs_new_high = False


# ─── VCP Quality Score ────────────────────────────────────────────────────────

def _score_vcp(result: VCPPattern) -> float:
    """
    0-100 quality score. INITIAL weights — must be rebalanced after the
    first backtest run by computing Pearson correlation of each factor
    with win/loss outcome (same process used to rebalance Darvas Box
    weights: RSI raised 5→20 pts, ADX dropped 10→5 pts).

    Factor weights:
      Trend template (9 criteria pass/partial) : 22 pts
      T-count quality (3T/4T ideal)            : 10 pts
      Width tightening (monotonic)             : 18 pts
      Final contraction tightness              : 12 pts
      Higher lows across sequence              : 10 pts
      Volume slope negative                    :  8 pts
      VDU day present                          :  6 pts
      RS rating                                :  8 pts
      Breakout volume surge                    :  6 pts
    Total: 100 pts
    """
    score = 0.0

    # Trend template (22 pts — partial credit for partially-passing criteria)
    if result.trend_template_details:
        passed = sum(1 for v in result.trend_template_details.values() if v)
        total  = len(result.trend_template_details)
        score += (passed / total) * 22
    elif result.trend_template_ok:
        score += 22

    # T-count (10 pts)
    if   result.t_count >= 4: score += 10
    elif result.t_count == 3: score += 8
    elif result.t_count == 2: score += 5

    # Width tightening (18 pts)
    if result.contractions_tightening:
        score += 18
    elif len(result.contractions) >= 2:
        cs = result.contractions
        tight = sum(
            1 for i in range(1, len(cs))
            if cs[i].width_pct < cs[i-1].width_pct
        )
        score += (tight / max(len(cs) - 1, 1)) * 9

    # Final contraction tightness (12 pts)
    if result.contractions:
        fw = result.contractions[-1].width_pct
        if fw <= 3.0:    score += 12
        elif fw <= 6.0:  score += 9
        elif fw <= 9.0:  score += 5
        elif fw <= VCP_MAX_LAST_WIDTH_PCT: score += 2

    # Higher lows (10 pts)
    if result.higher_lows_ok:
        score += 10

    # Volume slope negative (8 pts)
    if result.volume_slope_negative:
        score += 8
    elif result.contractions and result.contractions[-1].volume_ratio < 1.0:
        score += 3

    # VDU day (6 pts)
    if result.vdu_day_present:
        score += 6

    # RS rating (8 pts)
    if   result.rs_rating >= 90: score += 8
    elif result.rs_rating >= 85: score += 6
    elif result.rs_rating >= 70: score += 3
    elif result.rs_rating >= 50: score += 1

    # Breakout volume surge (6 pts)
    if result.is_breaking_out:
        if   result.breakout_volume_ratio >= 2.0:               score += 6
        elif result.breakout_volume_ratio >= 1.5:               score += 5
        elif result.breakout_volume_ratio >= VCP_BREAKOUT_VOLUME_RATIO: score += 4
    elif result.breakout_volume_ratio >= 1.1:
        score += 1

    return round(min(score, 100.0), 1)


# ─── Main Detection Function ──────────────────────────────────────────────────

def detect_vcp(
    symbol:    str,
    daily:     pd.DataFrame,
    weekly:    Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.DataFrame] = None,
) -> Optional[VCPPattern]:
    """
    Two-pass VCP detector. Returns VCPPattern (is_valid may be False for
    near-miss diagnostics) or None if not a VCP at all.

    daily     — DAILY OHLCV. ALL contraction geometry runs on daily bars.
    weekly    — optional; not used for geometry (only for future context).
    benchmark — Nifty 500 daily OHLCV for RS calculations.
    """
    if daily is None or len(daily) < 200:
        return None

    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    result = VCPPattern(symbol=symbol)

    # ── PASS 1: ATR compression pre-filter ─────────────────────────────────
    passes_atr, atr_ratio = _atr_compression_prefilter(daily)
    result.atr_compression_ratio = atr_ratio
    if not passes_atr:
        return None   # Not in VCP territory — skip expensive analysis

    # ── Trend Template ──────────────────────────────────────────────────────
    _check_trend_template(result, daily)

    # ── Prior uptrend ───────────────────────────────────────────────────────
    _check_prior_uptrend(result, daily)

    # ── PASS 2: Rolling window contraction finder ───────────────────────────
    raw_contractions, strategy = _find_contractions_by_windows(daily)

    if len(raw_contractions) < VCP_MIN_CONTRACTIONS:
        result.rejection_reasons.append(
            f"Found {len(raw_contractions)} window(s); minimum is {VCP_MIN_CONTRACTIONS}"
        )
        return result

    # Convert raw dicts → VCPContraction dataclasses
    result.contractions = [
        VCPContraction(
            index         = c["index"],
            start_date    = c["start_date"],
            end_date      = c["end_date"],
            high          = c["high"],
            low           = c["low"],
            close_high    = c["close_high"],
            width_pct     = c["width_pct"],
            duration_bars = c["duration_bars"],
            avg_volume    = c["avg_volume"],
            volume_ratio  = c["volume_ratio"],
        )
        for c in raw_contractions
    ]
    result.t_count             = len(result.contractions)
    result.window_strategy_used = strategy

    # ── Validate contraction sequence ───────────────────────────────────────
    _validate_contraction_sequence(result)

    # ── Higher lows ─────────────────────────────────────────────────────────
    result.higher_lows_ok = _check_higher_lows(result.contractions)
    if not result.higher_lows_ok:
        result.rejection_reasons.append(
            "Lower lows detected across contractions — distribution still ongoing, "
            "not accumulation. VCPs must show higher lows."
        )

    # ── Precise pivot from daily closing prices ─────────────────────────────
    final_c = result.contractions[-1]
    result.pivot_contraction = final_c
    pivot, stop_ref = _identify_pivot(daily, final_c)
    result.pivot_price   = pivot
    result.pivot_low     = stop_ref
    result.buy_zone_low  = pivot
    result.buy_zone_high = round(pivot * (1 + VCP_BUY_ZONE_PCT / 100), 2)

    # ── VDU day check ───────────────────────────────────────────────────────
    result.vdu_day_present = _check_vdu_day(daily, final_c)
    if not result.vdu_day_present:
        result.soft_warnings.append(
            "SOFT: No VDU day (52-week-low volume) in final contraction. "
            "Supply may not be fully exhausted — weakens the setup."
        )

    # ── Base start and duration ─────────────────────────────────────────────
    result.base_start_date     = result.contractions[0].start_date
    days_in_base               = (daily.index[-1].date() - result.base_start_date).days
    result.base_duration_weeks = days_in_base // 7

    # ── Breakout confirmation ───────────────────────────────────────────────
    _check_breakout(result, daily, benchmark)

    # ── RS line new high ────────────────────────────────────────────────────
    if benchmark is not None:
        _check_rs_new_high(result, daily, benchmark)

    # ── Quality score ───────────────────────────────────────────────────────
    result.quality_score = _score_vcp(result)

    # ── Final validity ──────────────────────────────────────────────────────
    hard_fails = [r for r in result.rejection_reasons if not r.startswith("SOFT:")]
    result.is_valid = (
        result.trend_template_ok
        and result.prior_uptrend_ok
        and result.t_count >= VCP_MIN_CONTRACTIONS
        and result.contractions_tightening
        and result.higher_lows_ok
        and len(hard_fails) == 0
    )

    log.debug(
        "%s VCP: %dT strategy=%s valid=%s score=%.1f atr=%.2f",
        symbol, result.t_count, strategy,
        result.is_valid, result.quality_score, atr_ratio,
    )
    return result
