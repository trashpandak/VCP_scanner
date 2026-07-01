"""
VCP Scanner v2 — Institutional-Grade Volatility Contraction Pattern Detector
=============================================================================
Redesigned based on backtest analysis of 196 trades:
  • RS Rating has the highest win-correlation (0.218) — weighted highest
  • SEPA trend template boosts WR by 9.6pp — kept as scoring factor
  • Score band 70-90 achieves 45% WR vs 30% below 60 — score gates validated
  • 5T dominates the dataset — multi-scale windows working correctly
  • Hard ATR gate was too strict — replaced with penalty scoring

Key improvements over v1:
  1. Multi-timeframe detection (Monthly / Weekly / Daily independently)
  2. Multi-scale lookback (9 window sizes per timeframe — never misses long bases)
  3. No hard rejections except extreme cases — everything is a penalty score
  4. Full diagnostic report for every pattern (passed and rejected)
  5. Adaptive contraction tolerance (not fixed percentages)
  6. Advanced volume analysis (trend, dry-up percentile, accumulation)
  7. Composite volatility compression (ATR + StdDev + BB width)
  8. Market regime filter (bull/bear/sideways)
  9. Explainable scoring — every point explained
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    # Multi-TF
    VCP_TF_WEIGHTS, VCP_TF_COMBO_BONUS,
    VCP_LOOKBACK_WINDOWS, VCP_WEEKLY_LOOKBACK_WINDOWS, VCP_MONTHLY_LOOKBACK_WINDOWS,
    # Contraction
    VCP_MIN_CONTRACTIONS, VCP_MAX_CONTRACTIONS,
    VCP_MIN_WINDOW_BARS, VCP_MAX_WINDOW_BARS,
    VCP_MIN_WIDTH_PCT, VCP_MAX_FIRST_WIDTH_PCT, VCP_MAX_LAST_WIDTH_PCT,
    VCP_TIGHTENING_RATIO, VCP_TIGHTENING_TOLERANCE, VCP_HIGHER_LOWS_TOLERANCE,
    # ATR
    VCP_ATR_FAST_PERIOD, VCP_ATR_BASE_PERIOD,
    VCP_ATR_COMPRESSION_MAX, VCP_ATR_GOOD_THRESHOLD, VCP_ATR_GREAT_THRESHOLD,
    # Volume
    VCP_FINAL_VOL_MAX_RATIO, VCP_VDU_TOLERANCE,
    VCP_BREAKOUT_VOLUME_RATIO, VCP_BREAKOUT_VOL_MA_PERIOD,
    # SEPA / trend
    VCP_TREND_MA_SHORT, VCP_TREND_MA_MID, VCP_TREND_MA_LONG,
    VCP_52WK_HIGH_MAX_PCT_BELOW, VCP_52WK_LOW_MIN_PCT_ABOVE,
    VCP_MA200_UPTREND_LOOKBACK,
    # Prior uptrend
    VCP_PRIOR_UPTREND_LOOKBACK_DAYS, VCP_PRIOR_UPTREND_MIN_PCT,
    # Pivot
    VCP_BUY_ZONE_PCT,
    # RS
    VCP_RS_MIN, VCP_RS_PREFERRED,
    # Market regime
    VCP_REGIME_MA_PERIOD, VCP_REGIME_SLOPE_DAYS,
    VCP_REGIME_BULL_BONUS, VCP_REGIME_BEAR_PENALTY,
    # Scoring
    VCP_SCORE_WEIGHTS, VCP_MIN_QUALITY_SCORE,
    # Misc
    MIN_AVG_VOLUME, RS_WEIGHTS,
)
from downloader import resample_weekly, resample_monthly
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VCPContraction:
    index:          int
    start_date:     date  = None
    end_date:       date  = None
    high:           float = 0.0
    low:            float = 0.0
    close_high:     float = 0.0
    width_pct:      float = 0.0
    duration_bars:  int   = 0
    avg_volume:     float = 0.0
    volume_ratio:   float = 0.0


@dataclass
class TimeframeResult:
    """VCP detection result for a single timeframe."""
    timeframe:              str    = ""    # "daily" / "weekly" / "monthly"
    found:                  bool   = False
    t_count:                int    = 0
    contractions:           list   = field(default_factory=list)
    contraction_widths:     list   = field(default_factory=list)
    contraction_lows:       list   = field(default_factory=list)
    contractions_tightening: bool  = False
    higher_lows_ok:         bool   = False
    volume_slope_negative:  bool   = False
    window_strategy:        str    = ""
    lookback_bars:          int    = 0
    candidate_score:        float  = 0.0
    pivot_price:            float  = 0.0
    pivot_low:              float  = 0.0


@dataclass
class ScoreBreakdown:
    """Detailed scoring with per-factor explanation."""
    prior_uptrend:          float = 0.0
    contraction_quality:    float = 0.0
    sepa_template:          float = 0.0
    rs_rating:              float = 0.0
    volume_dryup:           float = 0.0
    volatility_compression: float = 0.0
    breakout_quality:       float = 0.0
    market_regime:          float = 0.0
    timeframe_bonus:        float = 0.0
    total:                  float = 0.0
    explanations:           dict  = field(default_factory=dict)


@dataclass
class VCPDiagnostic:
    """Full transparency report — generated for EVERY stock, pass or fail."""
    symbol:             str
    scan_date:          date   = None
    overall_score:      float  = 0.0
    decision:           str    = "Rejected"   # Detected / Watching / Rejected
    confidence_pct:     float  = 0.0

    # Factor results
    checks_passed:      list   = field(default_factory=list)
    checks_failed:      list   = field(default_factory=list)
    penalties_applied:  list   = field(default_factory=list)
    rejection_reasons:  list   = field(default_factory=list)
    soft_warnings:      list   = field(default_factory=list)

    # Contraction detail
    best_timeframe:     str    = ""
    t_count:            int    = 0
    contraction_widths: list   = field(default_factory=list)
    contraction_lows:   list   = field(default_factory=list)
    tf_results:         dict   = field(default_factory=dict)  # per-TF results

    # Score breakdown
    score_breakdown:    ScoreBreakdown = field(default_factory=ScoreBreakdown)

    # Suggestion for false-negative analysis
    nearest_threshold:  str    = ""
    threshold_gap_pct:  float  = 0.0
    improvement_hint:   str    = ""


@dataclass
class VCPPattern:
    """The main output object — includes both detection and full diagnostic."""
    symbol: str

    # ── Validity ─────────────────────────────────────────────────────────────
    is_valid:           bool  = False
    is_breaking_out:    bool  = False
    is_extended:        bool  = False

    # ── Multi-timeframe ───────────────────────────────────────────────────────
    tf_daily:           Optional[TimeframeResult]   = None
    tf_weekly:          Optional[TimeframeResult]   = None
    tf_monthly:         Optional[TimeframeResult]   = None
    active_timeframes:  list  = field(default_factory=list)
    tf_combo_bonus:     float = 1.0
    best_timeframe:     str   = "daily"

    # ── Primary contraction sequence (best timeframe) ─────────────────────────
    contractions:       list  = field(default_factory=list)
    t_count:            int   = 0
    window_strategy:    str   = ""
    lookback_bars:      int   = 0
    contractions_tightening:  bool  = False
    contractions_shortening:  bool  = False
    volume_slope_negative:    bool  = False
    higher_lows_ok:           bool  = False
    vdu_day_present:          bool  = False

    # ── Pivot ────────────────────────────────────────────────────────────────
    pivot_price:        float = 0.0
    pivot_low:          float = 0.0
    buy_zone_high:      float = 0.0
    base_start_date:    date  = None
    base_duration_weeks: int  = 0

    # ── Breakout ─────────────────────────────────────────────────────────────
    breakout_volume_ratio:  float = 0.0
    rs_new_high:            bool  = False
    rs_rating:              float = 0.0

    # ── SEPA / Trend Template ─────────────────────────────────────────────────
    trend_template_ok:      bool  = False
    trend_template_details: dict  = field(default_factory=dict)
    sepa_score:             float = 0.0
    price_above_ma50:       bool  = False
    price_above_ma150:      bool  = False
    price_above_ma200:      bool  = False
    ma150_above_ma200:      bool  = False
    ma200_uptrend:          bool  = False
    ma50_above_ma150:       bool  = False
    ma50_above_ma200:       bool  = False
    pct_above_52wk_low:     float = 0.0
    pct_below_52wk_high:    float = 0.0
    prior_uptrend_pct:      float = 0.0
    prior_uptrend_ok:       bool  = False

    # ── Volatility ────────────────────────────────────────────────────────────
    atr_compression_ratio:  float = 1.0
    vol_compression_score:  float = 0.0   # composite (ATR + StdDev + BB)

    # ── Volume analysis ───────────────────────────────────────────────────────
    volume_dryup_score:     float = 0.0
    volume_trend_score:     float = 0.0
    accumulation_score:     float = 0.0

    # ── Market regime ─────────────────────────────────────────────────────────
    market_regime:          str   = "unknown"
    regime_multiplier:      float = 1.0

    # ── Scoring ───────────────────────────────────────────────────────────────
    quality_score:          float = 0.0
    score_breakdown:        ScoreBreakdown = field(default_factory=ScoreBreakdown)
    diagnostic:             VCPDiagnostic  = None


# ─── ATR Compression Pre-filter ───────────────────────────────────────────────

def _atr_compression(bars: pd.DataFrame) -> tuple[bool, float, float]:
    """
    Returns (passes_hard_gate, compression_ratio, compression_score_0_to_1).
    Hard gate fails only above VCP_ATR_COMPRESSION_MAX (0.75).
    Score is graduated: 0.40 = 1.0, 0.55 = 0.7, 0.75 = 0.2
    """
    min_bars = VCP_ATR_BASE_PERIOD + VCP_ATR_FAST_PERIOD + 5
    if len(bars) < min_bars:
        return False, 1.0, 0.0

    hi, lo, cl = bars["High"], bars["Low"], bars["Close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_fast = float(tr.rolling(VCP_ATR_FAST_PERIOD).mean().iloc[-1])
    atr_base = float(tr.rolling(VCP_ATR_BASE_PERIOD).mean().iloc[-1])
    price    = float(cl.iloc[-1])
    if atr_base <= 0 or price <= 0:
        return False, 1.0, 0.0

    ratio = (atr_fast / price) / (atr_base / price)
    ratio = round(ratio, 3)

    passes = ratio <= VCP_ATR_COMPRESSION_MAX

    # Graduated score
    if ratio <= VCP_ATR_GREAT_THRESHOLD:
        score = 1.0
    elif ratio <= VCP_ATR_GOOD_THRESHOLD:
        score = 0.75
    elif ratio <= 0.65:
        score = 0.50
    elif ratio <= VCP_ATR_COMPRESSION_MAX:
        score = 0.25
    else:
        score = max(0.0, 0.25 - (ratio - VCP_ATR_COMPRESSION_MAX) * 2)

    return passes, ratio, score


# ─── Composite Volatility Compression ────────────────────────────────────────

def _volatility_compression_score(bars: pd.DataFrame) -> float:
    """
    Combines ATR ratio, rolling StdDev compression, and Bollinger Band width
    into a single 0-1 score. Higher = more compressed = better VCP setup.
    """
    if len(bars) < 60:
        return 0.0

    close = bars["Close"]
    hi    = bars["High"]
    lo    = bars["Low"]

    # 1. ATR compression (already computed but recompute cleanly here)
    tr = pd.concat([
        hi - lo,
        (hi - close.shift(1)).abs(),
        (lo - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_fast = tr.rolling(10).mean().iloc[-1]
    atr_slow = tr.rolling(50).mean().iloc[-1]
    atr_ratio = (atr_fast / atr_slow) if atr_slow > 0 else 1.0

    # 2. Standard deviation compression
    std_fast = close.rolling(10).std().iloc[-1]
    std_slow = close.rolling(50).std().iloc[-1]
    price    = close.iloc[-1]
    std_ratio = (std_fast / price) / (std_slow / price) if std_slow > 0 and price > 0 else 1.0

    # 3. Bollinger Band width compression
    mid20     = close.rolling(20).mean()
    bb_std    = close.rolling(20).std()
    bb_width  = ((2 * bb_std) / mid20).iloc[-1]
    bb_width_6mo = ((2 * close.rolling(130).std()) / close.rolling(130).mean()).iloc[-1]
    bb_ratio  = bb_width / bb_width_6mo if bb_width_6mo > 0 else 1.0

    # Combine: lower ratio = more compression = higher score
    combined = (atr_ratio * 0.4 + std_ratio * 0.3 + bb_ratio * 0.3)
    score = max(0.0, min(1.0, 1.5 - combined))   # maps 0.5=1.0, 1.5=0.0
    return round(float(score), 3)


# ─── Volume Analysis ─────────────────────────────────────────────────────────

def _volume_analysis(bars: pd.DataFrame, contractions: list[dict]) -> dict:
    """
    Returns a dict of volume quality metrics:
      dryup_score     — how much volume has dried up (0-1)
      trend_score     — is volume declining? (0-1)
      accumulation    — up-volume vs down-volume ratio (0-1)
      vdu_day         — any 52wk low volume day in final contraction?
    """
    vol = bars["Volume"]
    close = bars["Close"]

    if len(vol) < 50:
        return {"dryup_score": 0.0, "trend_score": 0.0, "accumulation": 0.5, "vdu_day": False}

    # 1. Volume dry-up: recent 10d vs 50d average
    vol_10d  = float(vol.iloc[-10:].mean())
    vol_50d  = float(vol.iloc[-50:].mean())
    vol_252d = float(vol.iloc[-252:].mean()) if len(vol) >= 252 else vol_50d
    dryup_ratio = vol_10d / vol_50d if vol_50d > 0 else 1.0
    dryup_score = max(0.0, min(1.0, 1.5 - dryup_ratio))

    # 2. Volume trend: linear regression slope across contractions
    if len(contractions) >= 2:
        vols  = [c["avg_volume"] for c in contractions]
        x     = list(range(len(vols)))
        xm, vm = sum(x) / len(x), sum(vols) / len(vols)
        cov   = sum((xi - xm) * (vi - vm) for xi, vi in zip(x, vols))
        var   = sum((xi - xm) ** 2 for xi in x)
        slope = cov / var if var > 0 else 0
        # Negative slope = declining volume = good
        if slope < 0:
            reduction = abs(slope * len(vols)) / (vm if vm > 0 else 1)
            trend_score = min(1.0, reduction * 2)
        else:
            trend_score = 0.0
    else:
        trend_score = 0.5

    # 3. Accumulation score: up-volume vs down-volume
    returns   = close.pct_change().fillna(0)
    up_vol    = vol[returns > 0].sum()
    down_vol  = vol[returns < 0].sum()
    total_vol = up_vol + down_vol
    accum     = float(up_vol / total_vol) if total_vol > 0 else 0.5

    # 4. VDU day in final contraction
    vdu_day = False
    if contractions:
        final_c = contractions[-1]
        vol_52wk_low = float(vol.iloc[-252:].min()) if len(vol) >= 252 else float(vol.min())
        threshold    = vol_52wk_low * VCP_VDU_TOLERANCE
        start = pd.Timestamp(final_c["start_date"])
        end   = pd.Timestamp(final_c["end_date"])
        w     = bars[(bars.index >= start) & (bars.index <= end)]
        if not w.empty:
            vdu_day = bool((w["Volume"] <= threshold).any())

    return {
        "dryup_score":  round(dryup_score, 3),
        "trend_score":  round(trend_score, 3),
        "accumulation": round(accum, 3),
        "vdu_day":      vdu_day,
    }


# ─── Market Regime ────────────────────────────────────────────────────────────

def _market_regime(benchmark: Optional[pd.DataFrame]) -> tuple[str, float]:
    """
    Returns (regime_label, score_multiplier).
    regime: "bull" / "correction" / "bear" / "sideways" / "unknown"
    multiplier: 1.10 (bull) → 0.85 (bear)
    """
    if benchmark is None or len(benchmark) < VCP_REGIME_MA_PERIOD + VCP_REGIME_SLOPE_DAYS:
        return "unknown", 1.0

    close  = benchmark["Close"]
    ma200  = close.rolling(VCP_REGIME_MA_PERIOD).mean()
    price  = float(close.iloc[-1])
    ma_now = float(ma200.iloc[-1])

    # Slope of MA200 over last 3 months
    ma_63ago = float(ma200.iloc[-(VCP_REGIME_SLOPE_DAYS + 1)])
    slope_pct = (ma_now - ma_63ago) / ma_63ago * 100 if ma_63ago > 0 else 0

    pct_above_ma = (price / ma_now - 1) * 100 if ma_now > 0 else 0

    if pct_above_ma > 2 and slope_pct > 1:
        regime = "bull"
        mult   = 1.0 + VCP_REGIME_BULL_BONUS
    elif pct_above_ma < -10 and slope_pct < -2:
        regime = "bear"
        mult   = 1.0 + VCP_REGIME_BEAR_PENALTY
    elif -10 <= pct_above_ma <= 2 and abs(slope_pct) < 3:
        regime = "sideways"
        mult   = 0.95
    else:
        regime = "correction"
        mult   = 0.90

    return regime, round(mult, 3)


# ─── SEPA / Trend Template ────────────────────────────────────────────────────

def _sepa_check(result: VCPPattern, daily: pd.DataFrame) -> float:
    """
    Returns SEPA score 0-1 and populates result trend template fields.
    No hard rejection — every partial pass contributes partial score.
    """
    close = daily["Close"]
    if len(close) < VCP_TREND_MA_LONG + VCP_MA200_UPTREND_LOOKBACK + 10:
        return 0.0

    price     = float(close.iloc[-1])
    ma50      = float(close.rolling(VCP_TREND_MA_SHORT).mean().iloc[-1])
    ma150     = float(close.rolling(VCP_TREND_MA_MID).mean().iloc[-1])
    ma200     = float(close.rolling(VCP_TREND_MA_LONG).mean().iloc[-1])
    ma200_ago = float(close.rolling(VCP_TREND_MA_LONG).mean().iloc[-(VCP_MA200_UPTREND_LOOKBACK+1)])

    hi52  = float(daily["High"].rolling(252).max().iloc[-1])
    lo52  = float(daily["Low"].rolling(252).min().iloc[-1])

    criteria = {
        "price_above_ma50":   price > ma50,
        "price_above_ma150":  price > ma150,
        "price_above_ma200":  price > ma200,
        "ma150_above_ma200":  ma150 > ma200,
        "ma50_above_ma150":   ma50 > ma150,
        "ma50_above_ma200":   ma50 > ma200,
        "ma200_uptrend":      ma200 > ma200_ago,
        "near_52wk_high":     hi52 > 0 and price >= hi52 * (1 - VCP_52WK_HIGH_MAX_PCT_BELOW / 100),
        "above_52wk_low":     lo52 > 0 and (price / lo52 - 1) * 100 >= VCP_52WK_LOW_MIN_PCT_ABOVE,
    }

    result.trend_template_details = criteria
    result.trend_template_ok      = all(criteria.values())
    result.price_above_ma50       = criteria["price_above_ma50"]
    result.price_above_ma150      = criteria["price_above_ma150"]
    result.price_above_ma200      = criteria["price_above_ma200"]
    result.ma150_above_ma200      = criteria["ma150_above_ma200"]
    result.ma200_uptrend          = criteria["ma200_uptrend"]
    result.ma50_above_ma150       = criteria["ma50_above_ma150"]
    result.ma50_above_ma200       = criteria["ma50_above_ma200"]
    result.pct_above_52wk_low     = round((price/lo52-1)*100, 1) if lo52 > 0 else 0.0
    result.pct_below_52wk_high    = round((1-price/hi52)*100, 1) if hi52 > 0 else 100.0

    passed = sum(1 for v in criteria.values() if v)
    sepa   = passed / len(criteria)
    result.sepa_score = round(sepa, 3)

    # Explanation
    failed = [k for k, v in criteria.items() if not v]
    if failed:
        result.diagnostic.soft_warnings.append(
            f"SEPA: {len(failed)} criteria not met: {failed}"
        )
    return sepa


# ─── Prior Uptrend ────────────────────────────────────────────────────────────

def _prior_uptrend(result: VCPPattern, daily: pd.DataFrame,
                   base_start: Optional[date] = None) -> float:
    """
    Returns advance_pct (0-inf). Fills result.prior_uptrend_pct.
    Not a hard gate — low advance is a penalty, not a rejection.
    """
    lookback = min(VCP_PRIOR_UPTREND_LOOKBACK_DAYS, len(daily) - 1)
    prior    = daily["Close"].iloc[-(lookback + 1):-1]
    if prior.empty:
        return 0.0

    prior_low = float(prior.min())
    # Use base start high if available, else current price
    if base_start:
        ts = pd.Timestamp(base_start)
        before_base = daily[daily.index < ts]["Close"]
        base_hi = float(before_base.iloc[-1]) if not before_base.empty else float(daily["Close"].iloc[-1])
    else:
        base_hi = float(daily["Close"].iloc[-1])

    pct = (base_hi / prior_low - 1) * 100 if prior_low > 0 else 0.0
    result.prior_uptrend_pct = round(pct, 1)
    result.prior_uptrend_ok  = pct >= VCP_PRIOR_UPTREND_MIN_PCT
    return pct


# ─── Window Builder ───────────────────────────────────────────────────────────

def _build_windows(n: int, n_c: int, strategy: str, min_b: int) -> Optional[list]:
    if strategy == "equal":
        ws = n // n_c
        if ws < min_b:
            return None
        windows = []
        for i in range(n_c):
            s = i * ws
            e = (s + ws - 1) if i < n_c - 1 else n - 1
            windows.append((s, e))
        return windows
    elif strategy == "progressive":
        ratios     = [1.0 - 0.5 * i / max(n_c - 1, 1) for i in range(n_c)]
        total      = sum(ratios)
        base_size  = n / total
        sizes      = [max(min_b, round(base_size * r)) for r in ratios]
        sizes[-1]  = n - sum(sizes[:-1])
        if sizes[-1] < min_b:
            return None
        windows, cursor = [], 0
        for s in sizes:
            windows.append((cursor, cursor + s - 1))
            cursor += s
        return windows
    return None


def _score_candidate(contractions: list[dict]) -> float:
    """Score a contraction candidate (0-100). Used to pick best window split."""
    if len(contractions) < 2:
        return 0.0
    widths  = [c["width_pct"]  for c in contractions]
    volumes = [c["avg_volume"] for c in contractions]
    durs    = [c["duration_bars"] for c in contractions]
    score   = 0.0

    # Width tightening (50 pts) — adaptive tolerance
    tight = sum(
        1 for i in range(1, len(widths))
        if widths[i] <= widths[i-1] * (VCP_TIGHTENING_RATIO + VCP_TIGHTENING_TOLERANCE)
    )
    score += (tight / max(len(widths) - 1, 1)) * 50

    # Final width tightness (20 pts)
    fw = widths[-1]
    if fw <= 5:    score += 20
    elif fw <= 8:  score += 16
    elif fw <= 12: score += 10
    elif fw <= VCP_MAX_LAST_WIDTH_PCT: score += 4

    # Volume declining slope (20 pts)
    if len(volumes) >= 2:
        x  = list(range(len(volumes)))
        xm = sum(x) / len(x)
        vm = sum(volumes) / len(volumes)
        cov = sum((xi-xm)*(vi-vm) for xi, vi in zip(x, volumes))
        var = sum((xi-xm)**2 for xi in x)
        slope = cov / var if var > 0 else 0
        if slope < 0:
            red = abs(slope * len(volumes)) / (vm if vm > 0 else 1)
            score += min(20.0, red * 40)

    # Duration shortening (10 pts)
    short = sum(1 for i in range(1, len(durs)) if durs[i] <= durs[i-1])
    score += (short / max(len(durs)-1, 1)) * 10
    return round(score, 1)


# ─── Multi-Scale Window Finder ────────────────────────────────────────────────

def _find_best_contractions(bars: pd.DataFrame,
                            lookback_list: list[int]) -> tuple[list[dict], str, int]:
    """
    Try every lookback in lookback_list and every N=2..6 with equal/progressive.
    Returns (best_contractions, best_strategy, best_lookback).
    This is the key upgrade: multi-scale prevents missing long bases.
    """
    best_contractions: list[dict] = []
    best_score: float             = -1.0
    best_strategy: str            = ""
    best_lookback: int            = 0

    n_total = len(bars)

    for lookback in lookback_list:
        lb = min(lookback, n_total - 1)
        segment = bars.iloc[-lb:] if lb < n_total else bars
        n = len(segment)

        for n_c in range(VCP_MIN_CONTRACTIONS, VCP_MAX_CONTRACTIONS + 1):
            for strategy in ("equal", "progressive"):
                windows = _build_windows(n, n_c, strategy, VCP_MIN_WINDOW_BARS)
                if windows is None:
                    continue

                candidate: list[dict] = []
                valid = True

                for idx, (s, e) in enumerate(windows):
                    w   = segment.iloc[s: e + 1]
                    dur = len(w)
                    if dur < VCP_MIN_WINDOW_BARS or dur > VCP_MAX_WINDOW_BARS:
                        valid = False
                        break

                    hi_val  = float(w["High"].max())
                    lo_val  = float(w["Low"].min())
                    cl_hi   = float(w["Close"].max())
                    mid     = (hi_val + lo_val) / 2.0
                    wid_pct = (hi_val - lo_val) / mid * 100 if mid > 0 else 0.0
                    avg_vol = float(w["Volume"].mean())

                    if wid_pct < VCP_MIN_WIDTH_PCT:
                        valid = False
                        break

                    candidate.append({
                        "index":         idx + 1,
                        "start_date":    w.index[0].date(),
                        "end_date":      w.index[-1].date(),
                        "high":          round(hi_val, 2),
                        "low":           round(lo_val, 2),
                        "close_high":    round(cl_hi, 2),
                        "width_pct":     round(wid_pct, 1),
                        "duration_bars": dur,
                        "avg_volume":    round(avg_vol, 0),
                        "volume_ratio":  0.0,
                    })

                if not valid or len(candidate) < VCP_MIN_CONTRACTIONS:
                    continue

                c1_vol = candidate[0]["avg_volume"]
                for c in candidate:
                    c["volume_ratio"] = round(c["avg_volume"] / c1_vol, 3) if c1_vol > 0 else 1.0

                sc = _score_candidate(candidate)
                if sc > best_score:
                    best_score        = sc
                    best_contractions = [dict(c) for c in candidate]
                    best_strategy     = strategy
                    best_lookback     = lookback

    return best_contractions, best_strategy, best_lookback


# ─── Single-Timeframe VCP ─────────────────────────────────────────────────────

def _detect_on_timeframe(bars: pd.DataFrame,
                         lookback_list: list[int],
                         tf_name: str) -> TimeframeResult:
    """
    Run full VCP contraction detection on a single timeframe's bars.
    Returns a TimeframeResult (found=False if no valid structure found).
    """
    res = TimeframeResult(timeframe=tf_name)

    if bars is None or len(bars) < 30:
        return res

    raw, strategy, lookback = _find_best_contractions(bars, lookback_list)

    if len(raw) < VCP_MIN_CONTRACTIONS:
        return res

    widths = [c["width_pct"] for c in raw]
    lows   = [c["low"]       for c in raw]
    vols   = [c["avg_volume"] for c in raw]
    durs   = [c["duration_bars"] for c in raw]

    # Width tightening check (adaptive tolerance)
    tight = all(
        raw[i]["width_pct"] <= raw[i-1]["width_pct"] * (VCP_TIGHTENING_RATIO + VCP_TIGHTENING_TOLERANCE)
        for i in range(1, len(raw))
    )

    # Higher lows check (advisory, not hard gate)
    hl_ok = all(
        lows[i] >= lows[i-1] * (1 - VCP_HIGHER_LOWS_TOLERANCE)
        for i in range(1, len(lows))
    )

    # Volume slope
    vol_slope = False
    if len(vols) >= 2:
        x   = list(range(len(vols)))
        xm  = sum(x) / len(x)
        vm  = sum(vols) / len(vols)
        cov = sum((xi-xm)*(vi-vm) for xi, vi in zip(x, vols))
        var = sum((xi-xm)**2 for xi in x)
        vol_slope = (cov / var) < 0 if var > 0 else False

    # Pivot from closing prices
    final_c = raw[-1]
    start   = pd.Timestamp(final_c["start_date"])
    end     = pd.Timestamp(final_c["end_date"])
    w       = bars[(bars.index >= start) & (bars.index <= end)]
    pivot     = float(w["Close"].max()) if not w.empty else final_c["high"]
    pivot_low = float(w["Low"].min())   if not w.empty else final_c["low"]

    # Hard reject: first contraction too wide (crash) or all contractions same width
    if raw[0]["width_pct"] > VCP_MAX_FIRST_WIDTH_PCT:
        return res
    if max(widths) - min(widths) < 1.0:   # no real compression at all
        return res

    cs = _score_candidate(raw)
    res.found                  = tight and cs >= 30   # at least 30/100 geometry score
    res.t_count                = len(raw)
    res.contractions           = raw
    res.contraction_widths     = widths
    res.contraction_lows       = lows
    res.contractions_tightening = tight
    res.higher_lows_ok         = hl_ok
    res.volume_slope_negative  = vol_slope
    res.window_strategy        = strategy
    res.lookback_bars          = lookback
    res.candidate_score        = cs
    res.pivot_price            = round(pivot, 2)
    res.pivot_low              = round(pivot_low, 2)
    return res


# ─── Breakout Check ───────────────────────────────────────────────────────────

def _check_breakout(result: VCPPattern, daily: pd.DataFrame,
                    benchmark: Optional[pd.DataFrame]) -> None:
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

    result.breakout_volume_ratio = round(
        last_vol / avg_vol_50, 2
    ) if avg_vol_50 > 0 else 0.0

    above_pivot = last_close >= result.pivot_price
    buy_zone_hi = result.pivot_price * (1 + VCP_BUY_ZONE_PCT / 100)
    in_buy_zone = last_close <= buy_zone_hi

    result.is_breaking_out = above_pivot and in_buy_zone
    result.is_extended     = last_close > buy_zone_hi

    if result.is_extended:
        result.diagnostic.soft_warnings.append(
            f"Extended: {((last_close/result.pivot_price)-1)*100:.1f}% above pivot — outside buy zone"
        )

    # RS new high
    if benchmark is not None:
        try:
            bench   = benchmark["Close"].reindex(daily.index, method="ffill").dropna()
            stock   = daily["Close"].reindex(bench.index)
            rs_line = stock / bench * 100
            rs_now  = float(rs_line.iloc[-1])
            prior   = rs_line.iloc[-273:-21] if len(rs_line) >= 273 else rs_line.iloc[:-21]
            result.rs_new_high = rs_now >= float(prior.max()) if not prior.empty else False
        except Exception:
            result.rs_new_high = False


# ─── Composite Quality Score ──────────────────────────────────────────────────

def _compute_quality_score(result: VCPPattern, vol_analysis: dict) -> ScoreBreakdown:
    """
    Data-validated scoring weights from backtest analysis:
      RS Rating:      highest win correlation (0.218) → 15 pts
      SEPA template:  WR boost +9.6pp → 20 pts (highest single factor)
      Score band 70+: 45% WR vs 30% below 60 → validates score discriminates

    Returns ScoreBreakdown with per-factor explanations.
    """
    W  = VCP_SCORE_WEIGHTS
    sb = ScoreBreakdown()
    ex = {}

    # 1. Prior uptrend (18 pts max)
    pct = result.prior_uptrend_pct
    if pct >= 100:
        pts = W["prior_uptrend"]
    elif pct >= 50:
        pts = W["prior_uptrend"] * 0.85
    elif pct >= VCP_PRIOR_UPTREND_MIN_PCT:
        pts = W["prior_uptrend"] * 0.65
    elif pct >= 10:
        pts = W["prior_uptrend"] * 0.30
    else:
        pts = 0.0
    sb.prior_uptrend = round(pts, 1)
    ex["prior_uptrend"] = f"{pct:.1f}% advance → {pts:.1f}/{W['prior_uptrend']} pts"

    # 2. Contraction quality (18 pts max)
    # T-count sub-score (backtest: 5T dominates but 6T underperforms)
    tc = result.t_count
    if tc == 4 or tc == 5:
        tc_pts = W["contraction_quality"] * 0.50
    elif tc == 3:
        tc_pts = W["contraction_quality"] * 0.40
    elif tc == 2:
        tc_pts = W["contraction_quality"] * 0.25
    elif tc == 6:
        tc_pts = W["contraction_quality"] * 0.15   # 6T slightly overextended
    else:
        tc_pts = 0.0

    # Final width sub-score
    fw = result.contractions[-1].width_pct if result.contractions else 15.0
    if fw <= 5:    fw_pts = W["contraction_quality"] * 0.50
    elif fw <= 8:  fw_pts = W["contraction_quality"] * 0.40
    elif fw <= 12: fw_pts = W["contraction_quality"] * 0.25
    elif fw <= 20: fw_pts = W["contraction_quality"] * 0.10
    else:          fw_pts = 0.0

    cq_pts = tc_pts + fw_pts
    if not result.contractions_tightening:
        cq_pts *= 0.5   # penalty, not rejection
    sb.contraction_quality = round(min(cq_pts, W["contraction_quality"]), 1)
    ex["contraction_quality"] = (
        f"{tc}T, final width {fw:.1f}%, tightening={result.contractions_tightening} "
        f"→ {sb.contraction_quality}/{W['contraction_quality']} pts"
    )

    # 3. SEPA template (20 pts max — highest single factor from backtest)
    sepa_pts = round(result.sepa_score * W["sepa_template"], 1)
    sb.sepa_template = sepa_pts
    passed_n = sum(1 for v in result.trend_template_details.values() if v)
    ex["sepa_template"] = (
        f"{passed_n}/{len(result.trend_template_details)} criteria pass "
        f"→ {sepa_pts}/{W['sepa_template']} pts"
    )

    # 4. RS Rating (15 pts max — highest correlation 0.218)
    rs = result.rs_rating
    if rs >= 90:       rs_pts = W["rs_rating"]
    elif rs >= 80:     rs_pts = W["rs_rating"] * 0.85
    elif rs >= VCP_RS_PREFERRED: rs_pts = W["rs_rating"] * 0.70
    elif rs >= 60:     rs_pts = W["rs_rating"] * 0.45
    elif rs >= VCP_RS_MIN: rs_pts = W["rs_rating"] * 0.20
    else:              rs_pts = 0.0
    if result.rs_new_high:
        rs_pts = min(rs_pts * 1.15, W["rs_rating"])   # new high bonus
    sb.rs_rating = round(rs_pts, 1)
    ex["rs_rating"] = f"RS={rs:.0f}{'(new high)' if result.rs_new_high else ''} → {sb.rs_rating}/{W['rs_rating']} pts"

    # 5. Volume dry-up (10 pts max)
    vd_pts = vol_analysis["dryup_score"] * W["volume_dryup"] * 0.5
    vd_pts += vol_analysis["trend_score"] * W["volume_dryup"] * 0.3
    if vol_analysis["vdu_day"]:
        vd_pts += W["volume_dryup"] * 0.2
    sb.volume_dryup = round(min(vd_pts, W["volume_dryup"]), 1)
    ex["volume_dryup"] = (
        f"dryup={vol_analysis['dryup_score']:.2f}, trend={vol_analysis['trend_score']:.2f}, "
        f"VDU={vol_analysis['vdu_day']} → {sb.volume_dryup}/{W['volume_dryup']} pts"
    )
    result.vdu_day_present = vol_analysis["vdu_day"]
    result.volume_dryup_score = vol_analysis["dryup_score"]

    # 6. Volatility compression (8 pts max)
    vc_pts = round(result.vol_compression_score * W["volatility_compression"], 1)
    sb.volatility_compression = vc_pts
    ex["volatility_compression"] = (
        f"ATR ratio={result.atr_compression_ratio:.3f}, "
        f"composite={result.vol_compression_score:.2f} "
        f"→ {vc_pts}/{W['volatility_compression']} pts"
    )

    # 7. Breakout quality (6 pts max)
    bk_pts = 0.0
    if result.is_breaking_out:
        bk_pts += W["breakout_quality"] * 0.5
        if result.breakout_volume_ratio >= 2.0:
            bk_pts += W["breakout_quality"] * 0.5
        elif result.breakout_volume_ratio >= 1.5:
            bk_pts += W["breakout_quality"] * 0.35
        elif result.breakout_volume_ratio >= VCP_BREAKOUT_VOLUME_RATIO:
            bk_pts += W["breakout_quality"] * 0.20
    sb.breakout_quality = round(min(bk_pts, W["breakout_quality"]), 1)
    ex["breakout_quality"] = (
        f"breaking_out={result.is_breaking_out}, vol_ratio={result.breakout_volume_ratio:.2f} "
        f"→ {sb.breakout_quality}/{W['breakout_quality']} pts"
    )

    # 8. Market regime (5 pts max)
    if result.market_regime == "bull":
        rg_pts = W["market_regime"]
    elif result.market_regime == "correction":
        rg_pts = W["market_regime"] * 0.5
    elif result.market_regime in ("bear", "sideways"):
        rg_pts = 0.0
    else:
        rg_pts = W["market_regime"] * 0.5
    sb.market_regime = round(rg_pts, 1)
    ex["market_regime"] = f"regime={result.market_regime} → {rg_pts}/{W['market_regime']} pts"

    # 9. Timeframe alignment bonus (up to +8 pts)
    if result.tf_combo_bonus > 1.0:
        bonus = round((result.tf_combo_bonus - 1.0) * 40, 1)   # e.g. 1.20 → 8 pts
        sb.timeframe_bonus = bonus
        ex["timeframe_bonus"] = f"TF combo {result.active_timeframes} → +{bonus} pts"

    # Total
    raw_total = (
        sb.prior_uptrend + sb.contraction_quality + sb.sepa_template +
        sb.rs_rating + sb.volume_dryup + sb.volatility_compression +
        sb.breakout_quality + sb.market_regime + sb.timeframe_bonus
    )
    # Apply regime multiplier
    adjusted = raw_total * result.regime_multiplier
    sb.total = round(min(adjusted, 100.0), 1)
    sb.explanations = ex

    return sb


# ─── Main Detection Function ──────────────────────────────────────────────────

def detect_vcp(
    symbol:    str,
    daily:     pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    rs_rating: float = 50.0,
) -> Optional[VCPPattern]:
    """
    Institutional-grade VCP detector.
    Returns VCPPattern with full diagnostic, or None if hard-rejected.

    Hard reject only when:
      - Insufficient history (< 200 bars)
      - Volume too low (< MIN_AVG_VOLUME)
      - ATR ratio > 0.75 (no compression whatsoever)
      - No valid contraction structure found on any timeframe/lookback

    Everything else is a penalty on the score, not a rejection.
    """
    if daily is None or len(daily) < 200:
        return None

    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    result     = VCPPattern(symbol=symbol)
    diagnostic = VCPDiagnostic(symbol=symbol, scan_date=date.today())
    result.diagnostic = diagnostic

    # ── ATR pre-filter (only hard-reject if truly no compression) ─────────────
    passes_atr, atr_ratio, atr_score = _atr_compression(daily)
    result.atr_compression_ratio = atr_ratio
    if not passes_atr:
        diagnostic.rejection_reasons.append(
            f"ATR ratio {atr_ratio:.3f} > {VCP_ATR_COMPRESSION_MAX} — no volatility compression"
        )
        diagnostic.nearest_threshold = "VCP_ATR_COMPRESSION_MAX"
        diagnostic.threshold_gap_pct = round((atr_ratio - VCP_ATR_COMPRESSION_MAX) * 100, 1)
        diagnostic.improvement_hint  = (
            f"Raising VCP_ATR_COMPRESSION_MAX to {atr_ratio + 0.05:.2f} would include this stock"
        )
        diagnostic.decision = "Rejected"
        result.diagnostic   = diagnostic
        return result   # return pattern with diagnostic, is_valid=False

    # ── Composite volatility compression ──────────────────────────────────────
    result.vol_compression_score = _volatility_compression(daily)

    # ── Market regime ─────────────────────────────────────────────────────────
    regime, regime_mult          = _market_regime(benchmark)
    result.market_regime         = regime
    result.regime_multiplier     = regime_mult

    # ── Resample to weekly and monthly ────────────────────────────────────────
    try:
        weekly  = resample_weekly(daily)
    except Exception:
        weekly  = None
    try:
        monthly = resample_monthly(daily)
    except Exception:
        monthly = None

    # ── Multi-timeframe detection ─────────────────────────────────────────────
    tf_daily   = _detect_on_timeframe(daily,   VCP_LOOKBACK_WINDOWS,         "daily")
    tf_weekly  = _detect_on_timeframe(weekly,  VCP_WEEKLY_LOOKBACK_WINDOWS,  "weekly")  if weekly  is not None else TimeframeResult(timeframe="weekly")
    tf_monthly = _detect_on_timeframe(monthly, VCP_MONTHLY_LOOKBACK_WINDOWS, "monthly") if monthly is not None else TimeframeResult(timeframe="monthly")

    result.tf_daily   = tf_daily
    result.tf_weekly  = tf_weekly
    result.tf_monthly = tf_monthly

    active_tfs = []
    if tf_daily.found:   active_tfs.append("daily")
    if tf_weekly.found:  active_tfs.append("weekly")
    if tf_monthly.found: active_tfs.append("monthly")

    result.active_timeframes = active_tfs

    # Must have at least daily VCP
    if not tf_daily.found:
        diagnostic.rejection_reasons.append(
            "No valid VCP contraction structure found on daily timeframe "
            f"across {len(VCP_LOOKBACK_WINDOWS)} lookback windows"
        )
        # Record the best near-miss for diagnostics
        diagnostic.improvement_hint = (
            "Try lowering VCP_TIGHTENING_RATIO or VCP_MIN_CONTRACTIONS if stock "
            "visually shows a VCP but algorithm can't find a valid contraction sequence"
        )
        diagnostic.decision = "Rejected"
        result.diagnostic   = diagnostic
        return result

    # ── Timeframe combo bonus ─────────────────────────────────────────────────
    combo_key = tuple(sorted(active_tfs))
    result.tf_combo_bonus = VCP_TF_COMBO_BONUS.get(combo_key, 1.0)

    # ── Best timeframe = daily (always primary) ───────────────────────────────
    best = tf_daily
    result.best_timeframe          = "daily"
    result.contractions            = [
        VCPContraction(
            index=c["index"], start_date=c["start_date"], end_date=c["end_date"],
            high=c["high"], low=c["low"], close_high=c["close_high"],
            width_pct=c["width_pct"], duration_bars=c["duration_bars"],
            avg_volume=c["avg_volume"], volume_ratio=c["volume_ratio"],
        ) for c in best.contractions
    ]
    result.t_count                 = best.t_count
    result.window_strategy         = best.window_strategy
    result.lookback_bars           = best.lookback_bars
    result.contractions_tightening = best.contractions_tightening
    result.contractions_shortening = all(
        best.contractions[i]["duration_bars"] <= best.contractions[i-1]["duration_bars"]
        for i in range(1, len(best.contractions))
    )
    result.volume_slope_negative   = best.volume_slope_negative
    result.higher_lows_ok          = best.higher_lows_ok
    result.pivot_price             = best.pivot_price
    result.pivot_low               = best.pivot_low
    result.buy_zone_high           = round(best.pivot_price * (1 + VCP_BUY_ZONE_PCT / 100), 2)

    # ── Base start and duration ───────────────────────────────────────────────
    if best.contractions:
        result.base_start_date     = best.contractions[0]["start_date"]
        days                       = (daily.index[-1].date() - result.base_start_date).days
        result.base_duration_weeks = days // 7

    # ── SEPA / Trend Template ─────────────────────────────────────────────────
    _sepa_check(result, daily)

    # ── Prior uptrend ─────────────────────────────────────────────────────────
    _prior_uptrend(result, daily, result.base_start_date)

    # ── Volume analysis ───────────────────────────────────────────────────────
    vol_analysis = _volume_analysis(daily, best.contractions)
    result.volume_dryup_score = vol_analysis["dryup_score"]
    result.vdu_day_present    = vol_analysis["vdu_day"]
    result.volume_trend_score = vol_analysis["trend_score"]
    result.accumulation_score = vol_analysis["accumulation"]

    # ── RS rating ─────────────────────────────────────────────────────────────
    result.rs_rating = rs_rating

    # ── Breakout ──────────────────────────────────────────────────────────────
    _check_breakout(result, daily, benchmark)

    # ── Quality score ─────────────────────────────────────────────────────────
    sb = _compute_quality_score(result, vol_analysis)
    result.score_breakdown = sb
    result.quality_score   = sb.total

    # ── Validity: no hard gates except structure ───────────────────────────────
    result.is_valid = (
        tf_daily.found
        and result.t_count >= VCP_MIN_CONTRACTIONS
    )

    # ── Populate diagnostic ───────────────────────────────────────────────────
    diagnostic.overall_score   = result.quality_score
    diagnostic.confidence_pct  = round(min(result.quality_score, 100.0), 1)
    diagnostic.t_count         = result.t_count
    diagnostic.contraction_widths = [c.width_pct for c in result.contractions]
    diagnostic.contraction_lows   = [c.low       for c in result.contractions]
    diagnostic.best_timeframe     = result.best_timeframe
    diagnostic.score_breakdown    = sb

    # Build passed/failed lists for explainability
    for factor, explanation in sb.explanations.items():
        max_pts = VCP_SCORE_WEIGHTS.get(factor, 0)
        actual  = getattr(sb, factor, 0)
        if actual >= max_pts * 0.7:
            diagnostic.checks_passed.append(f"✓ {factor}: {explanation}")
        elif actual >= max_pts * 0.3:
            diagnostic.penalties_applied.append(f"~ {factor} (partial): {explanation}")
        else:
            diagnostic.checks_failed.append(f"✗ {factor}: {explanation}")

    if result.is_valid:
        diagnostic.decision = "Breaking Out" if result.is_breaking_out else (
            "Near Pivot" if daily["Close"].iloc[-1] >= result.pivot_price * 0.95
            else "Watching"
        )
    else:
        diagnostic.decision = "Rejected"

    # Active timeframes
    diagnostic.tf_results = {
        "daily":   {"found": tf_daily.found,   "t_count": tf_daily.t_count,   "score": tf_daily.candidate_score},
        "weekly":  {"found": tf_weekly.found,  "t_count": tf_weekly.t_count,  "score": tf_weekly.candidate_score},
        "monthly": {"found": tf_monthly.found, "t_count": tf_monthly.t_count, "score": tf_monthly.candidate_score},
    }

    log.debug(
        "%s VCP: %dT TFs=%s valid=%s score=%.1f atr=%.2f regime=%s",
        symbol, result.t_count, active_tfs, result.is_valid,
        result.quality_score, atr_ratio, regime,
    )
    return result


def _volatility_compression(daily: pd.DataFrame) -> float:
    """Thin wrapper so it's accessible at module level."""
    return _volatility_compression_score(daily)
