"""
VCP Scanner — Main Entry Point
================================
A standalone runner for the VCP (Volatility Contraction Pattern) scanner.
No Darvas Box. No Cup & Handle. VCP only.

Commands:
  python main.py                          — Run live VCP scan, save signals, print table
  python main.py --download               — Download / update NSE price data first
  python main.py --backtest RELIANCE.NS   — Backtest one symbol
  python main.py --backtest-all           — Backtest full NSE universe
  python main.py --backtest-all --limit 100  — Quick test on first 100 symbols
  python main.py --report RUN_ID          — Generate Excel report for a backtest run
  python main.py --list-runs              — List all saved backtest runs
  python main.py --debug RELIANCE.NS      — Deep diagnosis for one symbol
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from config import NIFTY50_SYMBOL, NIFTY500_SYMBOL
from database import init_db, save_vcp_signal, get_vcp_signals_df
from downloader import load_daily, run_download
from indicators import raw_weighted_return, cross_sectional_rs_rating
from logger_utils import get_logger
from universe import fetch_nse_symbols
from vcp_scanner import scan_vcp

import pandas as pd

log = get_logger("scanner")


# ─── Live scan ────────────────────────────────────────────────────────────────

def run_scan(symbols: list[str], benchmark: pd.DataFrame) -> tuple[list, list, list, dict]:
    """
    Scan the full universe for VCP signals.
    Returns (signals, watchlist, all_vcps, scan_meta).
      signals   — score >= VCP_MIN_QUALITY_SCORE
      watchlist — score >= VCP_WATCHLIST_MIN_SCORE and < VCP_MIN_QUALITY_SCORE
      all_vcps  — every detected VCP regardless of score
      scan_meta — run statistics dict
    """
    from config import VCP_MIN_QUALITY_SCORE, VCP_WATCHLIST_MIN_SCORE
    import time
    t_start = time.time()
    log.info("VCP scan: computing RS ratings for %d symbols …", len(symbols))

    # Cross-sectional RS (true O'Neil method — rank every stock vs every other)
    weighted_returns = {}
    cached_daily     = {}
    for sym in symbols:
        daily = load_daily(sym)
        if daily is None or len(daily) < 200:
            continue
        cached_daily[sym] = daily
        wr = raw_weighted_return(daily["Close"])
        if wr is not None:
            weighted_returns[sym] = wr

    rs_series = cross_sectional_rs_rating(pd.Series(weighted_returns))
    log.info("RS ratings computed for %d symbols", len(rs_series))

    signals   = []
    watchlist = []
    all_vcps  = []
    errors    = 0

    for i, (sym, daily) in enumerate(cached_daily.items(), 1):
        if i % 100 == 0:
            log.info("  Scanning %d/%d — signals: %d  watchlist: %d",
                     i, len(cached_daily), len(signals), len(watchlist))
        try:
            rs_val = float(rs_series.get(sym, 50.0))
            sig = scan_vcp(sym, daily, benchmark, rs_override=rs_val)
            if sig is not None:
                all_vcps.append(sig)
                if sig.quality_score >= VCP_MIN_QUALITY_SCORE:
                    signals.append(sig)
                elif sig.quality_score >= VCP_WATCHLIST_MIN_SCORE:
                    watchlist.append(sig)
        except Exception as e:
            errors += 1
            log.debug("Scan error %s: %s", sym, e)

    elapsed = round(time.time() - t_start, 1)
    scan_meta = {
        "symbols_scanned":   len(symbols),
        "symbols_with_data": len(cached_daily),
        "patterns_detected": len(all_vcps),
        "elapsed_secs":      elapsed,
        "market_regime":     all_vcps[0].market_regime if all_vcps else "unknown",
        "benchmark":         "^NSEI",
    }
    log.info("Scan complete in %.1fs: %d signals, %d watchlist, %d total VCPs (%d errors)",
             elapsed, len(signals), len(watchlist), len(all_vcps), errors)
    return signals, watchlist, all_vcps, scan_meta


def _print_signals(signals: list) -> None:
    if not signals:
        print("\n  No VCP signals found today.\n")
        return

    # Sort: Breaking Out first, then Near Pivot, then by score desc
    order = {"Breaking Out": 0, "Near Pivot": 1, "Watching": 2}
    signals.sort(key=lambda s: (order.get(s.status, 3), -s.quality_score))

    print(f"\n{'='*100}")
    print(f"  VCP SIGNALS — {date.today()}  ({len(signals)} found)")
    print(f"{'='*100}")
    fmt = "{:<14} {:>4}T {:>6} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>14}"
    print(fmt.format(
        "Symbol", "T", "Score", "Pivot", "CurrPx", "Stop", "Target2",
        "RS", "Width%", "Status"
    ))
    print("-" * 100)
    for s in signals:
        print(fmt.format(
            s.symbol[:13],
            s.t_count,
            f"{s.quality_score:.1f}",
            f"{s.pivot_price:.2f}",
            f"{s.current_price:.2f}",
            f"{s.stop_loss:.2f}",
            f"{s.target2:.2f}",
            f"{s.rs_rating:.0f}",
            f"{s.final_contraction_width:.1f}",
            s.status,
        ))
    print(f"{'='*100}\n")


# ─── Debug single symbol ──────────────────────────────────────────────────────

def debug_symbol(symbol: str, benchmark: pd.DataFrame) -> None:
    from vcp import detect_vcp

    print(f"\n=== VCP DEBUG: {symbol} ===")
    daily = load_daily(symbol)
    if daily is None:
        print("No data found — run --download first")
        return

    print(f"  History: {len(daily)} bars  ({daily.index[0].date()} → {daily.index[-1].date()})")
    print(f"  Last close: {daily['Close'].iloc[-1]:.2f}")
    print(f"  Avg 20d volume: {daily['Volume'].iloc[-20:].mean():,.0f}")
    print()

    pattern = detect_vcp(symbol, daily, benchmark=benchmark)

    if pattern is None:
        print("  Result: No VCP detected (ATR pre-filter rejected OR insufficient history)")
        return

    print(f"  ATR compression ratio : {pattern.atr_compression_ratio:.3f} (need ≤ 0.60)")
    print(f"  Trend template        : {'✅ PASS' if pattern.trend_template_ok else '❌ FAIL'}")
    if not pattern.trend_template_ok:
        failed = [k for k, v in pattern.trend_template_details.items() if not v]
        print(f"    Failed criteria     : {failed}")
    print(f"  Prior uptrend         : {pattern.prior_uptrend_pct:.1f}% "
          f"({'✅' if pattern.prior_uptrend_ok else '❌'} need ≥30%)")
    print(f"  T-count               : {pattern.t_count}T")

    if pattern.contractions:
        print(f"  Contraction widths    : {[c.width_pct for c in pattern.contractions]}")
        print(f"  Contraction lows      : {[c.low for c in pattern.contractions]}")
        print(f"  Width tightening      : {'✅' if pattern.contractions_tightening else '❌'}")
        print(f"  Higher lows           : {'✅' if pattern.higher_lows_ok else '❌'}")
        print(f"  Volume slope negative : {'✅' if pattern.volume_slope_negative else '❌'}")
        print(f"  VDU day present       : {'✅' if pattern.vdu_day_present else '❌'}")
        print(f"  Pivot price (close)   : {pattern.pivot_price:.2f}")
        print(f"  Pivot low (intraday)  : {pattern.pivot_low:.2f}")
        print(f"  Buy zone              : {pattern.pivot_price:.2f} → {pattern.buy_zone_high:.2f}")

    print(f"  Is breaking out       : {'✅ YES' if pattern.is_breaking_out else '❌ NO'}")
    print(f"  Is extended           : {'⚠️  YES (do not chase)' if pattern.is_extended else 'No'}")
    print(f"  Breakout vol ratio    : {pattern.breakout_volume_ratio:.2f}× (need ≥1.30)")
    print(f"  Quality score         : {pattern.quality_score:.1f}/100")
    print(f"  IS VALID              : {'✅ YES' if pattern.is_valid else '❌ NO'}")

    if pattern.rejection_reasons:
        print(f"\n  Hard rejection reasons:")
        for r in pattern.rejection_reasons:
            print(f"    • {r}")
    if pattern.soft_warnings:
        print(f"\n  Soft warnings:")
        for w in pattern.soft_warnings:
            print(f"    ⚠️  {w}")
    print()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_benchmark() -> pd.DataFrame:
    for sym in [NIFTY50_SYMBOL, NIFTY500_SYMBOL]:
        df = load_daily(sym)
        if df is not None and len(df) > 100:
            log.info("Benchmark: %s (%d bars)", sym, len(df))
            return df
    log.warning("No benchmark — RS defaults to 50.0")
    idx = pd.date_range(end=date.today(), periods=300, freq="B")
    return pd.DataFrame({"Close": [10000.0] * 300}, index=idx)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VCP (Volatility Contraction Pattern) Scanner for NSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--download",      action="store_true",
                   help="Download / update NSE price data before scanning")
    p.add_argument("--full-refresh",  action="store_true",
                   help="Re-download full history (used with --download)")
    p.add_argument("--no-telegram",   action="store_true",
                   help="Skip Telegram notification after scan")
    p.add_argument("--save",          action="store_true",
                   help="Save signals to database (default: print only)")
    p.add_argument("--backtest",      type=str, default=None, metavar="SYMBOL",
                   help="Backtest VCP on a single symbol")
    p.add_argument("--backtest-all",  action="store_true",
                   help="Backtest VCP across full NSE universe")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limit --backtest-all to first N symbols (quick test)")
    p.add_argument("--report",        type=str, default=None, metavar="RUN_ID",
                   help="Generate Excel analysis report for a backtest run_id")
    p.add_argument("--list-runs",     action="store_true",
                   help="List all saved backtest runs")
    p.add_argument("--debug",         type=str, default=None, metavar="SYMBOL",
                   help="Print detailed VCP diagnosis for one symbol")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    init_db()

    # ── Debug single symbol ───────────────────────────────────────────────────
    if args.debug:
        sym = args.debug.upper()
        if not sym.endswith(".NS"):
            sym += ".NS"
        benchmark = _load_benchmark()
        debug_symbol(sym, benchmark)
        sys.exit(0)

    # ── Single-symbol backtest ────────────────────────────────────────────────
    if args.backtest:
        from backtest import backtest_vcp_symbol
        sym = args.backtest.upper()
        if not sym.endswith(".NS"):
            sym += ".NS"
        benchmark = _load_benchmark()
        log.info("VCP backtest: %s", sym)
        summary, trades = backtest_vcp_symbol(sym, benchmark=benchmark)
        if summary:
            print(f"\n=== VCP BACKTEST: {sym} ===")
            for k, v in summary.items():
                print(f"  {k:25s}: {v}")
            print(f"\n  Trades: {len(trades)}")
            if trades:
                df = pd.DataFrame(trades)
                print(f"  Win rate : {(df['rr_realised'] > 0).mean() * 100:.1f}%")
                print(f"  Avg R    : {df['rr_realised'].mean():.3f}")
                print(f"  Outcomes : {df['outcome'].value_counts().to_dict()}")
        else:
            print(f"No VCP results for {sym} — need ≥260 days history and at least one valid pattern")
        sys.exit(0)

    # ── Universe backtest ─────────────────────────────────────────────────────
    if args.backtest_all:
        from backtest import backtest_universe_vcp
        from backtest_vcp_report import generate_vcp_report

        log.info("=" * 60)
        log.info("VCP UNIVERSE BACKTEST — NSE")
        log.info("=" * 60)

        symbols = fetch_nse_symbols(force_refresh=False)
        if args.limit:
            symbols = symbols[: args.limit]
            log.info("Limited to first %d symbols", len(symbols))

        # Download first if requested
        if args.download:
            benchmark_syms = [NIFTY50_SYMBOL, NIFTY500_SYMBOL]
            run_download(benchmark_syms + symbols, full_refresh=args.full_refresh)

        vcp_summary = backtest_universe_vcp(
            symbols, notes=f"limit={args.limit or 'none'}"
        )

        print("\n=== VCP BACKTEST SUMMARY ===")
        for k, v in vcp_summary.items():
            print(f"  {k:25s}: {v}")

        if vcp_summary.get("total_trades", 0) > 0:
            try:
                paths = generate_vcp_report(vcp_summary["run_id"])
                for p in paths:
                    print(f"\n  ✅ Excel report saved: {p}")
                print(
                    "\n  Sheets to check first:\n"
                    "    1. Verdict          — overall pass/fail\n"
                    "    2. T-Count Analysis — does 3T beat 2T?\n"
                    "    3. Contraction Analysis — tighter = better?\n"
                    "    4. Score Band Analysis — does the score predict outcomes?"
                )
            except Exception as e:
                log.error("Report generation failed: %s", e)
        else:
            print("\n  No trades — check ATR compression threshold and history length")
        sys.exit(0)

    # ── Generate report for existing run ────────────────────────────────────
    if args.report:
        from backtest_vcp_report import generate_vcp_report
        try:
            paths = generate_vcp_report(args.report)
            for p in paths:
                print(f"\n  ✅ VCP report saved: {p}")
        except ValueError as e:
            print(f"\n  ⚠️  {e}")
            print("     Use --list-runs to see available run IDs")
        sys.exit(0)

    # ── List saved backtest runs ──────────────────────────────────────────────
    if args.list_runs:
        from database import get_backtest_runs_df
        df = get_backtest_runs_df()
        if df.empty:
            print("No backtest runs saved yet. Run --backtest-all first.")
        else:
            cols = ["run_id", "run_date", "symbols_tested", "total_trades",
                    "win_rate", "profit_factor", "expectancy", "notes"]
            cols = [c for c in cols if c in df.columns]
            print("\n=== SAVED VCP BACKTEST RUNS ===")
            print(df[cols].to_string(index=False))
        sys.exit(0)

    # ── Live scan (default action) ────────────────────────────────────────────
    t0 = time.time()
    log.info("=" * 60)
    log.info("VCP SCANNER  |  %s", date.today())
    log.info("=" * 60)

    # Optionally download first
    if args.download:
        log.info("Downloading price data …")
        symbols_for_dl = fetch_nse_symbols(force_refresh=False)
        run_download(
            [NIFTY50_SYMBOL, NIFTY500_SYMBOL] + symbols_for_dl,
            full_refresh=args.full_refresh,
        )

    symbols   = fetch_nse_symbols(force_refresh=False)
    benchmark = _load_benchmark()

    signals, watchlist, all_vcps, scan_meta = run_scan(symbols, benchmark)
    _print_signals(signals)

    # Always generate the daily Excel report (Signals + Watchlist + Stats)
    # Works even when 0 signals — Watchlist sheet shows forming patterns
    try:
        from vcp_daily_report import generate_daily_vcp_report
        report_path = generate_daily_vcp_report(
            signals=signals,
            watchlist=watchlist,
            all_vcps=all_vcps,
            scan_meta=scan_meta,
        )
        if report_path:
            print(f"\n  ✅ Daily Excel report: {report_path}")
            print(f"     Signals: {len(signals)}  |  Watchlist: {len(watchlist)}  |  Total VCPs: {len(all_vcps)}\n")
    except Exception as e:
        log.error("Daily Excel report failed: %s", e)
        print(f"\n  ⚠️  Could not generate daily Excel report: {e}\n")

    # Save to DB if requested
    if args.save and signals:
        for sig in signals:
            try:
                save_vcp_signal(sig)
            except Exception as e:
                log.error("Save error %s: %s", sig.symbol, e)
        log.info("Saved %d VCP signals to database", len(signals))

    # Telegram
    if not args.no_telegram and signals:
        try:
            from telegram_notify import _send_telegram_message
            from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                breaking = [s for s in signals if s.is_breaking_out]
                near     = [s for s in signals if s.status == "Near Pivot"]
                msg_lines = [
                    f"🎯 *VCP Scanner — {date.today()}*",
                    f"Total signals: {len(signals)} | Breaking out: {len(breaking)} | Near pivot: {len(near)}",
                    "",
                ]
                for s in signals[:15]:  # cap at 15 for Telegram
                    icon = "🚀" if s.is_breaking_out else "👀" if s.status == "Near Pivot" else "📊"
                    msg_lines.append(
                        f"{icon} *{s.symbol}* {s.t_count}T | Score: {s.quality_score:.0f} | "
                        f"Pivot: {s.pivot_price:.2f} | RS: {s.rs_rating:.0f}"
                    )
                _send_telegram_message("\n".join(msg_lines))
        except Exception as e:
            log.error("Telegram error: %s", e)

    elapsed = time.time() - t0
    log.info("Done in %.1fs | %d VCP signals", elapsed, len(signals))
