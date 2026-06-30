"""
VCP Scanner — Daily Live Scan Excel Report
============================================
Generates a clean, single-sheet Excel workbook for TODAY's live VCP signals.
This is separate from backtest_vcp_report.py, which reports on historical
backtest runs. This module reports on the live scanner's current output.

Called automatically from main.py after every live scan (when signals exist),
regardless of whether --save was passed — the daily Excel report is always
produced so users have something to open even without a database.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config import REPORTS_DIR
from logger_utils import get_logger

log = get_logger("scanner")

CLR = {
    "header_bg":      "1F4E79",
    "header_fg":      "FFFFFF",
    "breaking_out":   "00B050",
    "near_pivot":     "FFEB9C",
    "watching":       "FCE4D6",
    "alt_row":        "EBF3FB",
    "white":          "FFFFFF",
    "border":         "BDD7EE",
}
THIN   = Side(style="thin", color=CLR["border"])
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

STATUS_COLOR = {
    "Breaking Out": CLR["breaking_out"],
    "Near Pivot":   CLR["near_pivot"],
    "Watching":     CLR["watching"],
}


def generate_daily_vcp_report(signals: list, scan_date: date | None = None) -> Path | None:
    """
    Build a single-sheet Excel workbook for today's live VCP signals.
    Returns the saved path, or None if there are no signals to report.

    signals — list of VCPSignal objects from vcp_scanner.scan_vcp()
    """
    if not signals:
        log.info("No VCP signals to report — skipping daily Excel report")
        return None

    scan_date = scan_date or date.today()

    # Sort: Breaking Out first, then Near Pivot, then Watching, all by score desc
    order = {"Breaking Out": 0, "Near Pivot": 1, "Watching": 2}
    ordered = sorted(
        signals,
        key=lambda s: (order.get(s.status, 3), -s.quality_score),
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "VCP Signals"
    ws.sheet_view.showGridLines = False

    n_breaking = sum(1 for s in ordered if s.status == "Breaking Out")
    n_near     = sum(1 for s in ordered if s.status == "Near Pivot")
    n_watch    = sum(1 for s in ordered if s.status == "Watching")

    # Title row
    ws.merge_cells("A1:N1")
    ws["A1"] = (
        f"VCP Live Scan — {scan_date.isoformat()}   |   "
        f"{len(ordered)} signals   |   "
        f"🚀 {n_breaking} Breaking Out   |   👀 {n_near} Near Pivot   |   📊 {n_watch} Watching"
    )
    ws["A1"].font      = Font(bold=True, size=13, color="FFFFFF")
    ws["A1"].fill      = PatternFill("solid", start_color=CLR["header_bg"])
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26

    headers = [
        "Status", "Symbol", "Sector", "T-Count", "Quality Score",
        "Pivot Price", "Current Price", "Buy Zone High", "Stop Loss",
        "Target 1", "Target 2", "R:R Ratio", "RS Rating",
        "Final Width %", "ATR Compression", "Trend Template",
        "VDU Day", "Higher Lows", "Timeframes", "Market Regime",
        "Position Size", "Capital Required",
    ]

    hdr_row = 3
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=hdr_row, column=c, value=h)
        cell.font      = Font(bold=True, color=CLR["header_fg"])
        cell.fill      = PatternFill("solid", start_color=CLR["header_bg"])
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = BORDER
    ws.row_dimensions[hdr_row].height = 30

    for i, s in enumerate(ordered):
        r = hdr_row + 1 + i
        row_vals = [
            s.status,
            s.symbol,
            s.sector,
            f"{s.t_count}T",
            round(s.quality_score, 1),
            s.pivot_price,
            s.current_price,
            s.buy_zone_high,
            s.stop_loss,
            s.target1,
            s.target2,
            s.rr_ratio,
            round(s.rs_rating, 1),
            round(s.final_contraction_width, 1),
            round(s.atr_compression_ratio, 2),
            "Yes" if s.trend_template_ok else "No",
            "Yes" if s.vdu_day_present else "No",
            "Yes" if s.higher_lows_ok else "No",
            "+".join(t[0].upper() for t in (s.active_timeframes or [])),
            s.market_regime,
            s.position_size,
            s.capital_required,
        ]
        for c, val in enumerate(row_vals, 1):
            cell           = ws.cell(row=r, column=c, value=val)
            cell.border    = BORDER
            cell.alignment = Alignment(horizontal="center")
            bg = CLR["alt_row"] if i % 2 == 1 else CLR["white"]
            cell.fill      = PatternFill("solid", start_color=bg)

        # Colour the status cell distinctly
        status_cell       = ws.cell(row=r, column=1)
        status_cell.fill  = PatternFill(
            "solid", start_color=STATUS_COLOR.get(s.status, CLR["white"])
        )
        status_cell.font  = Font(bold=True)

    # Auto width
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value)) if cell.value is not None else 0)
            for cell in col
        )
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 3, 28)

    ws.freeze_panes = f"A{hdr_row + 1}"

    out_path = REPORTS_DIR / f"vcp_daily_scan_{scan_date.isoformat()}.xlsx"
    wb.save(out_path)
    log.info("Daily VCP scan report saved → %s", out_path)
    return out_path
