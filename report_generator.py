"""
report_generator.py (v2) - Report Generator Agent.

Changes from v1
  * Altman cut-offs come from the variant stored in SCORE_RUN.model_version (no hard-coded 1.81 / 2.99).
  * The FR-22 'Score Deterioration' flag written by quant_scorer now contributes to the quantitative stream.
  * Processes every run (all companies, all years), not only run_id DESC.
  * Financial-sector runs (FR-20) are marked 'Not Applicable' instead of 'Low Risk'.
  * Missing scores are not treated as safe: the run is reported as partial.

The penalty sizes and band boundaries below are CALIBRATION CHOICES (SRS TBD-5): agree them with Nirali.

Usage:  py report_generator.py          (every company, one line per year)
        py report_generator.py 1        (company_id 1, with full flag list for the latest year)
"""
import sqlite3
import sys

import pandas as pd

from chart_of_accounts import ALTMAN_VARIANT, ALTMAN_VARIANTS, BENEISH_THRESHOLD

DB_PATH = "data/red_shift.db"

# Stream weights (SDS Component 5, Eq. 6) - must sum to 1
WEIGHT_QUANT, WEIGHT_QUAL, WEIGHT_MARKET = 0.40, 0.35, 0.25

# Quantitative stream penalties (each stream is capped at 1.0)
PEN_M_BREACH = 0.50
PEN_Z_DISTRESS = 0.30
PEN_Z_GREY = 0.15
PEN_F_WEAK = 0.20
PEN_DETERIORATION = 0.15
F_WEAK = 3

# Severity bands (FR-44)
BAND_HIGH, BAND_MODERATE = 65.0, 35.0

assert abs(WEIGHT_QUANT + WEIGHT_QUAL + WEIGHT_MARKET - 1.0) < 1e-9


def altman_cutoffs(model_version):
    """Pull the Altman variant out of e.g. 'v2.1-features/z_double_prime'."""
    parts = (model_version or "").split("/")
    variant = parts[1] if len(parts) > 1 and parts[1] in ALTMAN_VARIANTS else ALTMAN_VARIANT
    return variant, ALTMAN_VARIANTS[variant]["cutoffs"]


def band_for(score):
    if score >= BAND_HIGH:
        return "High Risk"
    return "Moderate Risk" if score >= BAND_MODERATE else "Low Risk"


def score_run(cur, run):
    run_id, _, fy, status, m, z, f, model_version = run

    # FR-20: banks / NBFCs / insurers get no composite
    if (model_version or "").endswith("/not-applicable"):
        cur.execute("UPDATE SCORE_RUN SET composite_score = NULL, severity_band = 'Not Applicable' WHERE run_id = ?",
                    (run_id,))
        return {"run_id": run_id, "fy": fy, "applicable": False}

    # ---- Q: quantitative stream ----
    variant, (z_lo, z_hi) = altman_cutoffs(model_version)
    q = 0.0
    if m is not None and m > BENEISH_THRESHOLD:
        q += PEN_M_BREACH
    if z is not None:
        if z < z_lo:
            q += PEN_Z_DISTRESS
        elif z <= z_hi:
            q += PEN_Z_GREY
    if f is not None and f <= F_WEAK:
        q += PEN_F_WEAK
    det = cur.execute("SELECT COUNT(*) FROM RED_FLAG WHERE run_id = ? AND flag_type LIKE 'Score Deterioration%'",
                      (run_id,)).fetchone()[0]
    if det:
        q += PEN_DETERIORATION
    q = min(q, 1.0)

    # ---- L: qualitative stream ----
    sev_pen = {"High": 0.35, "Medium": 0.20, "Low": 0.10}
    rows = cur.execute("SELECT severity FROM QUAL_FINDING WHERE run_id = ? AND verification_status = 'Verified'",
                       (run_id,)).fetchall()
    l = min(sum(sev_pen.get(s, 0.0) for (s,) in rows), 1.0)

    # ---- B: market stream ----
    b = 0.0
    for ftype, _sev in cur.execute("SELECT flag_type, severity FROM RED_FLAG WHERE run_id = ? AND category = 'Market-Behaviour'",
                                   (run_id,)).fetchall():
        if "Divergence" in ftype:
            b += 0.60
        if "Volume" in ftype:
            b += 0.40
    b = min(b, 1.0)

    composite = round(100.0 * (WEIGHT_QUANT * q + WEIGHT_QUAL * l + WEIGHT_MARKET * b), 2)
    band = band_for(composite)
    cur.execute("UPDATE SCORE_RUN SET composite_score = ?, severity_band = ? WHERE run_id = ?", (composite, band, run_id))
    return {"run_id": run_id, "fy": fy, "applicable": True, "q": q, "l": l, "b": b, "composite": composite,
            "band": band, "partial": None in (m, z, f), "variant": variant}


def run_report_generator(company_id=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    companies = cur.execute("SELECT company_id, symbol FROM COMPANY ORDER BY company_id").fetchall()
    if company_id is not None:
        companies = [c for c in companies if c[0] == company_id]

    for cid, symbol in companies:
        runs = cur.execute("""SELECT run_id, company_id, fiscal_year, status, m_score, z_score, f_score, model_version
                              FROM SCORE_RUN WHERE company_id = ? ORDER BY fiscal_year, run_id""", (cid,)).fetchall()
        if not runs:
            continue
        print(f"\n=== {symbol} ===")
        results = [score_run(cur, r) for r in runs]
        for r in results:
            if not r["applicable"]:
                print(f"  FY{r['fy']}: not applicable (financial sector)")
            else:
                note = "  [partial: a score was missing]" if r["partial"] else ""
                print(f"  FY{r['fy']}: Q={r['q']*100:5.1f}  L={r['l']*100:5.1f}  B={r['b']*100:5.1f}  "
                      f"-> composite {r['composite']:5.1f}/100  {r['band']}{note}")

        if company_id is not None:                      # detail for the latest year
            latest = results[-1]
            flags = pd.read_sql_query("SELECT category, flag_type, severity, evidence_ref FROM RED_FLAG WHERE run_id = ?",
                                      conn, params=(latest["run_id"],))
            print(f"\n  Flags for FY{latest['fy']} (run #{latest['run_id']}): {len(flags)}")
            for _, fr in flags.iterrows():
                print(f"   [{fr['severity']}] {fr['category']} -> {fr['flag_type']}")
                print(f"        Evidence: {fr['evidence_ref']}")

    conn.commit()
    conn.close()
    print("\nComposite scores and severity bands saved to SCORE_RUN.")


if __name__ == "__main__":
    run_report_generator(int(sys.argv[1]) if len(sys.argv) > 1 else None)