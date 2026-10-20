import sqlite3
import sys

import pandas as pd

from chart_of_accounts import BENEISH_THRESHOLD, PIOTROSKI_SIGNALS
from features import build_features

DB_PATH = "data/red_shift.db"
OUT_CSV = "RedShift_Feature_Dataset.csv"
F_SCORE_WEAK_FOR_LABEL = 4
INDEX_NAMES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]


def rule_label(m, f, z, safe_cutoff):
    if m is None or f is None or z is None:
        return None
    return int(m > BENEISH_THRESHOLD or f <= F_SCORE_WEAK_FOR_LABEL or z < safe_cutoff)


def export(db_path=DB_PATH, out_csv=OUT_CSV):
    conn = sqlite3.connect(db_path)
    companies = conn.execute("SELECT company_id, symbol, sector, is_financial FROM COMPANY ORDER BY company_id").fetchall()
    conn.close()

    rows = []
    for cid, symbol, sector, is_fin in companies:
        feats = build_features(company_id=cid, db_path=db_path)
        for fy, r in sorted(feats["years"].items()):
            if not r["has_prior_year"]:
                continue
            b, a, p = r["beneish"], r["altman"], r["piotroski"]
            row = {"symbol": symbol, "sector": sector, "fiscal_year": fy,
                   "model_applicable": 0 if is_fin else 1}
            if not is_fin:
                row.update({k: b.get("indices", {}).get(k) for k in INDEX_NAMES})
                m, z, f = b.get("m_score"), a.get("z_score"), p.get("f_score")
                row.update({"m_score": m, "altman_variant": a.get("variant"), "z_score": z,
                            "altman_zone": a.get("zone"), "f_score": f})
                row.update({f"F_{k}": p.get("signals", {}).get(k) for k in PIOTROSKI_SIGNALS})
                safe = a["cutoffs"][1] if a.get("cutoffs") else None
                row["rule_label"] = rule_label(m, f, z, safe) if safe is not None else None
                row["missing_inputs"] = "; ".join(b.get("missing", [])[:3] + a.get("missing", [])[:3])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    scored = df[df["model_applicable"] == 1]
    print(f"Saved {out_csv}: {len(df)} company-years ({len(scored)} non-financial)")
    print(f"  fully scored (M, Z and F present): {int(scored['rule_label'].notna().sum())}")
    print(f"  rule_label = 1 (risky): {int((scored['rule_label'] == 1).sum())}   = 0 (safe): {int((scored['rule_label'] == 0).sum())}")
    print("  rows per company:", df.groupby("symbol").size().to_dict())
    return df


if __name__ == "__main__":
    export(*(sys.argv[1:2]))