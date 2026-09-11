"""Chapter 10, Steps 3.8 - 3.9 — attribution by feature family and by V group."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain import config as C

TABLES = Path("reports/tables/ch10")
ANON   = "Anonymised V (retained)"


def load_shap(name="valid"):
    z = np.load(C.SHAP_DIR / f"{name}.npz", allow_pickle=False)
    return z["shap"], float(z["base"]), [str(f) for f in z["feature_names"]]


def main():
    sv, base, names = load_shap("valid")
    fmap   = json.loads((C.ARTIFACT_DIR / "feature_map.json").read_text())
    family, vgroup = fmap["feature_family"], fmap["v_group"]
    vmeta  = {g["group"]: g for g in
              json.loads((C.ARTIFACT_DIR / "v_groups.json").read_text())["groups"]}
    col = {f: i for i, f in enumerate(names)}
    mean_abs_feat = np.abs(sv).mean(0)

    def agg(members):
        ix = [col[c] for c in members]
        signed = sv[:, ix].sum(1)                       # exact, by additivity
        return (float(np.abs(signed).mean()),           # sum-then-absolute (reported)
                float(mean_abs_feat[ix].sum()),         # absolute-then-sum (comparison)
                signed)

    # --- Table 10.1, by family -----------------------------------------
    rows, fam_signed = [], {}
    for fam in sorted(set(family.values())):
        members = [f for f in names if family[f] == fam]
        sta, ats, signed = agg(members)
        fam_signed[fam] = signed
        rows.append({"family": fam, "n_features": len(members),
                     "mean_abs_shap": sta, "abs_then_sum": ats,
                     "top_member": max(members, key=lambda c: mean_abs_feat[col[c]])})
    t1 = pd.DataFrame(rows)
    t1["share"] = t1.mean_abs_shap / t1.mean_abs_shap.sum()
    t1 = t1.sort_values("share", ascending=False).reset_index(drop=True)

    print("=== attribution by feature family, validation partition "
          f"(n={sv.shape[0]:,}) ===")
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- named vs anonymised, on a common basis ------------------------
    ia  = [col[f] for f in names if family[f] == ANON]
    inn = [col[f] for f in names if family[f] != ANON]
    a, n = sv[:, ia].sum(1), sv[:, inn].sum(1)
    ma, mn = float(np.abs(a).mean()), float(np.abs(n).mean())
    ta, tn = float(mean_abs_feat[ia].sum()), float(mean_abs_feat[inn].sum())
    block = {
        "headline_basis": "two-block sum-then-absolute",
        "n_named": len(inn), "n_anonymised": len(ia),
        "named_share_two_block": mn / (mn + ma),
        "anon_share_two_block":  ma / (mn + ma),
        "named_share_abs_then_sum": tn / (tn + ta),
        "anon_share_abs_then_sum":  ta / (tn + ta),
        "named_share_family_normalised": float(t1.loc[t1.family != ANON, "share"].sum()),
        "cancellation_named": mn / tn,
        "cancellation_anon":  ma / ta,
    }
    print("\n=== named vs anonymised ===")
    for k, v in block.items():
        print(f"  {k:32s} {v if isinstance(v, str) else round(v, 4)}")
    (C.ARTIFACT_DIR / "summary.json").write_text(json.dumps(block, indent=2))

    # --- Table 10.2, by V group ----------------------------------------
    rows = []
    for g, meta in vmeta.items():
        members = [c for c in names if vgroup.get(c) == g]
        sta, ats, _ = agg(members)
        rows.append({"group": g, "n_columns": len(members),
                     "missing_share": meta["missing_share"],
                     "mean_abs_shap": sta, "abs_then_sum": ats})
    t2 = pd.DataFrame(rows)
    t2["share_of_anon"] = t2.mean_abs_shap / t2.mean_abs_shap.sum()
    t2 = t2.sort_values("missing_share").reset_index(drop=True)

    # exact check: signed group sums must reconstruct the anonymised block per row
    anon_cols = [col[c] for c in names if family[c] == ANON]
    recon = sum(sv[:, [col[c] for c in names if vgroup.get(c) == g]].sum(1)
                for g in vmeta)
    resid = float(np.abs(recon - sv[:, anon_cols].sum(1)).max())
    print(f"\n=== anonymised block, thirteen groups ===")
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\ngroup-sum reconstruction residual: {resid:.1e} (expect < 1e-4)")
    if resid > 1e-4:
        raise SystemExit("ABORT: V groups do not partition the anonymised block")

    # --- Step 3.10, ranking agreement -----------------------------------
    from scipy.stats import spearmanr
    from src.explain.load import load_estimator

    est  = load_estimator()
    gain = pd.Series(est.booster_.feature_importance("gain"),
                     index=est.booster_.feature_name())
    shp  = pd.Series(mean_abs_feat, index=names)
    auc  = (pd.read_csv("reports/tables/table_8_4_single_feature_auc.csv")
              .set_index("feature")["auc"])

    top50  = shp.nlargest(50).index
    common = shp.index.intersection(auc.index)
    overlap = len(set(shp.nlargest(10).index) & set(gain.nlargest(10).index))
    t5 = pd.DataFrame([
        {"comparison": "mean|SHAP| vs split gain (all features)",
         "n": len(shp), "spearman": spearmanr(shp, gain.reindex(shp.index))[0]},
        {"comparison": "mean|SHAP| vs split gain (top 50 by SHAP)",
         "n": 50, "spearman": spearmanr(shp[top50], gain[top50])[0]},
        {"comparison": "mean|SHAP| vs single-feature AUC (Ch.8 top-30 only)",
         "n": len(common), "spearman": spearmanr(shp[common], auc[common])[0]},
        {"comparison": "top-10 overlap, SHAP vs gain", "n": 10, "spearman": overlap / 10},
    ])
    print("\n=== ranking agreement ===")
    print(t5.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"top-10 overlap, SHAP vs gain: {overlap}/10")


    TABLES.mkdir(parents=True, exist_ok=True)
    t1.to_csv(TABLES / "table_10_1_families.csv", index=False)
    t2.to_csv(TABLES / "table_10_2_v_groups.csv", index=False)
    t5.to_csv(TABLES / "table_10_5_ranking_agreement.csv", index=False)
    print(f"\nwrote {TABLES}/table_10_1_families.csv and table_10_2_v_groups.csv")


if __name__ == "__main__":
    main()