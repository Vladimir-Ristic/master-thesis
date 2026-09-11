"""Chapter 10, Step 3.13 — ten protocol checks. All must pass."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from src.explain import config as C
from src.explain.aggregate import load_shap, TABLES
from src.explain.load import load_estimator
from src.models import data

ANON = "Anonymised V (retained)"
RESULTS = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            RESULTS.append((True, name, detail))
        except Exception as e:
            RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
    return deco


def main():
    sv, base, names = load_shap("valid")
    fmap = json.loads((C.ARTIFACT_DIR / "feature_map.json").read_text())
    family, vgroup = fmap["feature_family"], fmap["v_group"]
    est = load_estimator()
    valid = pd.read_parquet(C.DATA_DIR / "valid_v1.parquet")
    margin = est.predict(valid[names], raw_score=True)

    @check("1. test partition still sealed")
    def _():
        try:
            data.load_partition("test")
        except PermissionError:
            return "load_partition('test') raises PermissionError"
        raise AssertionError("test partition was readable")

    @check("2. additivity holds on every explained row")
    def _():
        r = float(np.abs(sv.sum(1) + base - margin).max())
        assert r < 1e-4, f"max residual {r:.1e}"
        return f"max residual {r:.1e}"

    @check("3. base + mean(sum phi) equals mean raw margin")
    def _():
        d = abs(base + sv.sum(1).mean() - margin.mean())
        assert d < 1e-4, f"gap {d:.1e}"
        return f"gap {d:.1e}"

    @check("4. explained model is fraud_detector v1")
    def _():
        b, p = est.booster_, est.get_params()
        assert b.num_trees() == 750 and b.num_feature() == 290
        assert p["num_leaves"] == 250 and p["min_child_samples"] == 186
        assert list(b.feature_name()) == list(names)
        return "750 trees, 290 features, 250 leaves, feature order matches"

    @check("5. thirteen groups partition the 143 retained columns")
    def _():
        anon = [f for f in names if family[f] == ANON]
        assert len(anon) == 143 and set(anon) == set(vgroup)
        assert len(set(vgroup.values())) == 13
        return "143 columns, 13 groups, no column in two groups"

    @check("6. family map covers all 290 features exactly once")
    def _():
        assert len(family) == 290 and set(family) == set(names)
        return f"290 features, {len(set(family.values()))} families"

    @check("7. family signed sums reconstruct the row total")
    def _():
        col = {f: i for i, f in enumerate(names)}
        recon = sum(sv[:, [col[f] for f in names if family[f] == fam]].sum(1)
                    for fam in set(family.values()))
        r = float(np.abs(recon - sv.sum(1)).max())
        assert r < 1e-4, f"max residual {r:.1e}"
        return f"max residual {r:.1e}"

    @check("8. no validation row entered the selection sample")
    def _():
        z = np.load(C.SHAP_DIR / "train_sample.npz", allow_pickle=False)
        tr = pd.read_parquet(C.DATA_DIR / "train_v1.parquet")
        picked = set(tr.iloc[z["row_index"]].transactionid)
        overlap = picked & set(valid.transactionid)
        assert not overlap, f"{len(overlap)} rows overlap"
        return f"{len(picked):,} selection rows, 0 in validation"

    @check("9. compact models scored under the Chapter 9 rule")
    def _():
        t = pd.read_csv(TABLES / "table_10_4_compact.csv").set_index("k")
        assert abs(t.loc[290, "pr_auc"] - 0.598894) < 1e-4
        assert abs(t.loc[290, "review_rate"] - 0.187980) < 1e-3
        return "k=290 reproduces PR-AUC 0.598894 and review rate 0.18798"

    @check("10. calibrator strictly monotone on validation scores")
    def _():
        cal = json.load(open("artifacts/models_v1/selection_summary.json"))["calibrator"]
        assert cal["a"] > 0
        p = expit(cal["a"] * margin + cal["b"])
        assert np.all(np.diff(p[np.argsort(margin)]) >= 0)
        return f"a = {cal['a']:.6f} > 0, ordering preserved"

    width = max(len(n) for _, n, _ in RESULTS)
    for ok, name, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    n_ok = sum(ok for ok, _, _ in RESULTS)
    print(f"\n{'all 10 protocol checks passed' if n_ok == 10 else f'{n_ok}/10 passed'}")
    if n_ok != 10:
        raise SystemExit(1)


if __name__ == "__main__":
    main()