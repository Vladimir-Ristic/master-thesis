"""Chapter 10 — the two aggregation maps, written once and read in all rebuilds."""
import json

from src.explain import config as C
from src.explain.load import load_estimator
from src.features.build_features import _feature_family

V_GROUPS = C.ARTIFACT_DIR / "v_groups.json"
OUT      = C.ARTIFACT_DIR / "feature_map.json"
ANON     = "Anonymised V (retained)"


def build():
    names  = load_estimator().booster_.feature_name()
    family = {f: _feature_family(f) for f in names}

    vmap = {}
    for g in json.loads(V_GROUPS.read_text())["groups"]:
        for c in g["columns"]:
            if c in vmap:
                raise SystemExit(f"ABORT: {c} appears in two V groups")
            vmap[c] = g["group"]

    # protocol checks 5 and 6, asserted at build time rather than trusted
    if len(family) != 290:
        raise SystemExit(f"ABORT: family map covers {len(family)} features, expected 290")
    anon = {f for f, fam in family.items() if fam == ANON}
    if anon != set(vmap):
        raise SystemExit(f"ABORT: V groups cover {len(vmap)} columns, model has {len(anon)}")

    named = sorted(f for f in names if family[f] != ANON)
    OUT.write_text(json.dumps({
        "n_features": len(names),
        "n_named": len(named), "n_anonymised": len(anon),
        "families": sorted(set(family.values())),
        "n_v_groups": len({*vmap.values()}),
        "feature_family": family,
        "v_group": vmap,
        "feature_order": names,
    }, indent=2))

    print(f"families      : {len(set(family.values()))}")
    print(f"V groups      : {len(set(vmap.values()))}")
    print(f"named/derived : {len(named)}  | anonymised: {len(anon)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()