"""Compare headline metrics of two final-eval runs (CI reproducibility check)."""

from __future__ import annotations

import json
import math
import sys

KEYS = [
    ("headline", "model_mae_kwh"),
    ("headline", "model_rmse_kwh"),
    ("headline", "improvement_vs_primary"),
    ("headline", "scored_hours"),
    ("uncertainty", "coverage"),
    ("uncertainty", "median_width_kwh"),
    ("advisory", "flags_as_served", "f1"),
]


def get(d, path):
    for k in path:
        d = d[k]
    return d


def main(a_path: str, b_path: str, rtol: float = 1e-6) -> int:
    a, b = (json.load(open(p, encoding="utf-8")) for p in (a_path, b_path))
    bad = 0
    for path in KEYS:
        x, y = get(a, path), get(b, path)
        ok = math.isclose(float(x), float(y), rel_tol=rtol, abs_tol=1e-9)
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {'.'.join(path):45s} committed={x} regenerated={y}")
    if a["frozen_decisions_sha256"] != b["frozen_decisions_sha256"]:
        print("DIFF frozen decisions hash")
        bad += 1
    print("reproduction matches" if not bad else f"{bad} metric(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
