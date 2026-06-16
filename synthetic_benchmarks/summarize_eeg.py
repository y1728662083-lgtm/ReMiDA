#!/usr/bin/env python
"""Summarize EEG experiment outputs produced by runners/eeg.py.

Usage:
  python summarize_eeg.py run/logs/<doc>/eeg_summary.json

Prints mean±std across seeds for:
  - session leakage probe accuracy
  - cross-session direction decoding accuracy
  - identity consistency vs ICA anchor (perm/sign agreement), if available
  - anchor match score, if available
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) == 0:
        return None, None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to eeg_summary.json")
    args = ap.parse_args()

    p = Path(args.path)
    obj = json.loads(p.read_text(encoding="utf-8"))

    seeds = obj.get("seeds", [])

    leak_acc = []
    decode_mean = []
    perm_agree = []
    sign_agree = []
    anchor_match = []

    for s in seeds:
        leak = (s.get("session_leakage") or {})
        leak_acc.append(leak.get("acc"))

        dec = s.get("decode_direction")
        decode_mean.append(None if dec is None else dec.get("mean"))

        ident = s.get("identity_vs_anchor")
        if ident is not None:
            perm_agree.append(ident.get("perm_agreement"))
            sign_agree.append(ident.get("sign_agreement"))

        am = s.get("anchor_match")
        if am is not None:
            anchor_match.append(am.get("mean"))

    m, sd = _mean_std(leak_acc)
    print(f"session_leakage_acc mean±std: {m} ± {sd} (n={len([x for x in leak_acc if x is not None])})")

    m, sd = _mean_std(decode_mean)
    print(f"decode_direction mean±std: {m} ± {sd} (n={len([x for x in decode_mean if x is not None])})")

    if len(perm_agree) > 0:
        m, sd = _mean_std(perm_agree)
        print(f"perm_agreement_vs_anchor mean±std: {m} ± {sd}")
    if len(sign_agree) > 0:
        m, sd = _mean_std(sign_agree)
        print(f"sign_agreement_vs_anchor mean±std: {m} ± {sd}")
    if len(anchor_match) > 0:
        m, sd = _mean_std(anchor_match)
        print(f"anchor_match_score mean±std: {m} ± {sd}")


if __name__ == "__main__":
    main()
