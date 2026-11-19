"""Routing evaluation (v3): frozen System-1 (dev-trained, calibrated), real cheap-LLM S2 answers on the
243 held-out items, thresholds chosen by NESTED grouped CV (no held-out item selects its own threshold).

Contract: label_only (the only contract with real S2 answers on this split). All label_only items are
admissible for S1, so the admissibility gate is a no-op here; feasibility violations are reported
separately (violations_offline.py, which is contract-level and needs no S2 answers).

Policies (routed correctness = correctness of the chosen system):
  R0  always-S2                       (no S1)
  R1  always-S1                       (no escalation)
  R5  global calibrated-confidence threshold tau   (nested)
  H1  legacy HARP rule: d-hat <= 0.45 AND conf >= 0.65 (fixed, from v2; uses the NEW calibrated conf)
  H3  HARP-v3: admissibility gate (no-op here) + per-family calibrated threshold tau_f (nested)
  H3d HARP-v3 + difficulty gate d-hat <= 0.45 (ablation: does the difficulty term help?)
  ORC oracle (S1 if correct, else S2) - an upper bound, not a policy

Output: logs/route_v3_summary.md, logs/route_v3_rows.jsonl, logs/route_v3.json
"""
import json
import math
import os
import random
import time

import numpy as np
from sklearn.model_selection import GroupKFold

import clean_protocol as CP
import s1_v3 as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "logs", "route_v3_summary.md")
OUT_ROWS = os.path.join(HERE, "logs", "route_v3_rows.jsonl")
OUT_JSON = os.path.join(HERE, "logs", "route_v3.json")
PILOT = os.path.join(HERE, "logs", "pilot_live_log.jsonl")
FAMS = ["intent", "sentiment", "nli", "multihop", "factoid", "math"]
TAU_GRID = [0.0] + [round(x, 2) for x in np.arange(0.50, 0.96, 0.025)] + [1.01]
TAU_D = 0.45
TAU_LEGACY = 0.65


def d_hat(it):
    """Difficulty formula of the paper (Eq. 2), as in harp.py: hops, ambiguity, knowledge, prompt length (chars//4 > 200)."""
    hops = it.get("hops", 1)
    d = 0.18 * hops + 0.20 * (it.get("ambiguity") == "high") + 0.15 * (it.get("knowledge") == "external")
    d += 0.10 * (len(str(it.get("prompt", ""))) // 4 > 200)   # prompt-token term, as in harp.py
    return min(1.0, round(d, 3))


def build_eval(frozen):
    held, _, _, _ = CP.build()
    pilot = {}
    for l in open(PILOT):
        r = json.loads(l)
        pilot[r["item_id"]] = r
    rows = []
    for fam in FAMS:
        fr = [h for h in held if h["family"] == fam]
        if not fr:
            continue
        m = frozen[fam]
        prompts = [h["prompt"] for h in fr]
        t0 = time.perf_counter()
        preds, conf = m.predict_with_conf(np.array(prompts, dtype=object))
        s1_ms = (time.perf_counter() - t0) * 1000 / len(fr)
        for h, p, c in zip(fr, preds, conf):
            pr = pilot.get(h["id"])
            if pr is None:
                continue
            s1_ok = str(p).strip().upper() == str(h["gold_label"]).strip().upper()
            s2_ok = bool(pr.get("correct")) and not pr.get("error")
            rows.append({"id": h["id"], "family": fam, "prompt": h["prompt"], "gold": h["gold_label"],
                         "s1_label": str(p), "s1_conf": float(c), "s1_correct": s1_ok,
                         "s2_correct": s2_ok, "s2_error": bool(pr.get("error")),
                         "s2_cost": float(pr.get("cost_usd") or 0.0),
                         "s2_latency": float(pr.get("latency_ms") or 0.0),
                         "s1_latency_ms": s1_ms, "d": d_hat(h), "hops": h.get("hops"),
                         "ambiguity": h.get("ambiguity"), "knowledge": h.get("knowledge")})
    return rows


def choose_global(train_rows):
    best = None
    for t in TAU_GRID:
        correct = sum((r["s1_correct"] if r["s1_conf"] >= t else r["s2_correct"]) for r in train_rows)
        s1n = sum(1 for r in train_rows if r["s1_conf"] >= t)
        key = (correct, s1n)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def choose_per_family(train_rows):
    out = {}
    for fam in FAMS:
        fr = [r for r in train_rows if r["family"] == fam]
        if not fr:
            out[fam] = 1.01
            continue
        best = None
        for t in TAU_GRID:
            correct = sum((r["s1_correct"] if r["s1_conf"] >= t else r["s2_correct"]) for r in fr)
            s1n = sum(1 for r in fr if r["s1_conf"] >= t)
            key = (correct, s1n)
            if best is None or key > best[0]:
                best = (key, t)
        out[fam] = best[1]
    return out


def route_outcome(route, r):
    """Returns (correct, used_s2, cost, latency) for a routed row."""
    if route == "S1":
        return r["s1_correct"], False, 0.0, r["s1_latency_ms"]
    return r["s2_correct"], True, r["s2_cost"], r["s2_latency"]


def evaluate(rows, router_fn):
    """Pooled outcome over rows, routes from router_fn(r)->'S1'|'S2'."""
    out = []
    for r in rows:
        route = router_fn(r)
        ok, s2, cost, lat = route_outcome(route, r)
        out.append({"id": r["id"], "family": r["family"], "route": route, "correct": bool(ok),
                    "s2": s2, "cost": cost, "latency": lat,
                    "s1_correct": r["s1_correct"], "s2_correct": r["s2_correct"]})
    return out


def nested(policy_name, rows, choose):
    """Outer 5-fold grouped CV on the held-out rows; thresholds chosen on outer-train only."""
    groups = np.array([CP.group_key(r["prompt"]) for r in rows])
    X = np.arange(len(rows))
    gkf = GroupKFold(n_splits=5)
    routed = {}
    for tr, te in gkf.split(X, X, groups):
        train_rows = [rows[i] for i in tr]
        test_rows = [rows[i] for i in te]
        fn = choose(train_rows)
        for r in test_rows:
            routed[r["id"]] = fn(r)
    return routed


def run():
    frozen, sel = S.run_selection()
    rows = build_eval(frozen)
    # policy outputs
    routes = {}
    routes["R0 always-S2"] = {r["id"]: "S2" for r in rows}
    routes["R1 always-S1"] = {r["id"]: "S1" for r in rows}
    routes["ORC oracle (not a policy)"] = {r["id"]: ("S1" if r["s1_correct"] else "S2") for r in rows}

    def make_global(train_rows):
        t = choose_global(train_rows)
        return lambda r: "S1" if r["s1_conf"] >= t else "S2"

    def make_per_family(train_rows):
        tf = choose_per_family(train_rows)
        return lambda r: "S1" if r["s1_conf"] >= tf[r["family"]] else "S2"

    def make_per_family_d(train_rows):
        tf = choose_per_family(train_rows)
        return lambda r: "S1" if (r["s1_conf"] >= tf[r["family"]] and r["d"] <= TAU_D) else "S2"

    routes["R5 global conf threshold (nested)"] = nested("R5", rows, make_global)
    routes["H3 HARP-v3 per-family conf (nested)"] = nested("H3", rows, make_per_family)
    routes["H3d HARP-v3 + difficulty gate (nested)"] = nested("H3d", rows, make_per_family_d)
    routes["H1 legacy HARP (d<=0.45 & conf>=0.65)"] = {
        r["id"]: ("S1" if (r["d"] <= TAU_D and r["s1_conf"] >= TAU_LEGACY) else "S2") for r in rows}

    results = {}
    per_row_out = {}
    for name, rm in routes.items():
        ev = evaluate(rows, lambda r, rm=rm: rm[r["id"]])
        per_row_out[name] = ev
        n = len(ev)
        acc = sum(e["correct"] for e in ev) / n
        s2share = sum(e["s2"] for e in ev) / n
        cost_1k = sum(e["cost"] for e in ev) / n * 1000
        lat = sorted(e["latency"] for e in ev)
        s1_routes = [e for e in ev if e["route"] == "S1"]
        missed = sum(1 for e in s1_routes if not e["s1_correct"]) / n
        unnec = sum(1 for e in ev if e["route"] == "S2" and e["s1_correct"]) / n
        fam_acc = {}
        for fam in FAMS:
            fe = [e for e in ev if e["family"] == fam]
            if fe:
                fam_acc[fam] = round(sum(e["correct"] for e in fe) / len(fe), 3)
        boot = bootstrap(ev, n_boot=2000)
        results[name] = {"n": n, "accuracy": round(acc, 4), "accuracy_ci95": boot,
                         "s2_share": round(s2share, 4), "s1_share": round(1 - s2share, 4),
                         "cost_per_1k_usd": round(cost_1k, 5),
                         "latency_p50_ms": round(lat[len(lat) // 2], 1),
                         "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
                         "missed_escalation_rate": round(missed, 4),
                         "unnecessary_escalation_rate": round(unnec, 4),
                         "per_family_accuracy": fam_acc}
    # S1 standalone quality on the held-out set
    s1_acc = sum(r["s1_correct"] for r in rows) / len(rows)
    s2_acc = sum(r["s2_correct"] for r in rows) / len(rows)
    results["_s1_standalone"] = {"accuracy": round(s1_acc, 4), "n": len(rows)}
    results["_s2_standalone"] = {"accuracy": round(s2_acc, 4), "n": len(rows)}
    results["_s1_selection"] = {f: sel[f]["selected"] for f in FAMS}
    ece = calibration(rows)
    results["_calibration"] = ece

    json.dump(results, open(OUT_JSON, "w"), indent=1)
    with open(OUT_ROWS, "w") as f:
        for name, ev in per_row_out.items():
            for e in ev:
                e2 = dict(e); e2["policy"] = name
                f.write(json.dumps(e2) + "\n")
    write_md(results)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                       if kk in ("accuracy", "s2_share", "accuracy_ci95")}) for k, v in results.items()
                      if not k.startswith("_")}, indent=1))
    return results


def bootstrap(ev, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(ev)
    vals = [1 if e["correct"] else 0 for e in ev]
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(vals[i] for i in idx) / n)
    stats.sort()
    return [round(stats[int(0.025 * n_boot)], 4), round(stats[int(0.975 * n_boot)], 4)]


def calibration(rows, bins=10):
    conf = np.array([r["s1_conf"] for r in rows])
    cor = np.array([1.0 if r["s1_correct"] else 0.0 for r in rows])
    ece = 0.0
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & (conf < hi if b < bins - 1 else conf <= hi)
        if m.sum():
            ece += m.sum() / len(rows) * abs(conf[m].mean() - cor[m].mean())
            out.append([round(float(conf[m].mean()), 3), round(float(cor[m].mean()), 3), int(m.sum())])
    return {"ece_10bin": round(float(ece), 4), "bins_conf_acc_n": out,
            "brier": round(float(np.mean((conf - cor) ** 2)), 4)}


def write_md(res):
    L = ["# Routing on the held-out 243 items (v3, nested CV, real cheap-LLM S2 answers)", "",
         f"S1 (dev-trained, frozen) standalone accuracy: {res['_s1_standalone']['accuracy']:.3f}; "
         f"S2 standalone accuracy: {res['_s2_standalone']['accuracy']:.3f} (S2 errors count as wrong).",
         f"S1 selection: {res['_s1_selection']}", "",
         f"Calibration (held-out, calibrated conf): ECE(10 bins) {res['_calibration']['ece_10bin']}, "
         f"Brier {res['_calibration']['brier']}.", "",
         "| policy | accuracy [95% CI] | S2 share | cost /1k (USD) | p50/p95 latency ms | missed escalation | unnecessary escalation |",
         "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v['accuracy']:.3f} [{v['accuracy_ci95'][0]:.3f}, {v['accuracy_ci95'][1]:.3f}] | "
                 f"{v['s2_share']:.3f} | {v['cost_per_1k_usd']:.4f} | {v['latency_p50_ms']:.0f}/{v['latency_p95_ms']:.0f} | "
                 f"{v['missed_escalation_rate']:.3f} | {v['unnecessary_escalation_rate']:.3f} |")
    L += ["", "Per-family accuracy by policy:", ""]
    for k, v in res.items():
        if k.startswith("_"):
            continue
        L.append(f"- {k}: {v['per_family_accuracy']}")
    open(OUT_MD, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
