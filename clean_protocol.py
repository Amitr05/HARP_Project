"""Leakage-free evaluation protocol (v3).

Why this exists: the v2 stand-in was trained on the train split PLUS data/train_pool.jsonl. An audit
found 51 held-out test prompts whose normalised text also appears in the pool, and SNLI premises
repeat across items. Those test numbers were therefore inflated. This module defines a grouped split:

  held-out E : the 243 test-split items (the only items with real S2 answers in the pilot log).
  dev-train T: train-split items + pool rows, with every group that touches E REMOVED.

Group key: normalised prompt; for NLI prompts ("Premise: ... Hypothesis: ...") the normalised premise,
so all hypotheses of one premise stay on one side of the split.

Run `python3 clean_protocol.py` to write logs/leak_audit.md and logs/clean_split_manifest.json.
"""
import hashlib
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "data", "bench_v02_items.jsonl")
POOL = os.path.join(HERE, "data", "train_pool.jsonl")
PILOT = os.path.join(HERE, "logs", "pilot_live_log.jsonl")
AUDIT = os.path.join(HERE, "logs", "leak_audit.md")
MANIFEST = os.path.join(HERE, "logs", "clean_split_manifest.json")

SEMANTIC = ("intent", "sentiment", "nli", "multihop")
MCQ = ("factoid", "math")


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


_PREM = re.compile(r"^\s*premise:\s*(.*?)\s*hypothesis:", re.I | re.S)


def group_key(prompt, family=""):
    m = _PREM.match(prompt)
    if m:
        return "prem::" + norm(m.group(1))
    return "txt::" + norm(prompt)


def load_items():
    rows = [json.loads(l) for l in open(ITEMS)]
    return rows


def load_pool():
    return [json.loads(l) for l in open(POOL)]


def build():
    items = load_items()
    pool = load_pool()
    held = [d for d in items if d["split"] == "test"]
    dev_items = [d for d in items if d["split"] == "train"]

    held_groups = {group_key(d["prompt"], d["family"]) for d in held}
    held_norm = {norm(d["prompt"]) for d in held}

    removed_items = [d for d in dev_items if group_key(d["prompt"], d["family"]) in held_groups]
    dev_keep = [d for d in dev_items if group_key(d["prompt"], d["family"]) not in held_groups]

    pool_removed = [p for p in pool if group_key(p["prompt"], p["family"]) in held_groups
                    or norm(p["prompt"]) in held_norm]
    pool_keep = [p for p in pool if p not in pool_removed]

    # within-dev near-duplicate check (grouped CV handles them, this just reports)
    dev_groups = Counter(group_key(d["prompt"], d["family"]) for d in dev_keep + pool_keep)

    audit = {
        "held_out_items": len(held),
        "held_out_groups": len(held_groups),
        "train_split_items": len(dev_items),
        "train_split_items_removed_for_leakage": len(removed_items),
        "pool_rows": len(pool),
        "pool_rows_removed_for_leakage": len(pool_removed),
        "pool_exact_matches_with_held_out": sum(1 for p in pool if norm(p["prompt"]) in held_norm),
        "dev_rows_kept": len(dev_keep) + len(pool_keep),
        "dev_groups": len(dev_groups),
        "dev_groups_with_multiple_rows": sum(1 for g, n in dev_groups.items() if n > 1),
        "held_family_counts": dict(Counter(d["family"] for d in held)),
        "dev_family_counts": dict(Counter(d["family"] for d in dev_keep + pool_keep)),
    }
    return held, dev_keep, pool_keep, audit


def fold_of(key, k=5):
    """Deterministic fold assignment by group key (stable across runs)."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % k


def write_audit(audit):
    lines = ["# Leak audit: clean protocol (v3)", "",
             "The v2 stand-in was trained on the train split plus `train_pool.jsonl`. Groups that touch",
             "the held-out test items were found in the training data. The table gives the counts before",
             "and after removal. Earlier test numbers from v2 are contaminated and are NOT used as",
             "independent results.", ""]
    for k, v in audit.items():
        lines.append(f"- {k}: {v}")
    open(AUDIT, "w").write("\n".join(lines) + "\n")
    json.dump(audit, open(MANIFEST, "w"), indent=1)


if __name__ == "__main__":
    h, d, p, a = build()
    write_audit(a)
    print(json.dumps(a, indent=1))
