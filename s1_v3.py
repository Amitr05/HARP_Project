"""System-1 candidates and selection (v3). Dev-only: grouped 5-fold CV on T (clean_protocol.build()).

Families: intent, sentiment, nli, multihop -> multiclass per family (labels from the family's data).
          factoid, math (MCQ)                -> option scorer: binary (stem || option) classifier;
                                               prediction = argmax option score.
Confidence: each candidate yields a raw score (top class probability or margin / top decision value);
a Platt calibrator (logistic regression on that score) maps it to P(correct), fitted on OUT-OF-FOLD dev
predictions only. Held-out items never touch selection or calibration.

Candidates (kept small on purpose; sample sizes are small):
  NB-count   : MultinomialNB on word + char_wb 3-5 counts (the v2 stand-in family, re-fit on clean dev)
  LR-tfidf   : LogisticRegression on word 1-2 + char_wb 2-5 TF-IDF, C in {3, 30}, class_weight None/balanced
  SVC-tfidf  : LinearSVC on the same features, C in {0.3, 1}, class_weight None/balanced
Selection criterion: out-of-fold macro-F1 (accuracy as tie-break), per family.
"""
import json
import os
import re

import numpy as np
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401  (kept for reference)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

import clean_protocol as CP

HERE = os.path.dirname(os.path.abspath(__file__))
SEL_JSON = os.path.join(HERE, "logs", "s1_selection.json")
SEL_MD = os.path.join(HERE, "logs", "s1_selection.md")
MCQ_OPT = re.compile(r"(?:^|\s)([A-D])\)\s*", re.S)


def parse_options(prompt):
    ms = list(MCQ_OPT.finditer(prompt))
    out = []
    for j, m in enumerate(ms):
        end = ms[j + 1].start() if j + 1 < len(ms) else len(prompt)
        out.append((m.group(1), prompt[m.end():end].strip()))
    return out


def stem_of(prompt):
    ms = list(MCQ_OPT.finditer(prompt))
    return prompt[:ms[0].start()].strip() if ms else prompt


def _feat_tfidf():
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2))


def _feat_count():
    return make_union(
        CountVectorizer(ngram_range=(1, 1)),
        CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2))


CANDIDATES = {
    "NB-count": lambda: make_pipeline(_feat_count(), MultinomialNB(alpha=0.5)),
    "LR-tfidf-C3": lambda: make_pipeline(_feat_tfidf(), LogisticRegression(C=3, max_iter=3000)),
    "LR-tfidf-C30": lambda: make_pipeline(_feat_tfidf(), LogisticRegression(C=30, max_iter=3000)),
    "LR-tfidf-C3-bal": lambda: make_pipeline(_feat_tfidf(), LogisticRegression(
        C=3, max_iter=3000, class_weight="balanced")),
    "SVC-tfidf-C0.3": lambda: make_pipeline(_feat_tfidf(), LinearSVC(C=0.3)),
    "SVC-tfidf-C1-bal": lambda: make_pipeline(_feat_tfidf(), LinearSVC(C=1.0, class_weight="balanced")),
}


def _scores(model, X):
    """Return (labels, score matrix). Probabilities for LR/NB, decision values for SVC."""
    clf = model[-1]
    if hasattr(clf, "predict_proba"):
        return clf.classes_, model.predict_proba(X)
    return clf.classes_, model.decision_function(X)


def _top_conf(S):
    """Raw confidence per row: top score (prob or decision); for multiclass SVC decision the top value."""
    if S.ndim == 1:
        S = np.vstack([-S, S]).T
    return S.max(axis=1), np.sort(S, axis=1)


def _raw_feature(S):
    """Feature for the calibrator: top value and margin to second (both are monotone-informative)."""
    if S.ndim == 1:
        S = np.vstack([-S, S]).T
    srt = np.sort(S, axis=1)
    top = srt[:, -1]
    second = srt[:, -2] if S.shape[1] > 1 else srt[:, -1] * 0
    return np.column_stack([top, top - second])


# ---------------- multiclass families ----------------
class MulticlassFamily:
    def __init__(self, name, cand_name):
        self.name, self.cand_name = name, cand_name
        self.model = CANDIDATES[cand_name]()
        self.cal = None

    def fit(self, texts, labels):
        self.model.fit(texts, labels)
        return self

    def predict_with_conf(self, texts):
        labels, S = _scores(self.model, texts)
        idx = np.argmax(S if S.ndim == 2 else np.vstack([-S, S]).T, axis=1)
        pred = np.array(labels)[idx]
        raw = _raw_feature(S)
        conf = self.cal.predict_proba(raw)[:, 1] if self.cal is not None else raw[:, 0]
        return pred, conf

    def fit_calibrator(self, raw, correct):
        self.cal = LogisticRegression(C=1.0, max_iter=1000).fit(raw, correct.astype(int))


# ---------------- MCQ families (option scorer) ----------------
class MCQFamily:
    def __init__(self, name, cand_name):
        self.name, self.cand_name = name, cand_name
        base = CANDIDATES[cand_name]()
        self.model = base
        self.cal = None

    @staticmethod
    def pairs(prompts):
        texts = []
        owners = []
        for i, p in enumerate(prompts):
            st = stem_of(p)
            for letter, opt in parse_options(p):
                texts.append(st + " || " + opt)
                owners.append((i, letter))
        return texts, owners

    def fit(self, prompts, golds):
        texts, owners = self.pairs(prompts)
        y = []
        for (i, letter) in owners:
            y.append(1 if letter == str(golds[i]).strip().upper() else 0)
        self.model.fit(texts, np.array(y))
        return self

    def option_scores(self, prompts):
        texts, owners = self.pairs(prompts)
        clf = self.model[-1]
        if hasattr(clf, "predict_proba"):
            s = self.model.predict_proba(texts)[:, 1]
            s = np.log(np.clip(s, 1e-9, 1)) - np.log(np.clip(1 - s, 1e-9, 1))
        else:
            s = self.model.decision_function(texts)
        per = [dict() for _ in prompts]
        for (i, letter), v in zip(owners, s):
            per[i][letter] = float(v)
        return per

    def predict_with_conf(self, prompts):
        per = self.option_scores(prompts)
        preds, raws = [], []
        for d in per:
            if not d:
                preds.append("A")
                raws.append([0.0, 0.0])
                continue
            srt = sorted(d.items(), key=lambda kv: -kv[1])
            top = srt[0][1]
            sec = srt[1][1] if len(srt) > 1 else top - 5.0
            preds.append(srt[0][0])
            raws.append([top, top - sec])
        raws = np.array(raws, dtype=float)
        conf = self.cal.predict_proba(raws)[:, 1] if self.cal is not None else raws[:, 1]
        return np.array(preds), conf

    def fit_calibrator(self, raw, correct):
        self.cal = LogisticRegression(C=1.0, max_iter=1000).fit(raw, correct.astype(int))


# ---------------- selection ----------------
def _dev_rows(family, held_out=False):
    held, dev, pool, _ = CP.build()
    rows = [d for d in dev if d["family"] == family] + [p for p in pool if p["family"] == family]
    return rows


def select_family(family, dev_rows, n_folds=5):
    """Grouped CV on dev rows. Returns (best_cand, table, oof_raw, oof_correct per candidate)."""
    groups = np.array([CP.group_key(r["prompt"], r["family"]) for r in dev_rows])
    gkf = GroupKFold(n_splits=n_folds)
    table = {}
    best = None
    prompts = np.array([r["prompt"] for r in dev_rows], dtype=object)
    golds = np.array([str(r["gold_label"]) for r in dev_rows], dtype=object)
    is_mcq = family in CP.MCQ
    cands = [c for c in CANDIDATES if (c.startswith(("NB", "LR", "SVC")))]
    for cand in cands:
        y_true, y_pred = [], []
        raws, corrects = [], []
        for tr, te in gkf.split(prompts, golds, groups):
            if is_mcq:
                m = MCQFamily(family, cand).fit(prompts[tr], golds[tr])
                pred, _ = m.predict_with_conf(prompts[te])
                per = m.option_scores(prompts[te])
                raw_te = []
                for d in per:
                    srt = sorted(d.items(), key=lambda kv: -kv[1]) if d else [("A", 0.0), ("B", 0.0)]
                    top = srt[0][1]
                    sec = srt[1][1] if len(srt) > 1 else top - 5
                    raw_te.append([top, top - sec])
                raw_te = np.array(raw_te, dtype=float)
            else:
                m = MulticlassFamily(family, cand).fit(prompts[tr], golds[tr])
                pred, _ = m.predict_with_conf(prompts[te])
                _, S = _scores(m.model, prompts[te])
                raw_te = _raw_feature(S)
            y_true.extend(golds[te]); y_pred.extend(pred)
            raws.append(raw_te)
            corrects.append(np.array([str(p).upper() == str(g).upper() for p, g in zip(pred, golds[te])]))
        raws_all = np.vstack(raws)
        correct_all = np.concatenate(corrects)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        table[cand] = {"accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
                       "n": len(y_true)}
        key = (round(mf1, 4), round(acc, 4))
        if best is None or key > best[0]:
            best = (key, cand, raws_all, correct_all)
    return best[1], table, best[2], best[3]


def fit_final(family, cand, dev_rows, raws_oof, correct_oof):
    prompts = np.array([r["prompt"] for r in dev_rows], dtype=object)
    golds = np.array([str(r["gold_label"]) for r in dev_rows], dtype=object)
    if family in CP.MCQ:
        m = MCQFamily(family, cand).fit(prompts, golds)
    else:
        m = MulticlassFamily(family, cand).fit(prompts, golds)
    m.fit_calibrator(raws_oof, correct_oof)
    return m


def run_selection():
    results = {}
    md = ["# System-1 model selection (dev only, grouped 5-fold CV)", "",
          "Selection criterion: out-of-fold macro-F1 (accuracy tie-break). Held-out test not used.", ""]
    _, dev, pool, audit = CP.build()
    frozen = {}
    for fam in ["intent", "sentiment", "nli", "multihop", "factoid", "math"]:
        rows = [d for d in dev if d["family"] == fam] + [p for p in pool if p["family"] == fam]
        if fam in CP.MCQ:
            rows = [d for d in dev if d["family"] == fam]   # pool has no MCQ rows
        cand, table, raws, corr = select_family(fam, rows)
        m = fit_final(fam, cand, rows, raws, corr)
        frozen[fam] = m
        results[fam] = {"selected": cand, "candidates": table, "dev_rows": len(rows),
                        "oof_calibrator_fit_on_rows": int(len(corr))}
        md.append(f"## {fam} (dev rows {len(rows)}; selected: **{cand}**)")
        md.append("")
        md.append("| candidate | OOF accuracy | OOF macro-F1 |")
        md.append("|---|---|---|")
        for c, v in table.items():
            md.append(f"| {c} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
        md.append("")
        print(fam, cand, table[cand], flush=True)
    results["_audit"] = audit
    json.dump(results, open(SEL_JSON, "w"), indent=1)
    open(SEL_MD, "w").write("\n".join(md) + "\n")
    return frozen, results


if __name__ == "__main__":
    run_selection()
