"""Judge frozen agent replies and compare with pre-recorded human ratings."""
import argparse
import hashlib
import json
import math
import time
import warnings

from sklearn.metrics import cohen_kappa_score
from hiver.paths import ROOT
from hiver.judge import ReplyJudge, Judgment, atomic_json

SAMPLE = ROOT / "results/agent_review/a3b4be5445390607"
DIMENSIONS = ("relevance", "grounding", "usefulness", "safety")


def load_review(folder):
    review = json.loads((folder / "human_review.json").read_text(encoding="utf-8"))
    candidates = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    rows = review["examples"]
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("Reviewer name is required.")
    ids = [c["sample_id"] for c in candidates]
    human_ids = [c["sample_id"] for c in rows]
    if len(ids) != 20 or len(set(ids)) != 20 or len(human_ids) != 20 or set(ids) != set(human_ids):
        raise ValueError("Need the same 20 distinct sample IDs.")
    by_id = {r["sample_id"]: r for r in rows}
    for candidate in candidates:
        human = by_id[candidate["sample_id"]]
        if any(human.get(key) != value for key, value in candidate.items()):
            raise ValueError("Candidate content changed during review.")
        scores = Judgment.model_validate(human["human_scores"])
        if not scores.explanation.strip():
            raise ValueError("Each rating needs an explanation.")
        if scores.critical_error and scores.safety != 0:
            raise ValueError("A critical error requires safety=0.")
    return review, candidates


def kappa(left, right, *, weighted=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = cohen_kappa_score(left, right, labels=[0, 1, 2] if weighted else [False, True],
                                  weights="linear" if weighted else None)
    return float(value) if math.isfinite(value) else None


def agreement(review, judgments):
    humans = {x["sample_id"]: x["human_scores"] for x in review["examples"]}
    judges = {x["sample_id"]: x["scores"] for x in judgments}
    if len(judges) != len(judgments) or set(humans) != set(judges):
        raise ValueError("Cannot report agreement before all replies are judged.")
    ids = sorted(humans)
    dimensions = {}
    total_matches = 0
    for dim in DIMENSIONS:
        h = [humans[s][dim] for s in ids]
        j = [judges[s][dim] for s in ids]
        matches = sum(a == b for a, b in zip(h, j))
        total_matches += matches
        dimensions[dim] = {"exact_agreement": matches / len(ids),
                           "matches": matches, "examples": len(ids),
                           "linear_weighted_kappa": kappa(h, j, weighted=True),
                           "mean_absolute_difference": sum(abs(a-b) for a,b in zip(h,j))/len(ids)}
    h = [humans[s]["critical_error"] for s in ids]
    j = [judges[s]["critical_error"] for s in ids]
    return {
        "replies": len(ids), "dimension_ratings": 4 * len(ids),
        "exact_dimension_matches": total_matches,
        "exact_dimension_agreement": total_matches / (4 * len(ids)),
        "dimensions": dimensions,
        "critical_error": {"exact_agreement": sum(a==b for a,b in zip(h,j))/len(ids),
                           "human_positive": sum(h), "judge_positive": sum(j),
                           "both_positive": sum(a and b for a,b in zip(h,j)),
                           "kappa": kappa(h,j)},
        "disagreements": [{"sample_id": s, "human_scores": humans[s], "judge_scores": judges[s]}
                          for s in ids if any(humans[s][d] != judges[s][d]
                                             for d in (*DIMENSIONS, "critical_error"))],
        "limitations": [
            "20 successful replies; one generation failure excluded from this review sample.",
            "80 dimension ratings are correlated within 20 replies, not 80 independent examples.",
            "One project reviewer; assistant gave interpretive feedback on review-02 before ratings were frozen.",
            "Reviewer may have seen development/test outputs; not fully independent blind validation.",
            "Agreement is not proof of correctness. Zero human critical flags cannot validate critical-error sensitivity.",
            "Undefined kappa is null; generator and judge may share model-family biases.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=15)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.delay < 0:
        parser.error("limit must be positive and delay nonnegative")
    review, candidates = load_review(SAMPLE)
    if args.check_only:
        print("Validated 20 human ratings and unchanged candidate content. No API calls.")
        return
    judge = ReplyJudge()
    signature = hashlib.sha256(json.dumps({
        "review": review, "candidates": candidates, "model": judge.model,
        "rubric": judge.rubric, "judge_code": (ROOT / "hiver/judge.py").read_text(encoding="utf-8"),
        "comparison_code": (ROOT / "hiver/judge_agent.py").read_text(encoding="utf-8"),
    }, sort_keys=True).encode()).hexdigest()[:16]
    folder = SAMPLE / "judge" / signature
    snapshot = folder / "human_review_snapshot.json"
    if not snapshot.exists():
        atomic_json(snapshot, review)
    output = folder / "judge_results.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "model": judge.model, "sample": SAMPLE.name, "judgments": [], "total": 20,
    }
    completed = {x["sample_id"] for x in report["judgments"]}
    print(f"Judge: {judge.model}; saved {len(completed)}/20", flush=True)
    try:
        processed = 0
        for candidate in candidates:
            if candidate["sample_id"] in completed:
                continue
            if processed >= args.limit:
                break
            if processed:
                time.sleep(args.delay)
            scores, cached = judge.score(candidate)
            report["judgments"].append({"sample_id": candidate["sample_id"], "scores": scores})
            completed.add(candidate["sample_id"])
            processed += 1
            report.update(completed=len(completed), complete=len(completed)==20)
            atomic_json(output, report)
            print(f"Saved {len(completed)}/20 ({'cache' if cached else 'API'})", flush=True)
        if len(completed) == 20:
            result = agreement(review, report["judgments"])
            result["judge_model"] = judge.model
            atomic_json(folder / "agreement.json", result)
            print(json.dumps({k:v for k,v in result.items() if k not in ("disagreements", "limitations")}, indent=2))
        else:
            print("Incomplete; resume the same command. No final agreement reported.")
    except Exception as error:
        # Avoid exposing request headers or API key contents.
        print(f"Stopped: {type(error).__name__}; status={getattr(error, 'status_code', None)}. Saved judgments preserved.", flush=True)
        raise SystemExit(1)
    finally:
        judge.close()
    print(f"Results: {folder}")


if __name__ == "__main__":
    main()
