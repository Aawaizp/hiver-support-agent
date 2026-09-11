"""Blind, paired review of real development baseline replies."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from hiver.paths import ROOT
from hiver.baselines import load_development
from hiver.judge import ReplyJudge, atomic_json
from hiver.retrieval import Retriever

SEED = 42
COUNT = 10


def selected_ids(ids):
    if len(set(ids)) != len(ids) or len(ids) < COUNT:
        raise ValueError('Need at least 10 distinct development messages.')
    return random.Random(SEED).sample(sorted(ids), COUNT)


def prepare():
    examples = load_development()
    inputs = {x['customer_tweet_id']: x for x in examples}
    predictions = {}
    for name in ('simple', 'trivial'):
        path = ROOT / 'results' / f'{name}_dev_predictions.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        lookup = {row['customer_tweet_id']: row for row in rows}
        if len(lookup) != len(rows) or set(lookup) != set(inputs):
            raise ValueError(f'{name} predictions must cover exactly the 50 development messages.')
        for key, row in lookup.items():
            if row['customer_text'] != inputs[key]['customer_text']:
                raise ValueError('Prediction text does not match the development input.')
        predictions[name] = lookup

    chosen = selected_ids(list(inputs))
    source_signature = {
        'seed': SEED,
        'development': examples,
        'predictions': predictions,
        'corpus_sha256': hashlib.sha256((ROOT/'data/prepared/retrieval_conversations_clean.jsonl').read_bytes()).hexdigest(),
        'retrieval_code_sha256': hashlib.sha256((ROOT/'hiver/retrieval.py').read_bytes()).hexdigest(),
        'sampler_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    sample_hash = hashlib.sha256(json.dumps(source_signature, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / 'results/judge_baselines' / sample_hash
    candidates_file = folder / 'candidates.json'
    if not candidates_file.exists():
        retriever = Retriever()
        paired = []
        for key in chosen:
            example = inputs[key]
            # Same reference cases for both systems; no gold labels or rationale.
            evidence = retriever.search(example['customer_text'], example['prior_context'])
            evidence = [{k: v for k, v in case.items() if k != 'similarity'} for case in evidence]
            for name in ('simple', 'trivial'):
                prediction = predictions[name][key]['prediction']
                if prediction.get('status') != 'ok':
                    raise ValueError('Cannot judge an unsuccessful prediction as a reply.')
                paired.append((name, key, {
                    'customer_message': example['customer_text'],
                    'prior_context': example['prior_context'],
                    'historical_cases': evidence,
                    'reply': prediction['reply'],
                    'action': prediction['action'],
                    'evidence_ids': prediction.get('evidence_ids', []),
                }))
        random.Random(SEED).shuffle(paired)
        candidates = []
        mapping = []
        for number, (name, key, candidate) in enumerate(paired, 1):
            sample_id = f'review-{number:02d}'
            candidates.append({'sample_id': sample_id, **candidate})
            mapping.append({'sample_id': sample_id, 'system': name, 'customer_tweet_id': key})
        atomic_json(folder/'source_mapping.json', mapping)
        atomic_json(candidates_file, candidates)
    else:
        candidates = json.loads(candidates_file.read_text(encoding='utf-8'))

    review = folder/'human_review.json'
    if not review.exists():
        atomic_json(review, {
            'dataset_type': 'real_development_baseline_replies',
            'sample_hash': sample_hash,
            'reviewer': '',
            'instructions': 'Read docs/judge_rubric.md. Score each reply yourself before opening judge_results or source_mapping. Fill only reviewer and human_scores. The sample has 20 replies to 10 messages, not 20 independent conversations.',
            'examples': [dict(candidate, human_scores={
                'relevance': None, 'grounding': None, 'usefulness': None,
                'safety': None, 'critical_error': None, 'explanation': '',
            }) for candidate in candidates],
        })
    atomic_json(folder/'sampling.json', {
        'seed': SEED, 'unique_messages': COUNT, 'replies': 2*COUNT,
        'method': 'Uniform random sample of 10 development messages, both leave-one-conversation-out baseline replies, shuffled for review.',
        'limitations': [
            'A small paired development sample, not a held-out test estimate.',
            'Generator names are omitted but reply style may reveal the baseline.',
            'No Gemini replies yet; agreement here does not validate judging Gemini.',
            'AI annotation labels and generator reasoning are not shown to the judge or reviewer.',
        ],
    })
    return folder, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, default=5, help='Maximum new judgments this run.')
    parser.add_argument('--delay', type=float, default=15, help='Seconds between requests to respect quota.')
    args = parser.parse_args()
    if args.limit < 1 or args.delay < 0:
        parser.error('limit must be positive and delay cannot be negative')
    folder, candidates = prepare()
    print(f'Your review file: {folder / "human_review.json"}')
    if args.prepare_only:
        print('Prepared 20 real replies. No API calls made.')
        return
    judge = ReplyJudge()
    request_version = hashlib.sha256(json.dumps({
        'model': judge.model, 'rubric': judge.rubric,
        'judge_code': (ROOT/'hiver/judge.py').read_text(encoding='utf-8'),
    }, sort_keys=True).encode()).hexdigest()[:16]
    path = folder/f'judge_results_{request_version}.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'model': judge.model, 'request_version': request_version,
        'sample_hash': folder.name, 'total': len(candidates),
        'judgments': [],
    }
    completed = {x['sample_id'] for x in report['judgments']}
    processed = 0
    try:
        for candidate in candidates:
            if candidate['sample_id'] in completed:
                continue
            if processed >= args.limit:
                break
            if processed:
                time.sleep(args.delay)
            scores, cached = judge.score(candidate)
            report['judgments'].append({'sample_id': candidate['sample_id'], 'scores': scores})
            completed.add(candidate['sample_id'])
            processed += 1
            report['completed'] = len(completed)
            report['complete'] = len(completed) == len(candidates)
            atomic_json(path, report)
            print(f'Saved {len(completed)}/{len(candidates)} judgments ({"cached" if cached else "API"}).')
    except Exception as error:
        print(f'Stopped: {type(error).__name__}; status={getattr(error, "status_code", None)}. Resume later; successful scores are saved.')
        raise SystemExit(1)
    finally:
        judge.close()
    print(f'Completed: {len(completed)}/{len(candidates)}. Keep judge scores closed until your ratings are done.')


if __name__ == '__main__':
    main()
