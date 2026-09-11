"""Reply-quality judge. Synthetic demos are not benchmark results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

from hiver.paths import ROOT

load_dotenv(ROOT / '.env')
RUBRIC_PATH = ROOT / 'docs/judge_rubric.md'


class Judgment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    relevance: int = Field(ge=0, le=2)
    grounding: int = Field(ge=0, le=2)
    usefulness: int = Field(ge=0, le=2)
    safety: int = Field(ge=0, le=2)
    critical_error: bool
    explanation: str = Field(min_length=1)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        temporary = Path(file.name)
    temporary.replace(path)


def demo_examples():
    # Entirely synthetic: no scores or real benchmark labels are included.
    common = {
        'customer_message': 'I was charged twice for my subscription.',
        'prior_context': [],
        'historical_cases': [{
            'case_id': 'synthetic-billing-1',
            'customer_text': 'There are two subscription charges on my statement.',
            'historical_replies': [
                {'text': 'Please contact official support for an account-specific billing review.'}
            ],
        }],
    }
    return [
        {'sample_id': 'demo-01', **common,
         'reply': "I'm sorry about the unexpected charges. Please contact official Spotify support so your billing can be reviewed.",
         'action': 'escalate', 'evidence_ids': ['synthetic-billing-1']},
        {'sample_id': 'demo-02', **common,
         'reply': 'I checked your account and refunded the extra charge. It will reach your bank tomorrow.',
         'action': 'auto_handle', 'evidence_ids': ['synthetic-billing-1']},
    ]


class ReplyJudge:
    def __init__(self):
        self.model = os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')
        self.rubric = RUBRIC_PATH.read_text(encoding='utf-8')
        self.client = None

    def close(self):
        if self.client is not None:
            self.client.close()

    def score(self, example):
        # Do not expose generator identity, predicted intent, gold labels,
        # expected scores, or model reasoning to the judge.
        payload = {key: example[key] for key in (
            'customer_message', 'prior_context', 'historical_cases',
            'reply', 'action', 'evidence_ids',
        )}
        request = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.rubric},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
            ],
            'temperature': 0,
            'max_completion_tokens': 2048,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'reply_judgment',
                    'strict': True,
                    'schema': Judgment.model_json_schema(),
                },
            },
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = ROOT / 'data/api_cache/judge' / f'{key}.json'
        if path.exists():
            value = Judgment.model_validate_json(path.read_text(encoding='utf-8'))
            return value.model_dump(), True

        if self.client is None:
            api_key = os.getenv('GROQ_API_KEY', '').strip()
            if not api_key:
                raise ValueError('Add GROQ_API_KEY to .env.')
            self.client = Groq(api_key=api_key, timeout=60, max_retries=0)

        response = self.client.chat.completions.create(**request)
        if not response.choices or response.choices[0].finish_reason != 'stop':
            raise RuntimeError('Judge response was incomplete; nothing was cached.')
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError('Judge returned no JSON text.')
        value = Judgment.model_validate_json(text)
        if value.critical_error and value.safety != 0:
            raise ValueError('Judge gave a critical error but not safety=0; review rubric adherence.')
        atomic_json(path, value.model_dump())
        return value.model_dump(), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', required=True,
                        help='Judge two synthetic examples, not real evaluation data.')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Create the blank human review file without any API calls.')
    args = parser.parse_args()
    examples = demo_examples()
    folder = ROOT / 'results/judge_demo'
    human_file = folder / 'human_review.json'
    if not human_file.exists():
        atomic_json(human_file, {
            'dataset_type': 'synthetic_demo',
            'instructions': 'Read docs/judge_rubric.md. Fill your own scores before opening judge results. Do not use an AI to supply your ratings.',
            'reviewer': '',
            'examples': [dict(example, human_scores={
                'relevance': None, 'grounding': None, 'usefulness': None,
                'safety': None, 'critical_error': None, 'explanation': '',
            }) for example in examples],
        })
    print(f'Human review file: {human_file}')
    if args.prepare_only:
        print('Prepared only. No API calls made.')
        return

    judge = ReplyJudge()
    rows = []
    rubric_hash = hashlib.sha256(judge.rubric.encode()).hexdigest()
    run_key = hashlib.sha256(json.dumps([judge.model, rubric_hash, examples], sort_keys=True).encode()).hexdigest()[:16]
    output = folder / f'judge_results_{run_key}.json'
    try:
        for example in examples:
            scores, cached = judge.score(example)
            rows.append({'sample_id': example['sample_id'], 'scores': scores})
            atomic_json(output, {
                'dataset_type': 'synthetic_demo', 'model': judge.model,
                'rubric_sha256': rubric_hash, 'completed': len(rows),
                'total': len(examples), 'complete': len(rows) == len(examples),
                'limitation': 'Two synthetic cases only. Not real reply quality or human agreement evidence.',
                'judgments': rows,
            })
            print(f"{example['sample_id']}: scored ({'cached' if cached else 'API'}).")
        print('Demo completed. Judge scores are saved separately so you can rate independently.')
        print(f'Judge results (open after your ratings): {output}')
    except Exception as error:
        # Avoid printing response bodies which may contain sensitive input.
        status = getattr(error, 'status_code', None)
        print(f'Stopped: {type(error).__name__}; HTTP status={status}. Completed judgments are saved.')
        raise SystemExit(1)
    finally:
        judge.close()


if __name__ == '__main__':
    main()
