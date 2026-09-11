import json
import unittest
from hiver.paths import ROOT
from hiver.baselines import Baselines, load_development
from hiver.evaluate_baselines import calculate_metrics
from hiver.retrieval import Retriever, get_context

class OfflineTests(unittest.TestCase):
    def test_data_and_splits(self):
        examples = []
        for path in (ROOT / "data/prepared").glob("*_batch_*.json"):
            examples.extend(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(len(examples), 200)
        self.assertEqual(len({x["conversation_id"] for x in examples}), 200)
        self.assertEqual(len(load_development()), 50)

    def test_context_excludes_future_and_stops_on_cycles(self):
        tweets = {
            "1": {"tweet_id": "1", "parent_id": None, "inbound": True, "text": "first"},
            "2": {"tweet_id": "2", "parent_id": 1, "inbound": False, "text": "earlier reply"},
            "3": {"tweet_id": "3", "parent_id": 2, "inbound": True, "text": "current"},
            "4": {"tweet_id": "4", "parent_id": 3, "inbound": False, "text": "future answer"},
        }
        self.assertEqual([x["text"] for x in get_context(tweets["3"], tweets)], ["first", "earlier reply"])
        tweets["1"]["parent_id"] = 3
        self.assertEqual(len(get_context(tweets["3"], tweets)), 2)

    def test_baseline_rejects_test_labels(self):
        example = dict(load_development()[0], split="test")
        with self.assertRaises(ValueError):
            Baselines([example])

    def test_metrics_denominator(self):
        rows = [{"expected_intent": "billing_payments", "expected_action": "escalate", "prediction": {"intent": "billing_payments", "action": "escalate"}}]
        metrics = calculate_metrics(rows)
        self.assertIsNone(metrics["unsafe_auto_handle_rate"])
        self.assertEqual(metrics["auto_handle_coverage"], 0)

    def test_retrieval_without_api(self):
        retriever = Retriever()
        self.assertEqual(len(retriever.cases), 4469)
        matches = retriever.search("I was charged twice for my subscription")
        self.assertEqual(len(matches), 3)
        self.assertEqual(len({x["conversation_id"] for x in matches}), 3)
        self.assertEqual(matches[0]["case_id"], "2309229:2309230")
        self.assertEqual(retriever.search(""), [])

if __name__ == "__main__":
    unittest.main()
