import unittest

from retrieval_metrics import hit_at_k, recall_at_k, mrr_at_k


class TestRetrievalMetrics(unittest.TestCase):
    def test_hit_at_k_positive(self):
        retrieved = ["doc1.txt", "doc2.txt", "doc3.txt"]
        relevant = ["doc2.txt"]
        self.assertEqual(hit_at_k(retrieved, relevant), 1.0)

    def test_hit_at_k_negative(self):
        retrieved = ["doc1.txt", "doc3.txt"]
        relevant = ["doc2.txt"]
        self.assertEqual(hit_at_k(retrieved, relevant), 0.0)

    def test_recall_at_k_partial(self):
        retrieved = ["doc1.txt", "doc2.txt"]
        relevant = ["doc2.txt", "doc3.txt"]
        # 1 relevant doc out of 2 appears in retrieved
        self.assertAlmostEqual(recall_at_k(retrieved, relevant), 0.5)

    def test_recall_at_k_full(self):
        retrieved = ["doc2.txt", "doc3.txt"]
        relevant = ["doc2.txt", "doc3.txt"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant), 1.0)

    def test_mrr_at_k_first(self):
        retrieved = ["doc2.txt", "doc1.txt", "doc3.txt"]
        relevant = ["doc2.txt", "doc3.txt"]
        # first relevant at position 1 → 1.0
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant), 1.0)

    def test_mrr_at_k_second(self):
        retrieved = ["doc1.txt", "doc3.txt", "doc2.txt"]
        relevant = ["doc2.txt", "doc3.txt"]
        # first relevant at position 2 → 0.5
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant), 0.5)

    def test_mrr_at_k_none(self):
        retrieved = ["doc1.txt", "doc4.txt"]
        relevant = ["doc2.txt", "doc3.txt"]
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant), 0.0)


if __name__ == "__main__":
    unittest.main()
