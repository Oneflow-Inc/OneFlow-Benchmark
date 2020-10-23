import tempfile
import unittest

import oneflow_transformer.compute_bleu as compute_bleu


class ComputeBleuTest(unittest.TestCase):

    def _create_temp_file(self, text):
        w = tempfile.NamedTemporaryFile(delete=False, mode="w")
        w.write(text)
        w.close()
        return w.name

    def test_bleu_same(self):
        ref = self._create_temp_file("test 1 two 3 \nmore tests!")
        hyp = self._create_temp_file("test 1 two 3 \nmore tests!")

        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertEqual(100, uncased_score)
        self.assertEqual(100, cased_score)

    def test_bleu_same_different_case(self):
        ref = self._create_temp_file("Test 1 two 3\nmore tests!")
        hyp = self._create_temp_file("test 1 two 3\nMore tests!")
        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertEqual(100, uncased_score)
        self.assertLess(cased_score, 100)

    def test_bleu_different(self):
        ref = self._create_temp_file("Testing\nmore tests!")
        hyp = self._create_temp_file("Dog\nCat")
        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertLess(uncased_score, 100)
        self.assertLess(cased_score, 100)

    def test_bleu_tokenize(self):
        s = "Test0, 1 two, 3"
        tokenized = compute_bleu.bleu_tokenize(s)
        self.assertEqual(["Test0", ",", "1", "two", ",", "3"], tokenized)


if __name__ == "__main__":
    unittest.main()
