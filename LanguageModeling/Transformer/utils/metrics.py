from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
import six
import math
import oneflow as flow
import oneflow.typing as tp
from six.moves import xrange
from typing import Tuple


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with flow.scope.namespace("pad_to_same_length"):
        x_length = x.shape[1]
        y_length = y.shape[1]

        # max_length = flow.math.maximum(x_length, y_length)
        max_length = max(x_length, y_length)

        x = flow.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])  # input dimension is 3D
        y = flow.pad(y, [[0, 0], [0, max_length - y_length]])  # target dimension is 2D
        return x, y


# # """
# # Test pad_tensor function
# # x = np.ones(shape=(64, 38, 1000), dtype=np.float32)
# # y = np.ones(shape=(64, 36), dtype=np.float32)
# #
# #
# # @flow.global_function()
# # def testpad(x: tp.Numpy.Placeholder(shape=(64, 38, 1000), dtype=flow.float32),
# #             y: tp.Numpy.Placeholder(shape=(64, 36), dtype=flow.float32)) -> Tuple[tp.Numpy, tp.Numpy]:
# #     with flow.scope.namespace("transformer"):
# #         x_1, y_1 = _pad_tensors_to_same_length(x, y)
# #     return x_1, y_1
# #
# #
# # x_1, y_1 = testpad(x, y)
# # print(x_1.shape)
# # print(y_1.shape)
# # """


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns a float32 tensor with shape
      [batch_size, max(length_logits, length_labels)]
  """
    with flow.scope.namespace("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with flow.scope.namespace("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            # low_confince = (1.0 - confidence) / flow.cast(vocab_size-1 ,dtype=flow.float32)
            low_confidence = (1.0 - confidence) / float(vocab_size - 1)

            soft_targets = flow.one_hot(
                flow.cast(labels, flow.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence,
                dtype=flow.float32
            )
            xentropy = flow.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)

            normalizing_constant = -(
                    confidence * math.log(confidence) + float(vocab_size - 1) *
                    low_confidence * math.log(low_confidence + 1e-20))

            xentropy -= normalizing_constant

        weights = flow.cast(flow.math.not_equal(labels,
                                                flow.constant(value=0,
                                                              dtype=flow.float32,
                                                              shape=labels.shape)),
                            dtype=flow.float32)
        return xentropy * weights, weights


# # # Test padded_cross_entropy_loss
# x = np.ones(shape=(64, 38, 1000), dtype=np.float32)
# y = np.ones(shape=(64, 36), dtype=np.float32)
#
#
# @flow.global_function()
# def testxcross(x: tp.Numpy.Placeholder(shape=(64, 38, 1000), dtype=flow.float32),
#                y: tp.Numpy.Placeholder(shape=(64, 36), dtype=flow.float32)) -> Tuple[tp.Numpy, tp.Numpy]:
#     with flow.scope.namespace("transformer"):
#         loss, _ = padded_cross_entropy_loss(x, y, 0.1, vocab_size=1000)
#     return loss, _
#
#
# loss, _ = testxcross(x, y)
# # print(x_1.shape)
# # print(y_1.shape)
# print(loss)


# def _convert_to_eval_metric(metric_fn):
#     """Wrap a metric fn that returns scores and weights as an eval metric fn.
#
#         The input metric_fn returns values for the current batch. The wrapper
#         aggregates the return values collected over all of the batches evaluated.
#
#         Args:
#         metric_fn: function that returns scores and weights for the current batch's
#           logits and predicted labels.
#
#         Returns:
#         function that aggregates the scores and weights from metric_fn.
#     """
#
#     def problem_metric_fn(*args):
#         """Returns an aggregation of the metric_fn's returned values."""
#         (scores, weights) = metric_fn(*args)
#
#         # The tf.metrics.mean function assures correct aggregation.
#         return tf.metrics.mean(scores, weights)
#
#     return problem_metric_fn

def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        # Add reference length
        reference_length += len(references)
        # Add translation length
        translation_length += len(translations)
        # Get n-gram counts, In BLEU, the max_order is 4
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        # Compute the overlap of translation and reference
        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]
    precisions = [0] * max_order
    smooth = 1.0

    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        # If use_bp is True, we add a penalty factor BP
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)
