from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import numpy as np
import sklearn as sk


# TODO: merge this with cluster_utils

def vec_lengths(x):
    return np.sqrt((x * x).sum(axis=1))


def max_lens(x):
    return np.max(x, axis=1)


def vec_seq_similarity(vs, search_vec):
    return cos_compare(vs, [search_vec])


def cos_compare(x, y):
    return sk.metrics.pairwise.cosine_similarity(x, y)


__private_value = 5
