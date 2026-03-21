"""
tests/test_chemistry.py
=======================
Sanity checks for 4-color and 2-color chemistry support.

Run with:  python -m pytest tests/test_chemistry.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.deconvolution import (
    encode_seq, pair_sum, k_sum_signal, generate_imaging_matrix,
    collision_stats, deconvolution_counts, deconvolution_counts_dropout,
    all_pairs, ENCODING_2COLOR,
)

# ---------------------------------------------------------------------------
# Synthetic spacer set — no CSV needed, covers all 4 bases at known positions
# ---------------------------------------------------------------------------
# Position:  0  1  2  3  4  5  6  7  8  9
SPACERS = [
    "ACGTACGTAC",   # 0: A C G T A C G T A C
    "TGCATGCATG",   # 1: T G C A T G C A T G
    "GGGGGGGGGG",   # 2: all G
    "CCCCCCCCCC",   # 3: all C
    "AAAAAAAAAAA",  # 4: all A
    "TTTTTTTTTT",   # 5: all T
]


# ---------------------------------------------------------------------------
# 1. encode_seq — 4color
# ---------------------------------------------------------------------------

def test_encode_seq_4color_shape():
    for L in [1, 5, 10]:
        X = encode_seq(SPACERS[0], L, chemistry="4color")
        assert X.shape == (L, 4), f"expected ({L},4), got {X.shape}"


def test_encode_seq_4color_one_hot():
    X = encode_seq(SPACERS[0], 10, chemistry="4color")
    assert np.all(X.sum(axis=1) == 1), "each row must sum to 1"
    assert set(np.unique(X)).issubset({0, 1}), "only 0/1 values"


# ---------------------------------------------------------------------------
# 2. encode_seq — 2color
# ---------------------------------------------------------------------------

def test_encode_seq_2color_shape():
    for L in [1, 5, 10]:
        X = encode_seq(SPACERS[0], L, chemistry="2color")
        assert X.shape == (L, 2), f"expected ({L},2), got {X.shape}"


def test_encode_seq_2color_values():
    """Explicit check of A/C/G/T encoding."""
    seq = "ACGT"
    X = encode_seq(seq, 4, chemistry="2color")
    assert list(X[0]) == [1, 1], f"A should be [1,1], got {list(X[0])}"
    assert list(X[1]) == [0, 1], f"C should be [0,1], got {list(X[1])}"
    assert list(X[2]) == [0, 0], f"G should be [0,0], got {list(X[2])}"
    assert list(X[3]) == [1, 0], f"T should be [1,0], got {list(X[3])}"


def test_encode_seq_2color_binary():
    for spacer in SPACERS:
        X = encode_seq(spacer, 10, chemistry="2color")
        assert set(np.unique(X)).issubset({0, 1}), "only 0/1 values expected"


def test_encode_seq_2color_encoding_dict():
    """ENCODING_2COLOR dict matches what encode_seq produces."""
    for base, expected in ENCODING_2COLOR.items():
        X = encode_seq(base, 1, chemistry="2color")
        assert list(X[0]) == expected, f"base {base}: expected {expected}, got {list(X[0])}"


# ---------------------------------------------------------------------------
# 3. encode_seq — bad chemistry
# ---------------------------------------------------------------------------

def test_encode_seq_bad_chemistry():
    with pytest.raises(ValueError, match="Unknown chemistry"):
        encode_seq(SPACERS[0], 5, chemistry="3color")


# ---------------------------------------------------------------------------
# 4. pair_sum shapes
# ---------------------------------------------------------------------------

def test_pair_sum_shape_4color():
    sig = pair_sum(SPACERS[0], SPACERS[1], L=10, chemistry="4color")
    assert sig.shape == (10, 4)
    assert sig.min() >= 0 and sig.max() <= 2


def test_pair_sum_shape_2color():
    sig = pair_sum(SPACERS[0], SPACERS[1], L=10, chemistry="2color")
    assert sig.shape == (10, 2)
    assert sig.min() >= 0 and sig.max() <= 2


def test_pair_sum_default_is_4color():
    """Calling pair_sum without chemistry should match chemistry='4color'."""
    s1 = pair_sum(SPACERS[0], SPACERS[1], L=10)
    s2 = pair_sum(SPACERS[0], SPACERS[1], L=10, chemistry="4color")
    assert np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# 5. k_sum_signal shapes
# ---------------------------------------------------------------------------

def test_k_sum_signal_shape_4color():
    combo = (0, 1, 2)
    S = k_sum_signal(SPACERS, combo, L=8, chemistry="4color")
    assert S.shape == (8, 4)
    assert np.all(S.sum(axis=1) == 3)  # k=3, all 4-color rows sum to k


def test_k_sum_signal_shape_2color():
    combo = (0, 1)
    S = k_sum_signal(SPACERS, combo, L=8, chemistry="2color")
    assert S.shape == (8, 2)
    assert S.min() >= 0 and S.max() <= 2


# ---------------------------------------------------------------------------
# 6. generate_imaging_matrix shapes
# ---------------------------------------------------------------------------

def test_generate_imaging_matrix_shape_4color():
    matrix, pairs = generate_imaging_matrix(SPACERS, L=6, chemistry="4color")
    n_pairs = len(all_pairs(len(SPACERS)))
    assert matrix.shape == (n_pairs, 6, 4)
    assert len(pairs) == n_pairs


def test_generate_imaging_matrix_shape_2color():
    matrix, pairs = generate_imaging_matrix(SPACERS, L=6, chemistry="2color")
    n_pairs = len(all_pairs(len(SPACERS)))
    assert matrix.shape == (n_pairs, 6, 2)


# ---------------------------------------------------------------------------
# 7. 2-color is at least as ambiguous as 4-color (fewer channels → less info)
# ---------------------------------------------------------------------------

def test_2color_ambiguity_geq_4color():
    """2-color always has ≥ ambiguous_fraction than 4-color at same L."""
    for L in [4, 6, 8, 10]:
        af4 = collision_stats(SPACERS, L, chemistry="4color")["ambiguous_fraction"]
        af2 = collision_stats(SPACERS, L, chemistry="2color")["ambiguous_fraction"]
        assert af2 >= af4, (
            f"L={L}: 2-color ambig ({af2:.3f}) should be >= 4-color ({af4:.3f})"
        )


# ---------------------------------------------------------------------------
# 8. Dropout with p=0 — no failed outcomes
# ---------------------------------------------------------------------------

def test_dropout_zero_p_no_failures_4color():
    res = deconvolution_counts_dropout(SPACERS, L=8, dropout_p=0.0,
                                       n_trials=10, seed=0, chemistry="4color")
    assert res["failed"] == 0, "p=0 dropout should never produce failed outcomes"


def test_dropout_zero_p_no_failures_2color():
    res = deconvolution_counts_dropout(SPACERS, L=8, dropout_p=0.0,
                                       n_trials=10, seed=0, chemistry="2color")
    assert res["failed"] == 0, "p=0 dropout should never produce failed outcomes"


# ---------------------------------------------------------------------------
# 9. deconvolution_counts — correct + ambiguous + failed == total
# ---------------------------------------------------------------------------

def test_deconvolution_counts_totals_4color():
    res = deconvolution_counts(SPACERS, L=8, chemistry="4color")
    assert res["correct"] + res["ambiguous"] + res["failed"] == res["total"]


def test_deconvolution_counts_totals_2color():
    res = deconvolution_counts(SPACERS, L=8, chemistry="2color")
    assert res["correct"] + res["ambiguous"] + res["failed"] == res["total"]
