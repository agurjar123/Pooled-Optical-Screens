"""
deconvolution.py
================
ISS barcode deconvolution for combinatorial CRISPR screens.

Supports 4-color and 2-color sequencing chemistry.  Simulates whether k-plex
gRNA combinations can be deconvolved from summed fluorescence signals.

Assumptions
-----------
- 4-color chemistry: A, C, G, T channels (one-hot encoded, shape (L,4))
- 2-color chemistry (Illumina NovaSeq/NextSeq): A=[1,1], T=[1,0], C=[0,1], G=[0,0]
  (shape (L,2))
- Signals ADD when cells contain multiple gRNAs (equal abundance, 50:50)
- Unordered combinations with replacement (order of gRNAs doesn't matter)
- Noiseless baseline; dropout simulation available separately

Authorship
----------
All code and commits: agurjar123 (GitHub).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, combinations
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASES = "ACGT"
B2I: Dict[str, int] = {b: i for i, b in enumerate(BASES)}

# Illumina 2-channel chemistry: A=[green,red], T=[green,_], C=[_,red], G=[_,_]
ENCODING_2COLOR: Dict[str, List[int]] = {
    "A": [1, 1], "C": [0, 1], "G": [0, 0], "T": [1, 0]
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spacers_from_csv(
    filepath: str,
    spacer_col: str = "Guide with handles",
    n: int = 52,
    handle_len: int = 20,
) -> List[str]:
    """Load gRNA spacer sequences from CSV, stripping the 5' cloning handle.

    The 'Guide with handles' column contains a 20 nt 5' handle followed by the
    20 nt spacer and a 3' scaffold tail.  We slice off the handle and return the
    remaining sequence (uppercase DNA).  Since all analyses use prefix_len ≤ 20,
    only the spacer bases are ever used even if the tail is included.

    Parameters
    ----------
    filepath : str
        Path to CSV containing the gRNA library.
    spacer_col : str
        Column name with the full construct sequence.
    n : int
        Number of guides to load (first n rows).
    handle_len : int
        Length of 5' handle to skip (default 20).

    Returns
    -------
    List[str]
        List of spacer sequences as uppercase DNA strings.
    """
    df = pd.read_csv(filepath)
    spacers = df[spacer_col].str[handle_len:].head(n).str.upper().tolist()
    return spacers


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def onehot_4color(seq: str, L: int) -> np.ndarray:
    """One-hot encode the first L bases of seq into a (L, 4) array.

    Parameters
    ----------
    seq : str
        DNA sequence (A/C/G/T, case-insensitive).
    L : int
        Number of positions to encode.

    Returns
    -------
    np.ndarray
        Shape (L, 4), dtype uint8.  Row p is [1,0,0,0] for A, [0,1,0,0] for C,
        [0,0,1,0] for G, [0,0,0,1] for T.
    """
    x = np.zeros((L, 4), dtype=np.uint8)
    for p, b in enumerate(seq[:L]):
        x[p, B2I[b.upper()]] = 1
    return x


def encode_seq(seq: str, L: int, chemistry: str = "4color") -> np.ndarray:
    """Encode the first L bases of seq according to chemistry.

    Parameters
    ----------
    seq : str
        DNA sequence (A/C/G/T, case-insensitive).
    L : int
        Number of positions to encode.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    np.ndarray
        Shape (L, 4) for '4color', (L, 2) for '2color'.
    """
    if chemistry == "4color":
        return onehot_4color(seq, L)
    if chemistry == "2color":
        x = np.zeros((L, 2), dtype=np.uint8)
        for p, b in enumerate(seq[:L]):
            x[p] = ENCODING_2COLOR[b.upper()]
        return x
    raise ValueError(f"Unknown chemistry: {chemistry!r}. Use '4color' or '2color'.")


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def pair_sum(seq1: str, seq2: str, L: int, chemistry: str = "4color") -> np.ndarray:
    """Summed signal for two spacers at equal abundance.

    Parameters
    ----------
    seq1, seq2 : str
        Spacer sequences.
    L : int
        Prefix length to use.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    np.ndarray
        Shape (L, 4) or (L, 2), dtype int16.  Values in {0, 1, 2}.
    """
    return (encode_seq(seq1, L, chemistry).astype(np.int16)
            + encode_seq(seq2, L, chemistry).astype(np.int16))


def k_sum_signal(
    spacers: List[str], combo: Tuple[int, ...], L: int, chemistry: str = "4color"
) -> np.ndarray:
    """Summed signal for a k-plex combination (indices, with replacement).

    Parameters
    ----------
    spacers : List[str]
    combo : tuple of int
        Sorted tuple of spacer indices (may repeat).
    L : int
        Prefix length.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    np.ndarray
        Shape (L, 4) or (L, 2), values in {0, ..., k}.
    """
    n_ch = 4 if chemistry == "4color" else 2
    s = np.zeros((L, n_ch), dtype=np.int16)
    for idx in combo:
        s += encode_seq(spacers[idx], L, chemistry)
    return s


def generate_imaging_matrix(
    spacers: List[str], L: int, chemistry: str = "4color"
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pre-compute all pair-sum signals as a stacked matrix.

    Parameters
    ----------
    spacers : List[str]
    L : int
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    matrix : np.ndarray
        Shape (n_pairs, L, 4) or (n_pairs, L, 2).
    pairs : list of (i, j)
    """
    pairs = all_pairs(len(spacers))
    matrix = np.stack([pair_sum(spacers[i], spacers[j], L, chemistry) for i, j in pairs])
    return matrix, pairs


# ---------------------------------------------------------------------------
# Hashing / keys
# ---------------------------------------------------------------------------

def signal_key(sig: np.ndarray) -> bytes:
    """Fast hashable key for an ndarray signal (uses raw bytes)."""
    return sig.tobytes()


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------

def all_pairs(n: int) -> List[Tuple[int, int]]:
    """All unordered pairs with replacement for n spacers."""
    return list(combinations_with_replacement(range(n), 2))


# ---------------------------------------------------------------------------
# Collision detection (k = 2)
# ---------------------------------------------------------------------------

def collision_stats(spacers: List[str], L: int, chemistry: str = "4color") -> Dict:
    """Exact collision statistics for all unordered 2-gRNA pairs.

    Parameters
    ----------
    spacers : List[str]
    L : int
        Prefix length.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    dict with keys:
        L, total_pairs, unique_signals, ambiguous_pairs, ambiguous_fraction,
        n_collision_groups, max_group_size, collision_groups, mapping.
    """
    pairs = all_pairs(len(spacers))
    mapping: Dict[bytes, List[Tuple[int, int]]] = {}
    for i, j in pairs:
        k = signal_key(pair_sum(spacers[i], spacers[j], L, chemistry))
        mapping.setdefault(k, []).append((i, j))
    total = len(pairs)
    collision_groups = {k: v for k, v in mapping.items() if len(v) > 1}
    ambiguous = sum(len(v) for v in collision_groups.values())
    return {
        "L": L,
        "total_pairs": total,
        "unique_signals": len(mapping),
        "ambiguous_pairs": ambiguous,
        "ambiguous_fraction": ambiguous / total,
        "n_collision_groups": len(collision_groups),
        "max_group_size": max((len(v) for v in collision_groups.values()), default=1),
        "collision_groups": collision_groups,
        "mapping": mapping,
    }


# ---------------------------------------------------------------------------
# Collision detection (k-plex, k ≥ 1)
# ---------------------------------------------------------------------------

def collision_stats_k(
    spacers: List[str], k: int, L: int, chemistry: str = "4color"
) -> Dict:
    """Exact collision statistics for all k-plex combinations (k ≥ 1).

    Parameters
    ----------
    spacers : List[str]
    k : int
        Number of gRNAs per cell (combination order).
    L : int
        Prefix length.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    dict with keys:
        k, L, total, unique, ambiguous, ambiguous_fraction,
        max_group_size, n_collision_groups.
    """
    n = len(spacers)
    mapping: Dict[bytes, int] = defaultdict(int)
    total = 0
    for combo in combinations_with_replacement(range(n), k):
        key = signal_key(k_sum_signal(spacers, combo, L, chemistry))
        mapping[key] += 1
        total += 1
    ambiguous = sum(cnt for cnt in mapping.values() if cnt >= 2)
    return {
        "k": k,
        "L": L,
        "total": total,
        "unique": len(mapping),
        "ambiguous": ambiguous,
        "ambiguous_fraction": ambiguous / total,
        "max_group_size": max(mapping.values()),
        "n_collision_groups": sum(1 for cnt in mapping.values() if cnt >= 2),
    }


def min_L_for_ambig(
    spacers: List[str], k: int, Ls=range(1, 21), thresh: float = 1e-3,
    chemistry: str = "4color",
) -> Tuple[Optional[int], Optional[float]]:
    """Find minimum prefix length L where ambiguous_fraction ≤ thresh."""
    for L in Ls:
        af = collision_stats_k(spacers, k, L, chemistry)["ambiguous_fraction"]
        if af <= thresh:
            return L, af
    return None, None


# ---------------------------------------------------------------------------
# Lookup table + deconvolution (k = 2)
# ---------------------------------------------------------------------------

def build_lookup_table(spacers: List[str], L: int, chemistry: str = "4color") -> Dict:
    """Build a signal → pair lookup table for exact deconvolution.

    Returns a dict mapping signal_key(bytes) to:
    - (i, j) tuple if the signal is unambiguous
    - list of (i, j) tuples if the signal is ambiguous (collision)
    """
    pairs = all_pairs(len(spacers))
    lookup: Dict = {}
    ambiguous: Dict = {}
    for i, j in pairs:
        k = signal_key(pair_sum(spacers[i], spacers[j], L, chemistry))
        if k in lookup:
            if k not in ambiguous:
                ambiguous[k] = [lookup[k]]
            ambiguous[k].append((i, j))
        else:
            lookup[k] = (i, j)
    for k, pairs_list in ambiguous.items():
        lookup[k] = pairs_list
    return lookup


def deconvolve_signal(
    observed: np.ndarray, lookup_table: Dict
) -> Tuple[Optional[object], bool]:
    """Deconvolve an observed sum-signal using a prebuilt lookup table.

    Parameters
    ----------
    observed : np.ndarray
        Shape (L, 4) observed signal.
    lookup_table : dict
        From build_lookup_table().

    Returns
    -------
    result : (i, j) tuple if unambiguous, list of (i, j) if ambiguous, None if missing.
    is_unambiguous : bool
    """
    k = signal_key(observed)
    if k not in lookup_table:
        return None, False
    result = lookup_table[k]
    return result, not isinstance(result, list)


def deconvolution_counts(spacers: List[str], L: int, chemistry: str = "4color") -> Dict:
    """Test noiseless deconvolution accuracy for all 2-gRNA pairs.

    Returns
    -------
    dict with keys:
        L, correct, ambiguous, failed, total,
        accuracy_strict (correct/total),
        accuracy_lenient ((correct+ambiguous)/total).
    """
    stats = collision_stats(spacers, L, chemistry)
    mapping = stats["mapping"]
    pairs = all_pairs(len(spacers))
    correct = ambiguous = failed = 0
    for i, j in pairs:
        k = signal_key(pair_sum(spacers[i], spacers[j], L, chemistry))
        cand = mapping.get(k, [])
        if len(cand) == 0:
            failed += 1
        elif len(cand) == 1 and cand[0] == (i, j):
            correct += 1
        else:
            ambiguous += 1
    total = len(pairs)
    return {
        "L": L,
        "correct": correct,
        "ambiguous": ambiguous,
        "failed": failed,
        "total": total,
        "accuracy_strict": correct / total,
        "accuracy_lenient": (correct + ambiguous) / total,
    }


# ---------------------------------------------------------------------------
# Dropout simulation
# ---------------------------------------------------------------------------

def simulate_dropout(
    signal: np.ndarray, p: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply independent Bernoulli position dropout to a signal.

    Each position p is independently masked with probability `p`.  At a masked
    position the 4-channel vector is zeroed out — we know a read occurred at
    that cycle but not which base was called.

    Parameters
    ----------
    signal : np.ndarray
        Shape (L, 4) observed signal.
    p : float
        Per-position dropout probability in [0, 1].
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    dropped_signal : np.ndarray
        Shape (L, 4) with zeroed rows at dropped positions.
    drop_mask : np.ndarray
        Shape (L,) boolean, True where position was dropped.
    """
    L = signal.shape[0]
    drop_mask = rng.random(L) < p
    dropped = signal.copy()
    dropped[drop_mask] = 0
    return dropped, drop_mask


def deconvolve_with_dropout(
    observed: np.ndarray,
    drop_mask: np.ndarray,
    pair_signals: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Find all candidate pairs compatible with a partially-observed signal.

    Dropped positions carry no information, so only non-dropped positions are
    compared.  A candidate pair is compatible if its predicted signal matches
    the observed signal at every non-dropped position.

    Parameters
    ----------
    observed : np.ndarray
        Shape (L, 4) — zeros at dropped positions.
    drop_mask : np.ndarray
        Shape (L,) bool — True where position was dropped.
    pair_signals : np.ndarray
        Shape (n_pairs, L, 4) precomputed from generate_imaging_matrix().
    pairs : List[Tuple[int, int]]
        Pair indices corresponding to each row of pair_signals.

    Returns
    -------
    List[Tuple[int, int]]
        All candidate pairs whose signal is compatible with observed.
    """
    non_dropped = ~drop_mask
    if not non_dropped.any():
        return list(pairs)  # all positions dropped → all pairs are candidates
    obs_nd = observed[non_dropped]          # (L', 4)
    pred_nd = pair_signals[:, non_dropped]  # (n_pairs, L', 4)
    match = np.all(pred_nd == obs_nd[None], axis=(1, 2))
    return [pairs[idx] for idx in np.where(match)[0]]


def deconvolution_counts_dropout(
    spacers: List[str],
    L: int,
    dropout_p: float,
    n_trials: int = 200,
    seed: int = 42,
    chemistry: str = "4color",
) -> Dict:
    """Monte Carlo dropout simulation for 2-gRNA pair deconvolution.

    For each true pair (i, j), runs n_trials trials each with independently
    drawn position dropout.  Averages outcome counts across trials.

    Parameters
    ----------
    spacers : List[str]
    L : int
        Prefix length.
    dropout_p : float
        Per-position dropout probability.
    n_trials : int
        Number of Monte Carlo trials per pair.
    seed : int
        Random seed for reproducibility.
    chemistry : str
        '4color' (default) or '2color'.

    Returns
    -------
    dict with keys:
        L, dropout_p, n_trials,
        correct, ambiguous, failed, total  (summed over all pairs × trials),
        accuracy_strict, accuracy_lenient.
    """
    rng = np.random.default_rng(seed)
    pair_signals, pairs = generate_imaging_matrix(spacers, L, chemistry)

    correct = ambiguous = failed = 0
    for true_i, true_j in pairs:
        true_signal = pair_sum(spacers[true_i], spacers[true_j], L, chemistry)
        for _ in range(n_trials):
            obs, mask = simulate_dropout(true_signal, dropout_p, rng)
            candidates = deconvolve_with_dropout(obs, mask, pair_signals, pairs)
            if len(candidates) == 1 and candidates[0] == (true_i, true_j):
                correct += 1
            elif (true_i, true_j) in candidates:
                ambiguous += 1
            else:
                failed += 1

    total = len(pairs) * n_trials
    return {
        "L": L,
        "dropout_p": dropout_p,
        "n_trials": n_trials,
        "correct": correct,
        "ambiguous": ambiguous,
        "failed": failed,
        "total": total,
        "accuracy_strict": correct / total,
        "accuracy_lenient": (correct + ambiguous) / total,
    }


def deconvolution_counts_positional(
    spacers: List[str], L: int, n_drop: int = 1, chemistry: str = "4color"
) -> Dict[Tuple[int, ...], Dict]:
    """Deterministic deconvolution for all combinations of exactly n_drop dropped positions.

    Exhaustively enumerates C(L, n_drop) position combinations.  For each
    combination the same drop mask is applied to every pair's signal, and
    deconvolution accuracy is computed over all pairs.

    Parameters
    ----------
    spacers : List[str]
    L : int
        Prefix length.
    n_drop : int
        Number of positions to drop simultaneously (default 1).
        Recommended: n_drop ≤ 3 for L=10, n_drop ≤ 5 for L=20.

    Returns
    -------
    dict mapping drop_combo (tuple of ints) -> {
        correct, ambiguous, failed, total,
        accuracy_strict, accuracy_lenient,
        drop_positions
    }
    """
    pair_signals, pairs = generate_imaging_matrix(spacers, L, chemistry)
    results: Dict[Tuple[int, ...], Dict] = {}

    for drop_positions in combinations(range(L), n_drop):
        drop_mask = np.zeros(L, dtype=bool)
        drop_mask[list(drop_positions)] = True

        correct = ambiguous = failed = 0
        for idx, (true_i, true_j) in enumerate(pairs):
            obs = pair_signals[idx].copy()
            obs[drop_mask] = 0
            candidates = deconvolve_with_dropout(obs, drop_mask, pair_signals, pairs)
            if len(candidates) == 1 and candidates[0] == (true_i, true_j):
                correct += 1
            elif (true_i, true_j) in candidates:
                ambiguous += 1
            else:
                failed += 1

        total = len(pairs)
        results[drop_positions] = {
            "drop_positions": drop_positions,
            "correct": correct,
            "ambiguous": ambiguous,
            "failed": failed,
            "total": total,
            "accuracy_strict": correct / total,
            "accuracy_lenient": (correct + ambiguous) / total,
        }

    return results


# ---------------------------------------------------------------------------
# Utility / sanity checks
# ---------------------------------------------------------------------------

def hamming_distance_prefix(a: str, b: str, L: int) -> int:
    """Hamming distance between the first L bases of two sequences."""
    return sum(ca != cb for ca, cb in zip(a[:L], b[:L]))


def sanity_onehot(
    spacers: List[str], L: int = 10, n_test: int = 5, chemistry: str = "4color"
) -> None:
    """Assert that encoding is correct for first n_test spacers."""
    n_ch = 4 if chemistry == "4color" else 2
    for idx in range(min(n_test, len(spacers))):
        X = encode_seq(spacers[idx], L, chemistry)
        assert X.shape == (L, n_ch), f"expected ({L},{n_ch}), got {X.shape}"
        assert set(np.unique(X)).issubset({0, 1}), "only 0/1 values expected"
        if chemistry == "4color":
            assert np.all(X.sum(axis=1) == 1), "4-color: each position must sum to 1"
    print(f"PASS: encoding checks ({chemistry})")


def sanity_ksum(
    spacers: List[str], k: int = 6, L: int = 7, n_trials: int = 50, seed: int = 0,
    chemistry: str = "4color",
) -> None:
    """Assert that k-sum signals have correct shape and value range."""
    import random
    random.seed(seed)
    n_ch = 4 if chemistry == "4color" else 2
    n = len(spacers)
    for _ in range(n_trials):
        combo = tuple(sorted([random.randrange(n) for _ in range(k)]))
        S = k_sum_signal(spacers, combo, L, chemistry)
        assert S.shape == (L, n_ch), f"expected ({L},{n_ch}), got {S.shape}"
        assert 0 <= S.min() and S.max() <= k
        if chemistry == "4color":
            assert np.all(S.sum(axis=1) == k), "4-color: each position must sum to k"
    print(f"PASS: k-sum signal checks ({chemistry})")


def sample_collision_audit(
    spacers: List[str], k: int = 6, L: int = 7, n_samples: int = 20000, seed: int = 0,
    chemistry: str = "4color",
) -> Optional[Tuple]:
    """Random-sample collision audit for k-plex signals."""
    import random
    random.seed(seed)
    n = len(spacers)
    seen: Dict = defaultdict(list)
    for _ in range(n_samples):
        combo = tuple(sorted([random.randrange(n) for _ in range(k)]))
        sig = signal_key(k_sum_signal(spacers, combo, L, chemistry))
        seen[sig].append(combo)

    collisions = []
    for sig, combos in seen.items():
        uniq = list(dict.fromkeys(combos))
        if len(uniq) >= 2:
            collisions.append((sig, uniq))

    print(f"Sampled combos: {n_samples}, collision groups found: {len(collisions)}")
    if collisions:
        _, combos = collisions[0]
        c1, c2 = combos[0], combos[1]
        S1 = k_sum_signal(spacers, c1, L, chemistry)
        S2 = k_sum_signal(spacers, c2, L, chemistry)
        print(f"Example collision: {c1} vs {c2}, equal? {np.array_equal(S1, S2)}")
        return c1, c2
    return None


def k1_prefix_check(spacers: List[str], L: int) -> None:
    """Print how many k=1 prefix collisions exist at length L."""
    prefs = [s[:L] for s in spacers]
    c = Counter(prefs)
    groups = sum(1 for v in c.values() if v > 1)
    collided = sum(v for v in c.values() if v > 1)
    print(f"k=1, L={L}: {groups} duplicate groups, {collided} collided guides")


def min_L_to_resolve(
    spacers: List[str], combo1: tuple, combo2: tuple, L_max: int = 30,
    chemistry: str = "4color",
) -> Optional[int]:
    """Find minimum prefix length at which two k-plex combos produce different signals."""
    for L in range(1, L_max + 1):
        if not np.array_equal(k_sum_signal(spacers, combo1, L, chemistry),
                              k_sum_signal(spacers, combo2, L, chemistry)):
            return L
    return None


# ---------------------------------------------------------------------------
# Visualizations — noiseless (deterministic)
# ---------------------------------------------------------------------------

def plot_ambiguity_vs_L(
    spacers: List[str],
    Ls=range(4, 21),
    mark: Tuple[int, ...] = (10, 20),
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Line plot of ambiguous_fraction vs prefix length (k=2 pairs)."""
    Ls = list(Ls)
    amb = [collision_stats(spacers, L, chemistry)["ambiguous_fraction"] for L in Ls]
    plt.figure()
    plt.plot(Ls, amb, marker="o")
    plt.xlabel("Prefix length L (nt)")
    plt.ylabel("Ambiguous fraction of 2-gRNA pairs")
    plt.title(f"Exact-collision ambiguity vs read length (noiseless {chemistry} sum)")
    for Lm in mark:
        if Lm in Ls:
            idx = Ls.index(Lm)
            plt.scatter([Lm], [amb[idx]])
            plt.text(Lm + 0.2, amb[idx], f"{Lm}nt", va="center")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_deconvolution_bars(
    spacers: List[str],
    Ls=(1, 2, 3, 4, 5, 8, 10, 15, 20),
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Grouped bar chart of correct / ambiguous / failed counts per prefix length."""
    cats = ["correct", "ambiguous", "failed"]
    vals = [[deconvolution_counts(spacers, L, chemistry)[c] for c in cats] for L in Ls]
    x = np.arange(len(Ls))
    w = 0.25
    plt.figure(figsize=(max(8, len(Ls) * 0.8), 4))
    plt.bar(x - w, [v[0] for v in vals], w, label="Correct (unique)")
    plt.bar(x,      [v[1] for v in vals], w, label="Ambiguous (collision)")
    plt.bar(x + w,  [v[2] for v in vals], w, label="Failed")
    plt.xticks(x, [f"{L}nt" for L in Ls])
    plt.ylabel("# of 2-gRNA pairs")
    plt.title("Noiseless deconvolution outcome (exact lookup)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_nearest_neighbor_risk(
    spacers: List[str],
    L: int = 10,
    n_show: int = 10,
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Heatmap grid of the n_show most confusable pair-sum signals (by NN distance)."""
    pairs = all_pairs(len(spacers))
    X = np.stack([pair_sum(spacers[i], spacers[j], L, chemistry).reshape(-1)
                  for i, j in pairs]).astype(float)
    dmin = np.full(len(pairs), np.inf)
    nn = np.full(len(pairs), -1, dtype=int)
    for a in range(len(pairs)):
        dist = np.linalg.norm(X - X[a], axis=1)
        dist[a] = np.inf
        b = int(dist.argmin())
        dmin[a] = dist[b]
        nn[a] = b

    riskiest = np.argsort(dmin)[:n_show]
    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    ch_labels = list(BASES) if chemistry == "4color" else ["green", "red"]
    for k, idx in enumerate(riskiest):
        i, j = pairs[idx]
        bi, bj = pairs[nn[idx]]
        sig = pair_sum(spacers[i], spacers[j], L, chemistry)
        axes[k].imshow(sig.T, aspect="auto", vmin=0, vmax=2)
        axes[k].set_yticks(range(len(ch_labels)))
        axes[k].set_yticklabels(ch_labels)
        axes[k].set_title(f"({i},{j}) nn=({bi},{bj})\nd={dmin[idx]:.2f}", fontsize=8)

    for k in range(n_show, len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"Nearest-neighbor confusability (L={L})", fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_nn_distance_distribution(
    spacers: List[str],
    L: int = 10,
    bins: int = 30,
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Histogram of nearest-neighbor distances across all pair-sum signals."""
    from scipy.spatial.distance import pdist, squareform
    pairs = all_pairs(len(spacers))
    X = np.stack([pair_sum(spacers[i], spacers[j], L, chemistry).reshape(-1)
                  for i, j in pairs]).astype(float)
    D = squareform(pdist(X, metric="euclidean"))
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)

    plt.figure()
    plt.hist(nn, bins=bins)
    plt.xlabel("Nearest-neighbor distance (Euclidean)")
    plt.ylabel("Count of gRNA pairs")
    plt.title(f"Distribution of nearest-neighbor distances (L={L} nt)")
    plt.grid(True, axis="y", alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()

    print(f"n pairs: {len(pairs)}")
    print(f"min/median/max: {nn.min():.3f} / {float(np.median(nn)):.3f} / {nn.max():.3f}")


def visualize_pairs_by_within_similarity(
    spacers: List[str],
    prefix_len: int = 10,
    n_show: int = 10,
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Heatmaps of n_show most within-similar non-self gRNA pairs (by Hamming distance)."""
    pairs_distinct = list(combinations(range(len(spacers)), 2))
    scored = sorted(
        [(hamming_distance_prefix(spacers[i], spacers[j], prefix_len), i, j)
         for i, j in pairs_distinct]
    )
    top = scored[:n_show]

    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    ch_labels = list(BASES) if chemistry == "4color" else ["green", "red"]
    for ax_idx, (d, i, j) in enumerate(top):
        sig = pair_sum(spacers[i], spacers[j], prefix_len, chemistry)
        axes[ax_idx].imshow(sig.T, aspect="auto", vmin=0, vmax=2)
        axes[ax_idx].set_yticks(range(len(ch_labels)))
        axes[ax_idx].set_yticklabels(ch_labels)
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_title(
            f"({i},{j}) ham={d}/{prefix_len}", fontsize=8
        )

    for ax_idx in range(n_show, len(axes)):
        axes[ax_idx].axis("off")

    plt.suptitle(
        f"Most similar non-self gRNA pairs (L={prefix_len})", fontsize=12
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_ambig_vs_k(stats: List[Dict], title: Optional[str] = None,
                    output_path: Optional[str] = None) -> None:
    """Line plot of ambiguous_fraction vs k for a fixed L."""
    ks = [s["k"] for s in stats]
    amb = [s["ambiguous_fraction"] for s in stats]
    plt.figure()
    plt.plot(ks, amb, marker="o")
    plt.xticks(ks)
    plt.xlabel("k (gRNAs per cell)")
    plt.ylabel("Ambiguous fraction (exact collisions)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title(title or f"Collision scaling (L={stats[0]['L']} nt)")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_ambig_vs_k_many_L(
    spacers: List[str],
    ks=range(1, 6),
    Ls=(3, 4, 5, 6, 10, 20),
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Ambiguous fraction vs k, one line per L value."""
    plt.figure()
    for L in Ls:
        amb = [collision_stats_k(spacers, k, L, chemistry)["ambiguous_fraction"] for k in ks]
        plt.plot(list(ks), amb, marker="o", label=f"L={L}")
    plt.xticks(list(ks))
    plt.xlabel("k (gRNAs per cell)")
    plt.ylabel("Ambiguous fraction (exact collisions)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title("Exact-collision ambiguity vs k for different barcode lengths")
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_ambig_vs_L_many_k(
    spacers: List[str],
    Ls=(3, 4, 5, 6, 10, 20),
    ks=range(1, 6),
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Ambiguous fraction vs L, one line per k value."""
    plt.figure()
    for k in ks:
        amb = [collision_stats_k(spacers, k, L, chemistry)["ambiguous_fraction"] for L in Ls]
        plt.plot(list(Ls), amb, marker="o", label=f"k={k}")
    plt.xticks(list(Ls))
    plt.xlabel("L (barcode length, nt)")
    plt.ylabel("Ambiguous fraction (exact collisions)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title("Exact-collision ambiguity vs barcode length for different k")
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Visualizations — dropout simulation
# ---------------------------------------------------------------------------

def plot_dropout_ambiguity_vs_L(
    spacers: List[str],
    Ls: List[int],
    dropout_ps: List[float],
    n_trials: int = 200,
    seed: int = 42,
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Ambiguous fraction vs L for multiple dropout rates (Monte Carlo).

    One line per dropout_p value; dropout_p=0 matches the noiseless baseline.
    """
    plt.figure()
    for p in dropout_ps:
        ambig = []
        for L in Ls:
            res = deconvolution_counts_dropout(spacers, L, p, n_trials, seed, chemistry)
            total = res["total"]
            ambig.append(res["ambiguous"] / total)
        plt.plot(Ls, ambig, marker="o", label=f"p={p:.2f}")
    plt.xlabel("Prefix length L (nt)")
    plt.ylabel("Ambiguous fraction")
    plt.title(f"Ambiguity vs L under position dropout (n_trials={n_trials})")
    plt.legend(title="Dropout prob.")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_dropout_accuracy_heatmap(
    results_grid: Dict[Tuple, Dict],
    Ls: List[int],
    dropout_ps: List[float],
    metric: str = "accuracy_strict",
    output_path: Optional[str] = None,
) -> None:
    """Heatmap of deconvolution accuracy across (L, dropout_p) grid.

    Parameters
    ----------
    results_grid : dict
        Mapping (L, dropout_p) -> result dict from deconvolution_counts_dropout().
    Ls, dropout_ps : lists
    metric : str
        'accuracy_strict' or 'accuracy_lenient'.
    """
    data = np.array([[results_grid[(L, p)][metric] for p in dropout_ps] for L in Ls])
    plt.figure(figsize=(max(6, len(dropout_ps) * 1.0), max(4, len(Ls) * 0.6)))
    im = plt.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn",
                    origin="lower")
    plt.colorbar(im, label=metric.replace("_", " "))
    plt.xticks(range(len(dropout_ps)), [f"{p:.2f}" for p in dropout_ps])
    plt.yticks(range(len(Ls)), Ls)
    plt.xlabel("Dropout probability per position")
    plt.ylabel("Prefix length L (nt)")
    plt.title(f"Deconvolution {metric.replace('_', ' ')} under position dropout")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_dropout_bars(
    spacers: List[str],
    L: int,
    dropout_ps: List[float],
    n_trials: int = 200,
    seed: int = 42,
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Grouped bar chart of correct/ambiguous/failed at fixed L, varying dropout."""
    cats = ["correct", "ambiguous", "failed"]
    vals = [
        [deconvolution_counts_dropout(spacers, L, p, n_trials, seed, chemistry)[c] for c in cats]
        for p in dropout_ps
    ]
    x = np.arange(len(dropout_ps))
    w = 0.25
    plt.figure()
    plt.bar(x - w, [v[0] for v in vals], w, label="Correct (unique)")
    plt.bar(x,      [v[1] for v in vals], w, label="Ambiguous")
    plt.bar(x + w,  [v[2] for v in vals], w, label="Failed")
    plt.xticks(x, [f"p={p:.2f}" for p in dropout_ps])
    plt.ylabel(f"# pair × trial outcomes  (L={L} nt)")
    plt.title(f"Deconvolution outcome under dropout (L={L} nt, n_trials={n_trials})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Visualizations — positional dropout
# ---------------------------------------------------------------------------

def plot_positional_dropout_bars(
    results: Dict[Tuple, Dict],
    L: int,
    n_drop: int = 1,
    metric: str = "accuracy_strict",
    output_path: Optional[str] = None,
) -> None:
    """Bar chart of accuracy vs dropped position (n_drop=1 only makes positional sense)."""
    if n_drop == 1:
        positions = sorted(results.keys(), key=lambda x: x[0])
        pos_labels = [str(dp[0]) for dp in positions]
        accs = [results[dp][metric] for dp in positions]
        plt.figure(figsize=(max(6, L * 0.6), 4))
        plt.bar(range(len(positions)), accs)
        plt.xticks(range(len(positions)), pos_labels)
        plt.xlabel("Dropped position index")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"Accuracy when single position dropped (L={L} nt)")
        plt.ylim(0, 1.05)
        plt.grid(True, axis="y", alpha=0.3)
    else:
        # Aggregate over all combos for n_drop > 1 — show distribution
        accs = [v[metric] for v in results.values()]
        plt.figure()
        plt.hist(accs, bins=20, edgecolor="black")
        plt.xlabel(metric.replace("_", " "))
        plt.ylabel("# drop combinations")
        plt.title(f"Accuracy distribution: {n_drop} positions dropped (L={L} nt)")
        plt.grid(True, axis="y", alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_n_drop_vs_accuracy(
    spacers: List[str],
    L: int,
    n_drops: List[int],
    metric: str = "accuracy_strict",
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Bar chart of mean ± std accuracy vs number of dropped positions.

    Runs deconvolution_counts_positional for each n_drop and aggregates results.

    Parameters
    ----------
    spacers : List[str]
    L : int
    n_drops : List[int]
        e.g. [1, 2, 3] for L=10, [1, 2, 3, 4, 5] for L=20.
    metric : str
    chemistry : str
        '4color' (default) or '2color'.
    """
    means = []
    stds = []
    for n_drop in n_drops:
        res = deconvolution_counts_positional(spacers, L, n_drop, chemistry)
        accs = np.array([v[metric] for v in res.values()])
        means.append(accs.mean())
        stds.append(accs.std())

    plt.figure()
    plt.bar(range(len(n_drops)), means, yerr=stds, capsize=5)
    plt.xticks(range(len(n_drops)), n_drops)
    plt.xlabel("# positions dropped")
    plt.ylabel(f"Mean {metric.replace('_', ' ')} ± std")
    plt.title(f"Deconvolution accuracy vs # dropped positions (L={L} nt)")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_positional_accuracy_heatmap(
    spacers: List[str],
    Ls: List[int],
    metric: str = "accuracy_strict",
    output_path: Optional[str] = None,
    chemistry: str = "4color",
) -> None:
    """Heatmap of single-position dropout accuracy: y=L, x=dropped position.

    Shows which positions carry the most discriminative information.
    Cells where dropped_position ≥ L are masked (shown as NaN).
    """
    max_L = max(Ls)
    data = np.full((len(Ls), max_L), np.nan)

    for li, L in enumerate(Ls):
        res = deconvolution_counts_positional(spacers, L, n_drop=1, chemistry=chemistry)
        for (pos,), v in res.items():
            data[li, pos] = v[metric]

    plt.figure(figsize=(max(6, max_L * 0.6), max(4, len(Ls) * 0.5)))
    im = plt.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn", origin="lower")
    plt.colorbar(im, label=metric.replace("_", " "))
    plt.xticks(range(max_L), range(max_L))
    plt.yticks(range(len(Ls)), Ls)
    plt.xlabel("Dropped position index")
    plt.ylabel("Prefix length L (nt)")
    plt.title(f"Single-position dropout accuracy ({metric.replace('_', ' ')})")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()
