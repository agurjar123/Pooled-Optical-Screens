"""
gRNA Deconvolution Pipeline
============================
Analyzes distinguishability of k gRNAs per cell using 4-color ISS chemistry.

Two levels of analysis:
1. INDIVIDUAL: Are single gRNA barcodes distinguishable from each other?
2. COMBINATORIAL: Are k-way combinations distinguishable from each other?

Usage:
    results = run_pipeline(spacers, k=2, prefix_len=20)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform


# =============================================================================
# ENCODING
# =============================================================================

BASE_TO_4COLOR = {
    'A': np.array([1, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0]),
    'G': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1]),
}

def encode_barcode(sequence: str, prefix_len: Optional[int] = None) -> np.ndarray:
    """Encode sequence as (L, 4) matrix."""
    seq = sequence[:prefix_len].upper() if prefix_len else sequence.upper()
    return np.array([BASE_TO_4COLOR[base] for base in seq])


def signal_to_hash(signal: np.ndarray) -> tuple:
    """Convert signal matrix to hashable tuple."""
    return tuple(map(tuple, signal))


# =============================================================================
# LEVEL 1: INDIVIDUAL gRNA ANALYSIS
# =============================================================================

def analyze_individual_grnas(
    spacers: List[str],
    prefix_len: Optional[int] = None
) -> Dict:
    """
    Analyze distinguishability of individual gRNAs (k=1).
    
    Returns
    -------
    dict with:
        - encoded: (n, L, 4) array of encoded barcodes
        - distance_matrix: (n, n) pairwise distances
        - collisions: list of (i, j) pairs with identical sequences
        - min_distance: smallest non-zero distance
        - most_similar: list of (i, j, dist) sorted by similarity
    """
    n = len(spacers)
    L = prefix_len or len(spacers[0])
    
    # Encode all
    encoded = np.array([encode_barcode(s, prefix_len) for s in spacers])
    
    # Check for identical sequences (collisions)
    seq_to_idx = defaultdict(list)
    for i, s in enumerate(spacers):
        seq_to_idx[s[:L].upper()].append(i)
    
    collisions = [(idxs[0], idxs[1]) for idxs in seq_to_idx.values() if len(idxs) > 1]
    
    # Distance matrix
    flat = encoded.reshape(n, -1)
    distance_matrix = squareform(pdist(flat, metric='euclidean'))
    
    # Find most similar pairs
    dist_copy = distance_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    
    most_similar = []
    for i in range(n):
        for j in range(i+1, n):
            most_similar.append((i, j, distance_matrix[i, j]))
    most_similar.sort(key=lambda x: x[2])
    
    min_dist = dist_copy.min() if dist_copy.size > 0 else np.inf
    
    return {
        'n_grnas': n,
        'prefix_len': L,
        'encoded': encoded,
        'distance_matrix': distance_matrix,
        'collisions': collisions,
        'n_collisions': len(collisions),
        'distinguishable': len(collisions) == 0,
        'min_distance': min_dist,
        'most_similar': most_similar[:10],  # top 10
    }


# =============================================================================
# LEVEL 2: COMBINATORIAL ANALYSIS
# =============================================================================

def compute_combo_signal(
    encoded_grnas: np.ndarray,
    indices: Tuple[int, ...]
) -> np.ndarray:
    """Sum encoded signals for a combination of gRNAs."""
    return sum(encoded_grnas[i] for i in indices)


def analyze_combinations(
    spacers: List[str],
    k: int,
    prefix_len: Optional[int] = None,
    max_combos: int = 500000  # limit for memory
) -> Dict:
    """
    Analyze distinguishability of k-way combinations.
    
    Returns
    -------
    dict with:
        - n_combos: total number of combinations
        - imaging_matrix: (n_combos, L, 4) array
        - combos: list of index tuples
        - collisions: dict mapping hash -> list of colliding combos
        - n_collision_groups: number of collision groups
        - distinguishable: bool
        - min_distance: smallest distance between distinct combos
        - most_similar: list of (combo1, combo2, dist)
    """
    n = len(spacers)
    L = prefix_len or len(spacers[0])
    
    # Generate combinations
    all_combos = list(combinations_with_replacement(range(n), k))
    
    if len(all_combos) > max_combos:
        print(f"Warning: {len(all_combos)} combos exceeds max ({max_combos}). Sampling.")
        np.random.seed(42)
        idx = np.random.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]
    
    # Encode individual gRNAs once
    encoded_grnas = np.array([encode_barcode(s, prefix_len) for s in spacers])
    
    # Compute all combo signals
    imaging_matrix = np.zeros((len(all_combos), L, 4))
    for idx, combo in enumerate(all_combos):
        imaging_matrix[idx] = compute_combo_signal(encoded_grnas, combo)
    
    # Detect collisions (identical signals)
    hash_to_combos = defaultdict(list)
    for idx, combo in enumerate(all_combos):
        h = signal_to_hash(imaging_matrix[idx])
        hash_to_combos[h].append(combo)
    
    collision_groups = {h: combos for h, combos in hash_to_combos.items() if len(combos) > 1}
    
    # Distance analysis (only if reasonable size)
    min_dist = np.inf
    most_similar = []
    
    if len(all_combos) <= 10000:
        flat = imaging_matrix.reshape(len(all_combos), -1)
        distances = squareform(pdist(flat, metric='euclidean'))
        np.fill_diagonal(distances, np.inf)
        
        min_dist = distances.min()
        
        # Find most similar pairs
        n_similar = min(10, len(all_combos) * (len(all_combos) - 1) // 2)
        flat_idx = np.argsort(distances.ravel())
        
        seen = set()
        for fi in flat_idx:
            i, j = np.unravel_index(fi, distances.shape)
            if i >= j:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            most_similar.append((all_combos[i], all_combos[j], distances[i, j]))
            if len(most_similar) >= 10:
                break
    
    return {
        'k': k,
        'n_combos': len(all_combos),
        'n_unique_signals': len(hash_to_combos),
        'imaging_matrix': imaging_matrix,
        'combos': all_combos,
        'collision_groups': collision_groups,
        'n_collision_groups': len(collision_groups),
        'n_colliding_combos': sum(len(v) for v in collision_groups.values()),
        'distinguishable': len(collision_groups) == 0,
        'min_distance': min_dist,
        'most_similar': most_similar,
        'prefix_len': L,
    }


# =============================================================================
# DECONVOLUTION TEST
# =============================================================================

def test_deconvolution(
    spacers: List[str],
    k: int,
    prefix_len: Optional[int] = None,
    noise_std: float = 0.0,
    n_trials: int = 1000,
    abundance_var: float = 0.0  # variation in gRNA abundance (0 = equal)
) -> Dict:
    """
    Test deconvolution accuracy: given a signal, can we recover the combo?
    
    Parameters
    ----------
    noise_std : float
        Gaussian noise standard deviation (0 = noiseless)
    abundance_var : float
        Variation in relative abundance (0 = 50:50, higher = more variable)
    
    Returns
    -------
    dict with accuracy metrics
    """
    combo_result = analyze_combinations(spacers, k, prefix_len)
    imaging_matrix = combo_result['imaging_matrix']
    combos = combo_result['combos']
    n_combos = len(combos)
    
    if n_combos == 0:
        return {'error': 'No combinations generated'}
    
    # Encode individual gRNAs for variable abundance
    encoded_grnas = np.array([encode_barcode(s, prefix_len) for s in spacers])
    L = combo_result['prefix_len']
    
    correct = 0
    correct_top3 = 0
    total_distance_to_true = 0
    
    for trial in range(n_trials):
        # Pick random true combination
        true_idx = np.random.randint(n_combos)
        true_combo = combos[true_idx]
        
        # Generate observed signal
        if abundance_var > 0:
            # Variable abundance
            weights = np.random.dirichlet(np.ones(k) / abundance_var)
            observed = sum(w * encoded_grnas[i] for w, i in zip(weights, true_combo))
        else:
            # Equal abundance
            observed = imaging_matrix[true_idx].copy()
        
        # Add noise
        if noise_std > 0:
            observed = observed + np.random.normal(0, noise_std, observed.shape)
        
        # Find nearest match
        distances = np.linalg.norm(imaging_matrix - observed, axis=(1, 2))
        pred_idx = np.argmin(distances)
        
        # Top-3 accuracy
        top3_idx = np.argsort(distances)[:3]
        
        if pred_idx == true_idx:
            correct += 1
        if true_idx in top3_idx:
            correct_top3 += 1
        
        total_distance_to_true += distances[true_idx]
    
    return {
        'k': k,
        'prefix_len': combo_result['prefix_len'],
        'n_combos': n_combos,
        'noise_std': noise_std,
        'abundance_var': abundance_var,
        'n_trials': n_trials,
        'accuracy': correct / n_trials,
        'top3_accuracy': correct_top3 / n_trials,
        'mean_distance_to_true': total_distance_to_true / n_trials,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    spacers: List[str],
    k: int = 2,
    prefix_len: Optional[int] = None,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    n_trials: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Run complete analysis pipeline.
    
    Parameters
    ----------
    spacers : List[str]
        gRNA spacer sequences
    k : int
        Number of gRNAs per cell
    prefix_len : int
        Barcode length to use (None = full length)
    noise_levels : List[float]
        Noise std values to test
    n_trials : int
        Number of trials for deconvolution test
    
    Returns
    -------
    dict with all results
    """
    results = {
        'k': k,
        'n_grnas': len(spacers),
        'prefix_len': prefix_len or len(spacers[0]),
    }
    
    if verbose:
        print("=" * 60)
        print(f"gRNA DECONVOLUTION PIPELINE (k={k})")
        print("=" * 60)
        print(f"Library: {len(spacers)} gRNAs")
        print(f"Prefix length: {results['prefix_len']}nt")
        print()
    
    # Level 1: Individual gRNA analysis
    if verbose:
        print("-" * 60)
        print("LEVEL 1: Individual gRNA Distinguishability")
        print("-" * 60)
    
    individual = analyze_individual_grnas(spacers, prefix_len)
    results['individual'] = individual
    
    if verbose:
        print(f"  Collisions (identical barcodes): {individual['n_collisions']}")
        print(f"  Distinguishable: {individual['distinguishable']}")
        print(f"  Min pairwise distance: {individual['min_distance']:.3f}")
        print(f"  Most similar pairs:")
        for i, j, d in individual['most_similar'][:5]:
            print(f"    gRNA {i} vs {j}: dist={d:.3f}")
        print()
    
    # Level 2: Combinatorial analysis
    if verbose:
        print("-" * 60)
        print(f"LEVEL 2: {k}-way Combination Distinguishability")
        print("-" * 60)
    
    combos = analyze_combinations(spacers, k, prefix_len)
    results['combinations'] = combos
    
    if verbose:
        print(f"  Total combinations: {combos['n_combos']}")
        print(f"  Unique signals: {combos['n_unique_signals']}")
        print(f"  Collision groups: {combos['n_collision_groups']}")
        print(f"  Distinguishable: {combos['distinguishable']}")
        if combos['min_distance'] < np.inf:
            print(f"  Min pairwise distance: {combos['min_distance']:.3f}")
        if combos['most_similar']:
            print(f"  Most similar combos:")
            for c1, c2, d in combos['most_similar'][:5]:
                print(f"    {c1} vs {c2}: dist={d:.3f}")
        print()
    
    # Level 3: Deconvolution accuracy with noise
    if verbose:
        print("-" * 60)
        print("LEVEL 3: Deconvolution Accuracy")
        print("-" * 60)
    
    deconv_results = []
    for noise in noise_levels:
        dr = test_deconvolution(spacers, k, prefix_len, noise_std=noise, n_trials=n_trials)
        deconv_results.append(dr)
        if verbose:
            print(f"  Noise σ={noise:.1f}: accuracy={dr['accuracy']:.1%}, top3={dr['top3_accuracy']:.1%}")
    
    results['deconvolution'] = deconv_results
    
    if verbose:
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pipeline_results(
    results: Dict,
    output_prefix: str = "outputs/pipeline"
) -> None:
    """Generate visualizations for pipeline results."""
    
    k = results['k']
    prefix_len = results['prefix_len']
    
    # Plot 1: Individual gRNA distance matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    im1 = ax1.imshow(results['individual']['distance_matrix'], cmap='viridis')
    ax1.set_xlabel('gRNA index')
    ax1.set_ylabel('gRNA index')
    ax1.set_title(f'Individual gRNA Pairwise Distances\n(prefix {prefix_len}nt)')
    plt.colorbar(im1, ax=ax1, label='Euclidean distance')
    
    # Plot 2: Deconvolution accuracy vs noise
    ax2 = axes[1]
    noise_vals = [d['noise_std'] for d in results['deconvolution']]
    acc_vals = [d['accuracy'] for d in results['deconvolution']]
    top3_vals = [d['top3_accuracy'] for d in results['deconvolution']]
    
    ax2.plot(noise_vals, acc_vals, 'bo-', label='Top-1 accuracy', linewidth=2, markersize=8)
    ax2.plot(noise_vals, top3_vals, 'g^-', label='Top-3 accuracy', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise σ')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Deconvolution Accuracy vs Noise (k={k})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_k{k}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_k{k}.png")
    
    # Plot 3: Most similar combo signals (if available)
    if results['combinations']['most_similar']:
        n_show = min(5, len(results['combinations']['most_similar']))
        fig, axes = plt.subplots(n_show, 2, figsize=(12, 2.5*n_show))
        
        imaging_matrix = results['combinations']['imaging_matrix']
        combos = results['combinations']['combos']
        
        for row, (c1, c2, dist) in enumerate(results['combinations']['most_similar'][:n_show]):
            idx1 = combos.index(c1)
            idx2 = combos.index(c2)
            
            axes[row, 0].imshow(imaging_matrix[idx1].T, aspect='auto', cmap='viridis', vmin=0, vmax=k)
            axes[row, 0].set_yticks([0,1,2,3])
            axes[row, 0].set_yticklabels(['A','C','G','T'])
            axes[row, 0].set_title(f'Combo {c1}')
            axes[row, 0].set_ylabel(f'd={dist:.2f}')
            
            axes[row, 1].imshow(imaging_matrix[idx2].T, aspect='auto', cmap='viridis', vmin=0, vmax=k)
            axes[row, 1].set_yticks([])
            axes[row, 1].set_title(f'Combo {c2}')
        
        plt.suptitle(f'Most Similar {k}-way Combinations (prefix {prefix_len}nt)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_similar_k{k}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_prefix}_similar_k{k}.png")


def compare_k_values(
    spacers: List[str],
    k_values: List[int] = [1, 2, 3, 4],
    prefix_len: Optional[int] = None,
    noise_std: float = 0.2,
    n_trials: int = 1000,
    output_path: str = "outputs/k_comparison.png"
) -> pd.DataFrame:
    """
    Compare distinguishability across different k values.
    """
    records = []
    
    print("=" * 60)
    print("COMPARING k VALUES")
    print("=" * 60)
    
    for k in k_values:
        print(f"\nAnalyzing k={k}...")
        
        # Combinations
        combo_result = analyze_combinations(spacers, k, prefix_len)
        
        # Deconvolution
        deconv = test_deconvolution(spacers, k, prefix_len, noise_std=noise_std, n_trials=n_trials)
        
        records.append({
            'k': k,
            'n_combos': combo_result['n_combos'],
            'n_unique': combo_result['n_unique_signals'],
            'n_collisions': combo_result['n_collision_groups'],
            'distinguishable': combo_result['distinguishable'],
            'min_distance': combo_result['min_distance'],
            f'accuracy_noise{noise_std}': deconv['accuracy'],
        })
    
    df = pd.DataFrame(records)
    print("\n" + df.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(df['k'], df['n_combos'], color='steelblue')
    axes[0].set_xlabel('k (gRNAs per cell)')
    axes[0].set_ylabel('Number of combinations')
    axes[0].set_title('Combinatorial Scaling')
    axes[0].set_yscale('log')
    
    axes[1].bar(df['k'], df['min_distance'], color='forestgreen')
    axes[1].set_xlabel('k (gRNAs per cell)')
    axes[1].set_ylabel('Min pairwise distance')
    axes[1].set_title('Separation Between Closest Combos')
    
    axes[2].bar(df['k'], df[f'accuracy_noise{noise_std}'], color='coral')
    axes[2].set_xlabel('k (gRNAs per cell)')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title(f'Deconvolution Accuracy (noise σ={noise_std})')
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Load data
    import pandas as pd
    
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "BMDMgRNApool.csv")
    df = pd.read_csv(data_path)
    spacers = df['spacer'].head(30).str.upper().tolist()

    print(f"Loaded {len(spacers)} spacers")
    print(f"First 3: {spacers[:3]}\n")

    # Run pipeline for different k values
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Single k analysis
    for k in [1, 2, 3]:
        results = run_pipeline(spacers, k=k, prefix_len=20)
        plot_pipeline_results(results, output_prefix=os.path.join(outputs_dir, "pipeline"))
    
    # Comparison across k
    df_comparison = compare_k_values(
        spacers, 
        k_values=[1, 2, 3, 4, 5],
        prefix_len=20,
        noise_std=0.2
    )
