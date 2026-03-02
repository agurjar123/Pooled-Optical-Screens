# Optical Pooled Screening — gRNA Barcode Deconvolution

Computational framework for evaluating whether CRISPR guide RNA (gRNA) barcodes can be reliably deconvolved from their summed 4-color in situ sequencing (ISS) signals in optical pooled screens.

---

## Background

Optical pooled screening (OPS) is a high-throughput method that couples genetic perturbations (CRISPR) with single-cell imaging readouts. Each cell receives one or more gRNAs, and the identity of those gRNAs is read out optically using in situ sequencing of short barcode sequences embedded in the gRNA construct.

A key challenge arises when multiple gRNAs are present in the same cell: the optical signals from each barcode **add together**, making it ambiguous which combination of gRNAs produced the observed signal. This project asks:

> Given the summed 4-color ISS signal from *k* co-infected gRNAs, can we uniquely recover which *k* gRNAs are present?

---

## Research Questions

1. **Individual distinguishability** — Are all gRNA barcodes in the library optically distinct from one another?
2. **Combination distinguishability** — For k-way gRNA combinations (k = 1, 2, 3, ...), does each unique combination produce a unique summed signal, or do collisions occur?
3. **Prefix length** — How many nucleotides of the barcode (10 nt vs 20 nt) are needed to eliminate collisions?
4. **Noise robustness** — How does deconvolution accuracy degrade as imaging noise increases?

---

## Methodology

### 4-Color ISS Encoding

Each nucleotide position is encoded as a one-hot vector over 4 channels (A, C, G, T):

```
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]
```

A barcode of length L is represented as an L × 4 matrix. When k gRNAs co-infect a cell, the observed signal is the **element-wise sum** of their encodings — an L × 4 matrix with integer entries in [0, k].

### Three-Level Analysis Pipeline

**Level 1 — Individual gRNAs**
Computes pairwise Euclidean distances between all individual gRNA encodings. Identifies the most similar pairs and flags any identical sequences (perfect collisions).

**Level 2 — k-way Combinations**
Enumerates all k-combinations (with replacement) of gRNAs, computes their sum signals, and detects collision groups — sets of distinct combinations that produce identical signals. Reports the fraction of combinations that are unambiguously deconvolvable.

**Level 3 — Deconvolution Under Noise**
Simulates realistic imaging conditions by adding Gaussian noise and variable gRNA abundance. Tests nearest-neighbor classification accuracy (top-1 and top-3) across a range of noise levels.

---

## Repository Structure

```
.
├── README.md
├── .gitignore
│
├── src/
│   ├── grna_simulation.py     # Core analysis engine (encoding, collision detection, deconvolution)
│   └── dataset_parser.py      # Utility to browse gRNA library data from the IDR FTP server
│
├── notebooks/
│   ├── barcode_deconvolution.ipynb   # Main analysis: collision analysis + imaging matrix visualization
│   └── exploratory_analysis.ipynb   # Exploratory: k-scaling, ambiguity fraction, sanity checks
│
├── data/
│   ├── BMDMgRNApool.csv       # BMDM macrophage gRNA library (~400 guides, 20 bp spacers)
│   └── ops_cloning.csv        # DNA damage response gRNA set (~52 guides, for validation)
│
└── outputs/
    ├── collision_analysis.png         # Unique patterns & collision groups vs barcode length
    ├── imaging_matrix_10nt.png        # Riskiest gRNA pairs at 10 nt prefix
    ├── imaging_matrix_20nt.png        # Riskiest gRNA pairs at 20 nt prefix
    ├── most_similar_grnas.png         # Top most-similar individual gRNA pairs
    ├── most_similar_nonself_pairs.png # Closest non-identical pairs
    └── most_similar_within_pair.png   # Within-pair similarity analysis
```

---

## Data

### `data/BMDMgRNApool.csv`
Primary dataset. BMDM (bone marrow-derived macrophage) CRISPR gRNA library with columns:
- `gene_id` — Ensembl mouse gene ID
- `gene_symbol` — gene name
- `spacer` — 20 bp gRNA spacer sequence (used for barcode analysis)
- `opsBarcode` — 8 bp OPS barcode (first 8 nt of spacer)
- `type` — `target` or control

### `data/ops_cloning.csv`
Secondary/validation dataset. Focused on DNA damage response genes (ATM, RAD51, TERF1, PML, etc.) with full sgRNA construct sequences including handles. Used to validate deconvolution at small library scales.

---

## Usage

### Run the full pipeline (command line)

```bash
python src/grna_simulation.py
```

This loads `data/BMDMgRNApool.csv`, runs the three-level analysis for k = 1, 2, 3, and saves comparison plots to `outputs/`.

### Interactive analysis (Jupyter)

```bash
jupyter notebook notebooks/barcode_deconvolution.ipynb
```

The main notebook walks through:
1. Loading spacer sequences
2. Building the 4-color encoding
3. Collision detection across barcode lengths (2–20 nt)
4. Noiseless deconvolution accuracy
5. Imaging matrix visualization for the highest-risk pairs

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
requests
beautifulsoup4
```

Install with:

```bash
pip install numpy pandas matplotlib scipy requests beautifulsoup4
```

---

## References

- Feldman, D. et al. (2019). Optical Pooled Screens in Human Cells. *Cell*, 179(3), 787–799.
- Feldman, D. et al. (2022). Optical Pooled Screens: A Framework for Pooled Imaging Screens in Human Cells. *Nature Protocols*, 17, 476–512.
- Kudo, T. et al. PerturbView — large-scale perturbation screen data. Image Data Resource (IDR), `idr0162`.
