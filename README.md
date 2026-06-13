# Ontology Alignment with DBpedia via Linked Open Data

This repository contains the source code for the experiments described in the paper submitted for double-blind review. It implements a multi-source, feature-rich pipeline for aligning entities from a reference ontology to DBpedia resources using Linked Open Data.

---

## Overview

The pipeline takes a set of reference ontology terms (classes, properties, and individuals) and produces ranked alignment candidates drawn from DBpedia. Candidates are scored using five complementary feature groups — lexical, structural, statistical, multi-source, and type-policy — combined through a weighted scoring model. The final model weights were derived from an ablation study comparing twelve model configurations.

---

## Requirements

- Python 3.10 or higher (uses `int | None` union syntax)
- [`requests`](https://pypi.org/project/requests/) — the only third-party dependency, used for downloading DBpedia dumps

Install it with:

```bash
pip install requests
```

All other modules (`csv`, `json`, `bz2`, `re`, `math`, `pathlib`, etc.) are part of the Python standard library.

---

## Repository Structure

```
.
setup_expanded_experiment_files.py # Helper – Set up the expanded experiment directory
├── experiment_config.py               # Central configuration: paths, experiment mode, output files
├── download_dbpedia.py                # Step 1 – Download DBpedia dumps (.ttl.bz2)
├── preprocess_dbpedia.py              # Step 2 – Parse dumps into CSV files
├── build_indexes.py                   # Step 3 – Build local lookup indexes (JSON)
├── candidate_generation.py            # Step 4 – Generate alignment candidates (multi-source)
├── feature_extraction.py              # Step 5 – Extract features for each candidate pair
├── alignment_ranking.py               # Step 6 – Score and rank candidates
├── evaluation.py                      # Step 7 – Evaluate rankings against a gold standard
├── ablation.py                        # Step 8 – Ablation study across 12 model configurations

```

All data and results are written to subdirectories under `experiments/`, created automatically at runtime.

---

## Experiment Modes

Two experiment modes are supported, controlled by `EXPERIMENT_NAME` in `experiment_config.py`:

| Mode | Description |
|---|---|
| `"pilot"` | Initial experiment with 7 reference entities |
| `"expanded"` | Full experiment with 50 reference entities |

To switch modes, edit the following line in `experiment_config.py`:

```python
EXPERIMENT_NAME = "expanded"  # or "pilot"
```

---

## Pipeline Execution

Run the scripts in order from the repository root. Each step reads the outputs of the previous one.

### Step 1 — Download DBpedia dumps

```bash
python download_dbpedia.py
```

Downloads seven DBpedia dump files (version `2020.10.01`, English) to `experiments/data/dbpedia_raw/`:

- `labels` — lexical labels for all resources
- `instance-types` — `rdf:type` assertions (specific mapping)
- `mappingbased-literals` — literal property values
- `mappingbased-objects` — object property relations
- `infobox-properties` — infobox-derived properties
- `infobox-property-definitions` — infobox property descriptions
- `commons-sameas-links` — `owl:sameAs` equivalence links

Download resumes automatically on failure (up to 3 retries). A manifest is saved to `experiments/data/dbpedia_raw/download_manifest.txt`.

### Step 2 — Preprocess dumps

```bash
python preprocess_dbpedia.py
```

Parses `.ttl.bz2` dump files line by line and writes flat CSV files to `experiments/data/dbpedia_processed/`. No external RDF library is required; a lightweight regex-based parser is used.

### Step 3 — Build indexes

```bash
python build_indexes.py
```

Builds five JSON indexes from the processed CSVs in `experiments/data/dbpedia_processed/indexes/`:

| Index | Contents |
|---|---|
| `label_index.json` | `resource URI → label` |
| `normalized_label_index.json` | `normalized label → [resource URIs]` |
| `resource_type_index.json` | `resource URI → [class URIs]` |
| `class_instance_index.json` | `class URI → [resource URIs]` |
| `entity_property_index.json` | `resource URI → {literal, object, infobox properties}` |
| `property_frequency_index.json` | `property URI → global frequency count` |
| `resource_token_index.json` | `normalized URI token → [resource URIs]` |

### Step 4 — Generate candidates

```bash
python candidate_generation.py
```

For each reference term, generates up to 30 candidate URIs using five parallel strategies:

1. **Direct URI construction** — builds `dbo:Class`, `dbo:property`, `dbp:property`, and `dbr:Resource` URIs directly from the reference term label
2. **Label search** — exact and expanded label matching against `labels.csv`
3. **URI token search** — matching by the last token of the resource URI
4. **Instance type search** — finding classes used as `rdf:type` and their associated resources
5. **Property frequency search** — lexical matching against `dbo`/`dbp` properties weighted by frequency

Controlled synonym expansion is applied for selected terms (e.g., *maker* → manufacturer, producer, creator).

Output: `candidate_alignments.csv`

### Step 5 — Extract features

```bash
python feature_extraction.py
```

Computes a feature vector for each candidate pair (reference entity, candidate URI). Features include:

**Lexical features** (computed against both label and URI token):
- Levenshtein similarity
- Token Jaccard similarity
- Stemmed token Jaccard similarity
- Character bigram Jaccard similarity
- Prefix similarity
- Containment similarity
- Exact match flags

**Structural features:**
- URI namespace type (`dbo:Class`, `dbo:property`, `dbp:property`, `dbr:Resource`)
- Type compatibility between reference entity type and candidate URI space
- Property evidence count (literal, object, infobox)
- Class instance count

**Statistical features:**
- Discriminative property score
- Average property frequency
- Total property count

**Multi-source features:**
- Number of retrieval strategies that returned this candidate
- Number of distinct evidence sources
- Average evidence weight
- Initial composite score

Output: `candidate_features.csv`

### Step 6 — Rank candidates

```bash
python alignment_ranking.py
```

Scores each candidate using a weighted combination of five component scores and ranks them per reference entity. The final model weights (derived from the ablation study) are:

| Component | Weight |
|---|---|
| Lexical | 0.15 |
| Structural | 0.55 |
| Statistical | 0.05 |
| Multi-source | 0.10 |
| Type-policy | 0.15 |

Post-scoring type-policy adjustments reward candidates whose URI namespace matches the expected type of the reference entity (class → `dbo:Class`; property → `dbo/dbp:property`; individual → `dbr:Resource`).

Output: `ranked_alignments.csv`, `best_alignments.csv`

### Step 7 — Evaluate

```bash
python evaluation.py
```

Compares ranked candidates against the gold standard. Metrics reported:

- **Precision@1**, **Precision@5**, **Precision@10**
- **MRR** (Mean Reciprocal Rank)
- **Coverage** (fraction of reference entities with at least one candidate)

Input required: `gold_standard.csv` (pilot) or `gold_standard_expanded.csv` (expanded), placed in the appropriate reference ontology directory.

Output: `evaluation_metrics.csv`, `evaluation_details.csv`

### Step 8 — Ablation study

```bash
python ablation.py
```

Re-ranks all candidates under twelve model configurations to measure the contribution of each feature group:

| Model | Description |
|---|---|
| `lexical_only` | Lexical features only |
| `structural_only` | Structural features only |
| `statistical_only` | Statistical features only |
| `multisource_only` | Multi-source features only |
| `type_policy_only` | Type-policy score only |
| `lexical_structural` | Lexical + structural |
| `lexical_type_policy` | Lexical + type-policy |
| `structural_type_policy` | Structural + type-policy |
| `lexical_structural_type_policy` | Lexical + structural + type-policy |
| `lexical_structural_statistical` | Lexical + structural + statistical |
| `full_model` | All components (original weights) |
| `type_aware_full_model` | All components (adjusted weights from ablation) |

Output: `ablation_metrics.csv`, `ablation_details.csv`, `ablation_rankings.csv`

---

## Input File Format

The reference terms file (`reference_terms.csv` or `reference_terms_expanded.csv`) must contain the following columns:

| Column | Description |
|---|---|
| `entity_id` | Unique identifier for the reference entity |
| `label` | Human-readable label |
| `entity_type` | One of: `class`, `property`, `individual` |

The gold standard file (`gold_standard.csv` or `gold_standard_expanded.csv`) must contain:

| Column | Description |
|---|---|
| `reference_entity_id` | Matches `entity_id` in the reference terms file |
| `expected_candidate_uri` | Expected DBpedia URI for this entity |

Multiple rows with the same `reference_entity_id` are supported (one-to-many mappings).

---

## Expanded Experiment Setup

To set up the directory structure for the expanded experiment, run the helper script first:

```bash
python setup_expanded_experiment_files.py
```

This copies `reference_terms_expanded.csv` and `gold_standard_expanded.csv` from `experiments/src/` to the expected locations under `experiments/data/reference_ontologies/expanded/`.

---

## Output Directory Structure

After a full pipeline run with `EXPERIMENT_NAME = "expanded"`, the output tree will be:

```
experiments/
├── data/
│   ├── dbpedia_raw/               # Downloaded .ttl.bz2 dumps
│   ├── dbpedia_processed/         # Parsed CSV files and JSON indexes
│   │   └── indexes/
│   └── reference_ontologies/
│       └── expanded/              # Reference terms and gold standard
└── results/
    └── expanded/
        ├── alignments/            # candidate_alignments.csv, candidate_features.csv,
        │                          # ranked_alignments.csv, best_alignments.csv,
        │                          # ablation_rankings.csv
        ├── metrics/               # evaluation_metrics.csv, evaluation_details.csv,
        │                          # ablation_metrics.csv, ablation_details.csv
        └── logs/                  # manifest .txt files for each pipeline step
```

---

## Reproducibility

- DBpedia version is fixed to `2020.10.01` in `download_dbpedia.py`.
- All randomness is absent; the pipeline is fully deterministic.
- No external NLP libraries or pre-trained models are used; all similarity functions are implemented from scratch.
- Setting `MAX_LABEL_ROWS`, `MAX_INSTANCE_TYPE_ROWS`, or `MAX_ROWS_PER_FILE` to an integer in the respective scripts enables fast partial runs for testing.
