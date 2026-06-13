from pathlib import Path


# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================

# Use:
#   "pilot"    para o experimento inicial com 7 entidades
#   "expanded" para o experimento expandido com 50 entidades
EXPERIMENT_NAME = "expanded"


PROJECT_ROOT = Path("C:/Users/user/Desktop/CodigoDoutorado")

DATA_DIR = PROJECT_ROOT / "experiments" / "data"
RESULTS_ROOT_DIR = PROJECT_ROOT / "experiments" / "results"

DBPEDIA_PROCESSED_DIR = DATA_DIR / "dbpedia_processed"
DBPEDIA_INDEX_DIR = DBPEDIA_PROCESSED_DIR / "indexes"

REFERENCE_ROOT_DIR = DATA_DIR / "reference_ontologies"


# ============================================================
# PASTAS POR EXPERIMENTO
# ============================================================

if EXPERIMENT_NAME == "pilot":
    REFERENCE_DIR = REFERENCE_ROOT_DIR
    RESULTS_DIR = RESULTS_ROOT_DIR / "pilot"

    REFERENCE_TERMS_FILE = REFERENCE_DIR / "reference_terms.csv"
    GOLD_STANDARD_FILE = RESULTS_ROOT_DIR / "alignments" / "gold_standard.csv"

elif EXPERIMENT_NAME == "expanded":
    REFERENCE_DIR = REFERENCE_ROOT_DIR / "expanded"
    RESULTS_DIR = RESULTS_ROOT_DIR / "expanded"

    REFERENCE_TERMS_FILE = REFERENCE_DIR / "reference_terms_expanded.csv"
    GOLD_STANDARD_FILE = REFERENCE_DIR / "gold_standard_expanded.csv"

else:
    raise ValueError(
        f"EXPERIMENT_NAME inválido: {EXPERIMENT_NAME}. "
        "Use 'pilot' ou 'expanded'."
    )


ALIGNMENTS_DIR = RESULTS_DIR / "alignments"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"


# ============================================================
# ARQUIVOS DBPEDIA
# ============================================================

LABELS_CSV = DBPEDIA_PROCESSED_DIR / "labels.csv"
INSTANCE_TYPES_CSV = DBPEDIA_PROCESSED_DIR / "instance_types.csv"
LITERAL_PROPERTIES_CSV = DBPEDIA_PROCESSED_DIR / "literal_properties.csv"
OBJECT_PROPERTIES_CSV = DBPEDIA_PROCESSED_DIR / "object_properties.csv"
INFOBOX_PROPERTIES_CSV = DBPEDIA_PROCESSED_DIR / "infobox_properties.csv"

PROPERTY_FREQUENCY_INDEX = DBPEDIA_INDEX_DIR / "property_frequency_index.json"


# ============================================================
# ARQUIVOS DE SAÍDA
# ============================================================

CANDIDATE_ALIGNMENTS_FILE = ALIGNMENTS_DIR / "candidate_alignments.csv"
CANDIDATE_FEATURES_FILE = ALIGNMENTS_DIR / "candidate_features.csv"
RANKED_ALIGNMENTS_FILE = ALIGNMENTS_DIR / "ranked_alignments.csv"
BEST_ALIGNMENTS_FILE = ALIGNMENTS_DIR / "best_alignments.csv"

EVALUATION_METRICS_FILE = METRICS_DIR / "evaluation_metrics.csv"
EVALUATION_DETAILS_FILE = METRICS_DIR / "evaluation_details.csv"

ABLATION_METRICS_FILE = METRICS_DIR / "ablation_metrics.csv"
ABLATION_DETAILS_FILE = METRICS_DIR / "ablation_details.csv"
ABLATION_RANKINGS_FILE = ALIGNMENTS_DIR / "ablation_rankings.csv"


# ============================================================
# MANIFESTOS
# ============================================================

CANDIDATE_GENERATION_MANIFEST_FILE = LOGS_DIR / "candidate_generation_manifest.txt"
FEATURE_EXTRACTION_MANIFEST_FILE = LOGS_DIR / "feature_extraction_manifest.txt"
ALIGNMENT_RANKING_MANIFEST_FILE = LOGS_DIR / "alignment_ranking_manifest.txt"
EVALUATION_MANIFEST_FILE = LOGS_DIR / "evaluation_manifest.txt"
ABLATION_MANIFEST_FILE = LOGS_DIR / "ablation_manifest.txt"


def ensure_experiment_directories() -> None:
    directories = [
        RESULTS_DIR,
        ALIGNMENTS_DIR,
        METRICS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)