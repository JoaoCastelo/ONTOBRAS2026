"""
ablation_study.py

Estudo de ablação para avaliar o impacto dos grupos de features no
alinhamento entre entidades de uma ontologia de referência e entidades
da DBpedia.

Esta versão usa experiment_config.py para alternar entre:
    EXPERIMENT_NAME = "pilot"
    EXPERIMENT_NAME = "expanded"

Entrada:
    candidate_features.csv
    gold_standard.csv ou gold_standard_expanded.csv

Saída:
    ablation_metrics.csv
    ablation_details.csv
    ablation_rankings.csv
    ablation_manifest.txt

Modelos avaliados:
    lexical_only
    structural_only
    statistical_only
    multisource_only
    type_policy_only
    lexical_structural
    lexical_type_policy
    structural_type_policy
    lexical_structural_type_policy
    lexical_structural_statistical
    full_model
    type_aware_full_model

Autor: João Castelo
Projeto: Validação Fuzzy-LODAlign com DBpedia
"""

from datetime import datetime
import csv
import math
import re
import sys

from experiment_config import (
    EXPERIMENT_NAME,
    CANDIDATE_FEATURES_FILE,
    GOLD_STANDARD_FILE,
    ABLATION_METRICS_FILE,
    ABLATION_DETAILS_FILE,
    ABLATION_RANKINGS_FILE,
    ABLATION_MANIFEST_FILE,
    ensure_experiment_directories,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_ENCODING = "utf-8"

TOP_K_VALUES = [1, 5, 10]
TOP_K_RANKING = 30


# ============================================================
# MODELOS DE ABLAÇÃO
# ============================================================

ABLATION_MODELS = {
    "lexical_only": {
        "lexical_score": 1.00,
        "structural_score": 0.00,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    "structural_only": {
        "lexical_score": 0.00,
        "structural_score": 1.00,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    "statistical_only": {
        "lexical_score": 0.00,
        "structural_score": 0.00,
        "statistical_score": 1.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    "multisource_only": {
        "lexical_score": 0.00,
        "structural_score": 0.00,
        "statistical_score": 0.00,
        "multisource_score": 1.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    "type_policy_only": {
        "lexical_score": 0.00,
        "structural_score": 0.00,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 1.00,
        "use_post_policy": True,
    },

    "lexical_structural": {
        "lexical_score": 0.58,
        "structural_score": 0.42,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    "lexical_type_policy": {
        "lexical_score": 0.75,
        "structural_score": 0.00,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.25,
        "use_post_policy": True,
    },

    "structural_type_policy": {
        "lexical_score": 0.00,
        "structural_score": 0.75,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.25,
        "use_post_policy": True,
    },

    "lexical_structural_type_policy": {
        "lexical_score": 0.50,
        "structural_score": 0.35,
        "statistical_score": 0.00,
        "multisource_score": 0.00,
        "type_policy_score": 0.15,
        "use_post_policy": True,
    },

    "lexical_structural_statistical": {
        "lexical_score": 0.48,
        "structural_score": 0.36,
        "statistical_score": 0.16,
        "multisource_score": 0.00,
        "type_policy_score": 0.00,
        "use_post_policy": False,
    },

    # Modelo completo original.
    # Mantido para comparação com a versão ajustada.
    "full_model": {
        "lexical_score": 0.42,
        "structural_score": 0.34,
        "statistical_score": 0.09,
        "multisource_score": 0.10,
        "type_policy_score": 0.05,
        "use_post_policy": True,
    },

    # Novo modelo completo ajustado a partir do estudo de ablação expandido.
    #
    # A ablação mostrou que structural_only e type_policy_only obtiveram
    # desempenho superior ao full_model original. Assim, este modelo reduz
    # a dominância lexical e prioriza compatibilidade estrutural/tipológica.
    "type_aware_full_model": {
        "lexical_score": 0.15,
        "structural_score": 0.55,
        "statistical_score": 0.05,
        "multisource_score": 0.10,
        "type_policy_score": 0.15,
        "use_post_policy": True,
    },
}


# ============================================================
# UTILITÁRIOS
# ============================================================

def ensure_directories() -> None:
    ensure_experiment_directories()
    print(f"[OK] Experimento ativo: {EXPERIMENT_NAME}")
    print(f"[OK] Features input: {CANDIDATE_FEATURES_FILE}")
    print(f"[OK] Gold standard: {GOLD_STANDARD_FILE}")
    print(f"[OK] Metrics output: {ABLATION_METRICS_FILE}")
    print(f"[OK] Details output: {ABLATION_DETAILS_FILE}")
    print(f"[OK] Rankings output: {ABLATION_RANKINGS_FILE}")
    print(f"[OK] Log: {ABLATION_MANIFEST_FILE}")


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def uri_last_token(uri: str) -> str:
    if not uri:
        return ""

    uri = uri.rstrip("/#")
    return re.split(r"[/#]", uri)[-1]


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default

        return int(float(value))

    except (ValueError, TypeError):
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def safe_log_normalization(value: float, denominator: float = 10.0) -> float:
    if value <= 0:
        return 0.0

    return clamp(math.log(value + 1.0) / denominator)


def is_dbo_class(uri: str) -> bool:
    if not uri.startswith("http://dbpedia.org/ontology/"):
        return False

    token = uri_last_token(uri)
    return bool(token and token[0].isupper())


def is_dbo_property(uri: str) -> bool:
    if not uri.startswith("http://dbpedia.org/ontology/"):
        return False

    token = uri_last_token(uri)
    return bool(token and token[0].islower())


def is_dbp_property(uri: str) -> bool:
    return uri.startswith("http://dbpedia.org/property/")


def is_dbr_resource(uri: str) -> bool:
    return uri.startswith("http://dbpedia.org/resource/")


def is_property_uri(uri: str) -> bool:
    return is_dbo_property(uri) or is_dbp_property(uri)


def normalize_reference_type(reference_type: str) -> str:
    reference_type = normalize_text(reference_type)

    if reference_type in {"class", "concept"}:
        return "class"

    if reference_type in {
        "property",
        "objectproperty",
        "dataproperty",
        "object property",
        "data property",
        "annotation property",
    }:
        return "property"

    if reference_type in {"individual", "instance", "resource"}:
        return "individual"

    return reference_type or "unknown"


def read_csv(path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    rows = []

    with open(path, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def write_csv(rows: list[dict], path) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para salvar em {path}")

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding=CSV_ENCODING, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[OK] Arquivo salvo: {path}")


# ============================================================
# GOLD STANDARD
# ============================================================

def load_gold_standard() -> dict:
    rows = read_csv(GOLD_STANDARD_FILE)

    required_columns = {
        "reference_entity_id",
        "expected_candidate_uri",
    }

    if rows:
        available_columns = set(rows[0].keys())

        if not required_columns.issubset(available_columns):
            raise ValueError(
                f"O gold standard precisa conter as colunas: {required_columns}. "
                f"Colunas encontradas: {available_columns}"
            )

    gold = {}

    for row in rows:
        reference_id = row.get("reference_entity_id", "").strip()
        expected_uri = row.get("expected_candidate_uri", "").strip()

        if not reference_id or not expected_uri:
            continue

        gold.setdefault(reference_id, set()).add(expected_uri)

    print(f"[OK] Entidades no gold standard: {len(gold)}")
    print(f"[OK] Alinhamentos esperados no gold standard: {sum(len(v) for v in gold.values())}")

    return gold


# ============================================================
# EVIDÊNCIAS
# ============================================================

def has_real_dbpedia_evidence(row: dict) -> bool:
    evidence_sources = row.get("evidence_sources", "")

    real_sources = [
        "dbpedia_labels",
        "dbpedia_instance_types_classes",
        "dbpedia_instance_types_resources",
        "dbpedia_properties",
        "dbpedia_property_frequency",
    ]

    return any(source in evidence_sources for source in real_sources)


def has_controlled_direct_evidence(row: dict) -> bool:
    evidence_sources = row.get("evidence_sources", "")
    matched_terms = normalize_text(row.get("matched_terms", ""))
    candidate_token = normalize_text(uri_last_token(row.get("candidate_uri", "")))

    if "controlled_direct_property_candidate" in evidence_sources:
        return True

    controlled_tokens = {
        "manufacturer",
        "producer",
        "maker",
        "creator",
    }

    if candidate_token in controlled_tokens and any(
        term in matched_terms
        for term in ["maker", "manufacturer", "producer", "creator"]
    ):
        return True

    return False


def is_weak_constructed_candidate(row: dict) -> bool:
    strategies = row.get("strategies", "")
    evidence_sources = row.get("evidence_sources", "")

    if "dbpedia_uri_construction" not in strategies:
        return False

    if has_real_dbpedia_evidence(row):
        return False

    if has_controlled_direct_evidence(row):
        return False

    total_property_count = to_float(row.get("total_property_count"))
    average_property_frequency = to_float(row.get("average_property_frequency"))
    class_instance_count = to_float(row.get("class_instance_count"))

    if total_property_count > 0 or average_property_frequency > 0 or class_instance_count > 0:
        return False

    weak_sources = [
        "dbpedia_ontology_class_construction",
        "dbpedia_ontology_property_construction",
        "dbpedia_property_construction",
        "dbpedia_resource_construction",
    ]

    return any(source in evidence_sources for source in weak_sources)


# ============================================================
# SCORES
# ============================================================

def compute_lexical_score(row: dict) -> float:
    levenshtein_label = to_float(row.get("levenshtein_label_similarity"))
    token_jaccard = to_float(row.get("token_jaccard_similarity"))
    stemmed_jaccard = to_float(row.get("stemmed_jaccard_similarity"))
    char_bigram_jaccard = to_float(row.get("char_bigram_jaccard_similarity"))
    prefix = to_float(row.get("prefix_similarity"))

    uri_levenshtein = to_float(row.get("uri_levenshtein_similarity"))
    uri_token_jaccard = to_float(row.get("uri_token_jaccard_similarity"))
    uri_stemmed_jaccard = to_float(row.get("uri_stemmed_jaccard_similarity"))
    uri_char_bigram_jaccard = to_float(row.get("uri_char_bigram_jaccard_similarity"))
    uri_prefix = to_float(row.get("uri_prefix_similarity"))

    containment = to_float(row.get("containment_similarity"))
    exact_label = to_float(row.get("exact_label_match"))
    exact_uri = to_float(row.get("exact_uri_token_match"))

    label_score = (
        0.24 * levenshtein_label +
        0.18 * token_jaccard +
        0.14 * stemmed_jaccard +
        0.14 * char_bigram_jaccard +
        0.10 * prefix +
        0.12 * containment +
        0.08 * exact_label
    )

    uri_score = (
        0.28 * uri_levenshtein +
        0.20 * uri_token_jaccard +
        0.14 * uri_stemmed_jaccard +
        0.14 * uri_char_bigram_jaccard +
        0.10 * uri_prefix +
        0.14 * exact_uri
    )

    lexical_score = 0.72 * label_score + 0.28 * uri_score

    if exact_label == 1.0 or exact_uri == 1.0:
        lexical_score = min(lexical_score + 0.06, 1.0)

    if is_weak_constructed_candidate(row):
        lexical_score *= 0.82

    return round(clamp(lexical_score), 6)


def compute_type_policy_score(row: dict) -> float:
    reference_type = normalize_reference_type(row.get("reference_entity_type", ""))
    candidate_uri = row.get("candidate_uri", "")

    if reference_type == "class":
        if is_dbo_class(candidate_uri):
            return 1.0

        if is_dbr_resource(candidate_uri):
            return 0.35

        if is_property_uri(candidate_uri):
            return 0.0

    if reference_type == "property":
        if is_dbo_property(candidate_uri):
            return 1.0

        if is_dbp_property(candidate_uri):
            return 0.90

        if is_dbo_class(candidate_uri):
            return 0.0

        if is_dbr_resource(candidate_uri):
            return 0.15

    if reference_type == "individual":
        if is_dbr_resource(candidate_uri):
            return 1.0

        if is_dbo_class(candidate_uri):
            return 0.35

        if is_property_uri(candidate_uri):
            return 0.0

    return 0.0


def compute_structural_score(row: dict) -> float:
    reference_type = normalize_reference_type(row.get("reference_entity_type", ""))

    has_type = to_float(row.get("has_dbpedia_type"))
    dbpedia_type_count = to_float(row.get("dbpedia_type_count"))

    total_property_count = to_float(row.get("total_property_count"))
    class_instance_count = to_float(row.get("class_instance_count"))

    type_policy_score = compute_type_policy_score(row)

    type_evidence = clamp(
        0.60 * has_type +
        0.40 * safe_log_normalization(dbpedia_type_count, denominator=5.0)
    )

    property_evidence = safe_log_normalization(total_property_count, denominator=8.0)
    class_evidence = safe_log_normalization(class_instance_count, denominator=12.0)

    if reference_type == "class":
        structural_score = (
            0.62 * type_policy_score +
            0.13 * type_evidence +
            0.10 * property_evidence +
            0.15 * class_evidence
        )

    elif reference_type == "property":
        structural_score = (
            0.70 * type_policy_score +
            0.20 * property_evidence +
            0.10 * type_evidence
        )

    elif reference_type == "individual":
        structural_score = (
            0.62 * type_policy_score +
            0.23 * type_evidence +
            0.15 * property_evidence
        )

    else:
        structural_score = (
            0.50 * type_policy_score +
            0.20 * type_evidence +
            0.15 * property_evidence +
            0.15 * class_evidence
        )

    if has_controlled_direct_evidence(row):
        structural_score = min(structural_score + 0.12, 1.0)

    if is_weak_constructed_candidate(row):
        structural_score *= 0.78

    return round(clamp(structural_score), 6)


def compute_statistical_score(row: dict) -> float:
    discriminative_score = to_float(row.get("discriminative_property_score"))
    avg_frequency = to_float(row.get("average_property_frequency"))
    total_property_count = to_float(row.get("total_property_count"))

    if avg_frequency <= 0:
        frequency_balance = 0.0
    else:
        frequency_balance = 1.0 / math.log(avg_frequency + 2.0)

    property_presence = safe_log_normalization(total_property_count, denominator=8.0)

    statistical_score = (
        0.55 * clamp(discriminative_score) +
        0.25 * clamp(frequency_balance) +
        0.20 * property_presence
    )

    if has_controlled_direct_evidence(row):
        statistical_score = min(statistical_score + 0.08, 1.0)

    if is_weak_constructed_candidate(row):
        statistical_score *= 0.70

    return round(clamp(statistical_score), 6)


def compute_multisource_score(row: dict) -> float:
    strategy_count = to_float(row.get("strategy_count"))
    evidence_source_count = to_float(row.get("evidence_source_count"))
    average_evidence_weight = to_float(row.get("average_evidence_weight"))
    initial_score = to_float(row.get("initial_score_numeric"))

    strategy_component = clamp(strategy_count / 5.0)
    source_component = clamp(evidence_source_count / 5.0)

    multisource_score = (
        0.28 * strategy_component +
        0.32 * source_component +
        0.20 * clamp(average_evidence_weight) +
        0.20 * clamp(initial_score)
    )

    if has_real_dbpedia_evidence(row):
        multisource_score = min(multisource_score + 0.08, 1.0)

    if has_controlled_direct_evidence(row):
        multisource_score = min(multisource_score + 0.10, 1.0)

    if is_weak_constructed_candidate(row):
        multisource_score *= 0.68

    return round(clamp(multisource_score), 6)


def add_component_scores(row: dict) -> dict:
    row = row.copy()

    row["lexical_score"] = compute_lexical_score(row)
    row["structural_score"] = compute_structural_score(row)
    row["statistical_score"] = compute_statistical_score(row)
    row["multisource_score"] = compute_multisource_score(row)
    row["type_policy_score"] = compute_type_policy_score(row)

    row["has_real_dbpedia_evidence"] = int(has_real_dbpedia_evidence(row))
    row["has_controlled_direct_evidence"] = int(has_controlled_direct_evidence(row))
    row["is_weak_constructed_candidate"] = int(is_weak_constructed_candidate(row))

    return row


# ============================================================
# SCORE DO MODELO
# ============================================================

def apply_post_policy_adjustments(row: dict, score: float) -> float:
    candidate_uri = row.get("candidate_uri", "")
    reference_type = normalize_reference_type(row.get("reference_entity_type", ""))

    if reference_type == "class" and is_dbo_class(candidate_uri):
        score += 0.08

    if reference_type == "class" and is_dbr_resource(candidate_uri):
        score -= 0.05

    if reference_type == "property" and is_property_uri(candidate_uri):
        score += 0.08

    if reference_type == "individual" and is_dbr_resource(candidate_uri):
        score += 0.08

    if has_controlled_direct_evidence(row):
        score += 0.08

    if is_weak_constructed_candidate(row):
        score -= 0.15

    return score


def compute_model_score(row: dict, model_config: dict) -> float:
    score = (
        model_config["lexical_score"] * to_float(row.get("lexical_score")) +
        model_config["structural_score"] * to_float(row.get("structural_score")) +
        model_config["statistical_score"] * to_float(row.get("statistical_score")) +
        model_config["multisource_score"] * to_float(row.get("multisource_score")) +
        model_config["type_policy_score"] * to_float(row.get("type_policy_score"))
    )

    if model_config.get("use_post_policy", False):
        score = apply_post_policy_adjustments(row, score)

    return round(clamp(score), 6)


# ============================================================
# RANQUEAMENTO
# ============================================================

def rank_for_model(rows: list[dict], model_name: str, model_config: dict) -> list[dict]:
    model_rows = []

    for row in rows:
        scored_row = row.copy()
        scored_row["ablation_model"] = model_name
        scored_row["ablation_score"] = compute_model_score(scored_row, model_config)
        model_rows.append(scored_row)

    grouped = {}

    for row in model_rows:
        reference_id = row.get("reference_entity_id", "")
        grouped.setdefault(reference_id, []).append(row)

    ranked_rows = []

    for reference_id, group_rows in grouped.items():
        group_rows.sort(
            key=lambda item: (
                to_float(item.get("ablation_score")),
                to_float(item.get("type_policy_score")),
                to_float(item.get("structural_score")),
                to_float(item.get("lexical_score")),
                to_int(item.get("has_controlled_direct_evidence")),
                to_int(item.get("has_real_dbpedia_evidence")),
                -to_int(item.get("is_weak_constructed_candidate")),
            ),
            reverse=True,
        )

        for rank, item in enumerate(group_rows, start=1):
            item["ablation_rank"] = rank
            ranked_rows.append(item)

    ranked_rows.sort(
        key=lambda item: (
            item.get("ablation_model", ""),
            item.get("reference_entity_id", ""),
            to_int(item.get("ablation_rank")),
        )
    )

    return [
        row for row in ranked_rows
        if to_int(row.get("ablation_rank")) <= TOP_K_RANKING
    ]


# ============================================================
# MÉTRICAS
# ============================================================

def hit_at_k(ranked_candidates: list[dict], expected_uris: set[str], k: int) -> int:
    for candidate in ranked_candidates[:k]:
        candidate_uri = candidate.get("candidate_uri", "").strip()

        if candidate_uri in expected_uris:
            return 1

    return 0


def reciprocal_rank(ranked_candidates: list[dict], expected_uris: set[str]) -> float:
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate_uri = candidate.get("candidate_uri", "").strip()

        if candidate_uri in expected_uris:
            return 1.0 / index

    return 0.0


def rank_of_first_hit(ranked_candidates: list[dict], expected_uris: set[str]) -> int:
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate_uri = candidate.get("candidate_uri", "").strip()

        if candidate_uri in expected_uris:
            return index

    return 0


def evaluate_model(model_name: str, ranked_rows: list[dict], gold: dict) -> tuple[list[dict], list[dict]]:
    grouped = {}

    for row in ranked_rows:
        reference_id = row.get("reference_entity_id", "")
        grouped.setdefault(reference_id, []).append(row)

    for reference_id, rows in grouped.items():
        rows.sort(
            key=lambda row: to_int(row.get("ablation_rank")),
            reverse=False,
        )

    total_references = len(gold)

    if total_references == 0:
        raise ValueError("Gold standard vazio.")

    hit_sums = {k: 0 for k in TOP_K_VALUES}
    reciprocal_rank_sum = 0.0
    missing_predictions = 0

    details = []

    for reference_id, expected_uris in gold.items():
        ranked_candidates = grouped.get(reference_id, [])

        if not ranked_candidates:
            missing_predictions += 1

        rr = reciprocal_rank(ranked_candidates, expected_uris)
        reciprocal_rank_sum += rr

        first_hit_rank = rank_of_first_hit(ranked_candidates, expected_uris)

        detail_row = {
            "experiment_name": EXPERIMENT_NAME,
            "ablation_model": model_name,
            "reference_entity_id": reference_id,
            "expected_candidate_uris": "|".join(sorted(expected_uris)),
            "num_candidates": len(ranked_candidates),
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": round(rr, 6),
        }

        for k in TOP_K_VALUES:
            hit = hit_at_k(ranked_candidates, expected_uris, k)
            hit_sums[k] += hit
            detail_row[f"hit_at_{k}"] = hit

        if ranked_candidates:
            best_candidate = ranked_candidates[0]
            detail_row["top1_candidate_uri"] = best_candidate.get("candidate_uri", "")
            detail_row["top1_candidate_label"] = best_candidate.get("candidate_label", "")
            detail_row["top1_score"] = best_candidate.get("ablation_score", "")
        else:
            detail_row["top1_candidate_uri"] = ""
            detail_row["top1_candidate_label"] = ""
            detail_row["top1_score"] = ""

        details.append(detail_row)

    metrics = []

    for k in TOP_K_VALUES:
        metrics.append(
            {
                "experiment_name": EXPERIMENT_NAME,
                "ablation_model": model_name,
                "metric": f"Precision@{k}",
                "value": round(hit_sums[k] / total_references, 6),
            }
        )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "ablation_model": model_name,
            "metric": "MRR",
            "value": round(reciprocal_rank_sum / total_references, 6),
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "ablation_model": model_name,
            "metric": "Coverage",
            "value": round((total_references - missing_predictions) / total_references, 6),
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "ablation_model": model_name,
            "metric": "Evaluated references",
            "value": total_references,
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "ablation_model": model_name,
            "metric": "Missing predictions",
            "value": missing_predictions,
        }
    )

    return metrics, details


# ============================================================
# EXECUÇÃO DA ABLAÇÃO
# ============================================================

def run_ablation_study(feature_rows: list[dict], gold: dict) -> tuple[list[dict], list[dict], list[dict]]:
    print("\n" + "=" * 80)
    print("[INFO] Calculando scores de componentes")
    print("=" * 80)

    rows_with_components = []

    for index, row in enumerate(feature_rows, start=1):
        rows_with_components.append(add_component_scores(row))

        if index % 1000 == 0:
            print(f"[PROGRESSO] {index:,} candidatos processados")

    all_metrics = []
    all_details = []
    all_rankings = []

    print("\n" + "=" * 80)
    print("[INFO] Executando modelos de ablação")
    print("=" * 80)

    for model_name, model_config in ABLATION_MODELS.items():
        print(f"[MODELO] {model_name}")

        ranked_rows = rank_for_model(
            rows=rows_with_components,
            model_name=model_name,
            model_config=model_config,
        )

        metrics, details = evaluate_model(
            model_name=model_name,
            ranked_rows=ranked_rows,
            gold=gold,
        )

        all_metrics.extend(metrics)
        all_details.extend(details)
        all_rankings.extend(ranked_rows)

    return all_metrics, all_details, all_rankings


# ============================================================
# MANIFESTO
# ============================================================

def save_manifest(
    feature_rows: list[dict],
    gold: dict,
    metrics: list[dict],
    details: list[dict],
    rankings: list[dict],
) -> None:
    with open(ABLATION_MANIFEST_FILE, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Ablation Study Manifest\n")
        file.write("===============================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        file.write(f"Input features file: {CANDIDATE_FEATURES_FILE}\n")
        file.write(f"Gold standard file: {GOLD_STANDARD_FILE}\n")
        file.write(f"Output metrics file: {ABLATION_METRICS_FILE}\n")
        file.write(f"Output details file: {ABLATION_DETAILS_FILE}\n")
        file.write(f"Output rankings file: {ABLATION_RANKINGS_FILE}\n\n")

        file.write(f"Feature rows: {len(feature_rows)}\n")
        file.write(f"Gold references: {len(gold)}\n")
        file.write(f"Gold expected mappings: {sum(len(v) for v in gold.values())}\n")
        file.write(f"Metric rows: {len(metrics)}\n")
        file.write(f"Detail rows: {len(details)}\n")
        file.write(f"Ranking rows: {len(rankings)}\n\n")

        file.write("Ablation models:\n")
        for model_name, config in ABLATION_MODELS.items():
            file.write(f"  {model_name}:\n")
            for component, value in config.items():
                file.write(f"    {component}: {value}\n")

        file.write("\nRanking policy:\n")
        file.write("  class -> prefer dbo:Class\n")
        file.write("  property -> prefer dbo:property/dbp:property\n")
        file.write("  individual -> prefer dbr:Resource\n")
        file.write("  weak constructed candidates are penalized\n")
        file.write("  controlled direct candidates are rewarded\n")
        file.write("\nInterpretation:\n")
        file.write("  type_aware_full_model is the adjusted full model derived from the expanded ablation study.\n")
        file.write("  It assigns higher weight to structural and type-policy evidence while preserving lexical,\n")
        file.write("  statistical, and multi-source signals as complementary information.\n")

    print(f"[OK] Manifesto salvo: {ABLATION_MANIFEST_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DBpedia Ablation Study - Experiment-aware Type-aware Version")
    print("=" * 80)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Input features: {CANDIDATE_FEATURES_FILE}")
    print(f"Gold standard: {GOLD_STANDARD_FILE}")
    print(f"Output metrics: {ABLATION_METRICS_FILE}")
    print("=" * 80)

    try:
        ensure_directories()

        feature_rows = read_csv(CANDIDATE_FEATURES_FILE)
        print(f"[OK] Linhas de features carregadas: {len(feature_rows):,}")

        gold = load_gold_standard()

        metrics, details, rankings = run_ablation_study(
            feature_rows=feature_rows,
            gold=gold,
        )

        write_csv(metrics, ABLATION_METRICS_FILE)
        write_csv(details, ABLATION_DETAILS_FILE)
        write_csv(rankings, ABLATION_RANKINGS_FILE)

        save_manifest(
            feature_rows=feature_rows,
            gold=gold,
            metrics=metrics,
            details=details,
            rankings=rankings,
        )

        print("\n" + "=" * 80)
        print("RESUMO DO ESTUDO DE ABLAÇÃO")
        print("=" * 80)

        for model_name in ABLATION_MODELS.keys():
            model_metrics = [
                row for row in metrics
                if row["ablation_model"] == model_name
            ]

            metric_text = " | ".join(
                f"{row['metric']}={row['value']}"
                for row in model_metrics
                if row["metric"] in {
                    "Precision@1",
                    "Precision@5",
                    "Precision@10",
                    "MRR",
                    "Coverage",
                    "Missing predictions",
                }
            )

            print(f"{model_name}: {metric_text}")

        print("\n[OK] Estudo de ablação concluído com sucesso.")

    except Exception as error:
        print(f"\n[ERRO] Falha no estudo de ablação: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()