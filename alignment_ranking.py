"""
alignment_ranking.py

Ranqueamento final dos candidatos de alinhamento gerados a partir da DBpedia.

Esta versão usa experiment_config.py para alternar entre:
    EXPERIMENT_NAME = "pilot"
    EXPERIMENT_NAME = "expanded"

Entrada:
    candidate_features.csv

Saída:
    ranked_alignments.csv
    best_alignments.csv
    alignment_ranking_manifest.txt

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
    RANKED_ALIGNMENTS_FILE,
    BEST_ALIGNMENTS_FILE,
    ALIGNMENT_RANKING_MANIFEST_FILE,
    ensure_experiment_directories,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_ENCODING = "utf-8"

TOP_K = 30


# Pesos ajustados após o estudo de ablação expandido.
#
# A ablação mostrou que structural_only e type_policy_only obtiveram
# desempenho superior ao full_model lexicalmente balanceado.
#
# Portanto, o modelo final passa a priorizar evidências estruturais e
# compatibilidade tipológica entre entidade de referência e espaço URI
# da DBpedia:
#
#     class      -> dbo:Class
#     property   -> dbo/dbp:property
#     individual -> dbr:Resource
#
# A similaridade lexical continua sendo usada, mas com menor peso,
# principalmente para ordenar candidatos dentro do mesmo tipo semântico.
WEIGHTS = {
    "lexical_score": 0.15,
    "structural_score": 0.55,
    "statistical_score": 0.05,
    "multisource_score": 0.10,
    "type_policy_score": 0.15,
}

# ============================================================
# UTILITÁRIOS
# ============================================================

def ensure_directories() -> None:
    ensure_experiment_directories()
    print(f"[OK] Experimento ativo: {EXPERIMENT_NAME}")
    print(f"[OK] Entrada: {CANDIDATE_FEATURES_FILE}")
    print(f"[OK] Saída ranked: {RANKED_ALIGNMENTS_FILE}")
    print(f"[OK] Saída best: {BEST_ALIGNMENTS_FILE}")
    print(f"[OK] Log: {ALIGNMENT_RANKING_MANIFEST_FILE}")


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
# DETECÇÃO DE EVIDÊNCIA
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
# SCORES PARCIAIS
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


# ============================================================
# SCORE FINAL
# ============================================================

def compute_final_score(row: dict) -> dict:
    row = row.copy()

    lexical_score = compute_lexical_score(row)
    structural_score = compute_structural_score(row)
    statistical_score = compute_statistical_score(row)
    multisource_score = compute_multisource_score(row)
    type_policy_score = compute_type_policy_score(row)

    final_score = (
        WEIGHTS["lexical_score"] * lexical_score +
        WEIGHTS["structural_score"] * structural_score +
        WEIGHTS["statistical_score"] * statistical_score +
        WEIGHTS["multisource_score"] * multisource_score +
        WEIGHTS["type_policy_score"] * type_policy_score
    )

    candidate_uri = row.get("candidate_uri", "")
    reference_type = normalize_reference_type(row.get("reference_entity_type", ""))

    if reference_type == "class" and is_dbo_class(candidate_uri):
        final_score += 0.08

    if reference_type == "class" and is_dbr_resource(candidate_uri):
        final_score -= 0.05

    if reference_type == "property" and is_property_uri(candidate_uri):
        final_score += 0.08

    if reference_type == "individual" and is_dbr_resource(candidate_uri):
        final_score += 0.08

    if has_controlled_direct_evidence(row):
        final_score += 0.08

    if is_weak_constructed_candidate(row):
        final_score -= 0.15

    row["lexical_score"] = lexical_score
    row["structural_score"] = structural_score
    row["statistical_score"] = statistical_score
    row["multisource_score"] = multisource_score
    row["type_policy_score"] = round(type_policy_score, 6)
    row["final_alignment_score"] = round(clamp(final_score), 6)

    row["has_real_dbpedia_evidence"] = int(has_real_dbpedia_evidence(row))
    row["has_controlled_direct_evidence"] = int(has_controlled_direct_evidence(row))
    row["is_weak_constructed_candidate"] = int(is_weak_constructed_candidate(row))

    return row


def rank_candidates(rows: list[dict]) -> list[dict]:
    scored_rows = []

    for index, row in enumerate(rows, start=1):
        scored_rows.append(compute_final_score(row))

        if index % 1000 == 0:
            print(f"[PROGRESSO] {index:,} candidatos ranqueados")

    grouped = {}

    for row in scored_rows:
        reference_id = row.get("reference_entity_id", "")
        grouped.setdefault(reference_id, []).append(row)

    ranked_rows = []

    for reference_id, group_rows in grouped.items():
        group_rows.sort(
            key=lambda item: (
                to_float(item.get("final_alignment_score")),
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
            item["rank"] = rank
            ranked_rows.append(item)

    ranked_rows.sort(
        key=lambda item: (
            item.get("reference_entity_id", ""),
            to_int(item.get("rank")),
        )
    )

    return ranked_rows


def select_best_alignments(ranked_rows: list[dict]) -> list[dict]:
    best_rows = []

    for row in ranked_rows:
        if to_int(row.get("rank")) == 1:
            best_rows.append(row)

    best_rows.sort(
        key=lambda item: item.get("reference_entity_id", "")
    )

    return best_rows


# ============================================================
# MANIFESTO
# ============================================================

def save_manifest(input_rows: list[dict], ranked_rows: list[dict], best_rows: list[dict]) -> None:
    with open(ALIGNMENT_RANKING_MANIFEST_FILE, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Alignment Ranking Manifest\n")
        file.write("==================================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        file.write(f"Input features file: {CANDIDATE_FEATURES_FILE}\n")
        file.write(f"Output ranked file: {RANKED_ALIGNMENTS_FILE}\n")
        file.write(f"Output best file: {BEST_ALIGNMENTS_FILE}\n\n")

        file.write(f"Input candidates: {len(input_rows)}\n")
        file.write(f"Ranked candidates: {len(ranked_rows)}\n")
        file.write(f"Best alignments: {len(best_rows)}\n\n")

        file.write("Weights:\n")
        for key, value in WEIGHTS.items():
            file.write(f"  {key}: {value}\n")

        file.write("\nRanking policy:\n")
        file.write("  class -> prefer dbo:Class\n")
        file.write("  property -> prefer dbo:property/dbp:property\n")
        file.write("  individual -> prefer dbr:Resource\n")
        file.write("  weak constructed candidates without real DBpedia evidence are penalized\n")
        file.write("  controlled direct candidates are rewarded\n")

    print(f"[OK] Manifesto salvo: {ALIGNMENT_RANKING_MANIFEST_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DBpedia Alignment Ranking - Experiment-aware Type-aware Version")
    print("=" * 80)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Input: {CANDIDATE_FEATURES_FILE}")
    print(f"Ranked output: {RANKED_ALIGNMENTS_FILE}")
    print(f"Best output: {BEST_ALIGNMENTS_FILE}")
    print("=" * 80)

    try:
        ensure_directories()

        input_rows = read_csv(CANDIDATE_FEATURES_FILE)
        print(f"[OK] Candidatos com features carregados: {len(input_rows):,}")

        ranked_rows = rank_candidates(input_rows)

        ranked_rows_top_k = [
            row for row in ranked_rows
            if to_int(row.get("rank")) <= TOP_K
        ]

        best_rows = select_best_alignments(ranked_rows_top_k)

        write_csv(ranked_rows_top_k, RANKED_ALIGNMENTS_FILE)
        write_csv(best_rows, BEST_ALIGNMENTS_FILE)

        save_manifest(
            input_rows=input_rows,
            ranked_rows=ranked_rows_top_k,
            best_rows=best_rows,
        )

        print("\n" + "=" * 80)
        print("RESUMO")
        print("=" * 80)
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Candidatos de entrada: {len(input_rows):,}")
        print(f"Candidatos ranqueados top-{TOP_K}: {len(ranked_rows_top_k):,}")
        print(f"Melhores alinhamentos: {len(best_rows):,}")
        print(f"Arquivo best: {BEST_ALIGNMENTS_FILE}")
        print("\n[OK] Ranqueamento concluído com sucesso.")

    except Exception as error:
        print(f"\n[ERRO] Falha no ranqueamento: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()