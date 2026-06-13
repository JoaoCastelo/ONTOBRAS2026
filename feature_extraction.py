"""
feature_extraction.py

Extração de features para candidatos de alinhamento entre entidades de uma
ontologia de referência e entidades da DBpedia.

Esta versão usa experiment_config.py para alternar entre:
    EXPERIMENT_NAME = "pilot"
    EXPERIMENT_NAME = "expanded"

Entrada:
    candidate_alignments.csv

Saída:
    candidate_features.csv
    feature_extraction_manifest.txt
"""

from datetime import datetime
import csv
import json
import math
import re
import sys

from experiment_config import (
    EXPERIMENT_NAME,
    CANDIDATE_ALIGNMENTS_FILE,
    CANDIDATE_FEATURES_FILE,
    FEATURE_EXTRACTION_MANIFEST_FILE,
    INSTANCE_TYPES_CSV,
    LITERAL_PROPERTIES_CSV,
    OBJECT_PROPERTIES_CSV,
    INFOBOX_PROPERTIES_CSV,
    PROPERTY_FREQUENCY_INDEX,
    ensure_experiment_directories,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_ENCODING = "utf-8"
JSON_ENCODING = "utf-8"

MAX_INSTANCE_TYPE_ROWS = None
MAX_LITERAL_PROPERTY_ROWS = None
MAX_OBJECT_PROPERTY_ROWS = None
MAX_INFOBOX_PROPERTY_ROWS = None


# ============================================================
# UTILITÁRIOS
# ============================================================

def ensure_directories() -> None:
    ensure_experiment_directories()
    print(f"[OK] Experimento ativo: {EXPERIMENT_NAME}")
    print(f"[OK] Entrada: {CANDIDATE_ALIGNMENTS_FILE}")
    print(f"[OK] Saída: {CANDIDATE_FEATURES_FILE}")
    print(f"[OK] Log: {FEATURE_EXTRACTION_MANIFEST_FILE}")


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


def split_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


def uri_last_token(uri: str) -> str:
    if not uri:
        return ""

    uri = uri.rstrip("/#")
    return re.split(r"[/#]", uri)[-1]


def uri_namespace(uri: str) -> str:
    if not uri:
        return "unknown"

    if "/resource/" in uri:
        return "resource"

    if "/ontology/" in uri:
        return "ontology"

    if "/property/" in uri:
        return "property"

    return "other"


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def safe_log(value: float) -> float:
    if value <= 0:
        return 0.0

    return math.log(value + 1.0)


def simple_stem(token: str) -> str:
    token = normalize_text(token)

    if not token:
        return ""

    suffixes = [
        "ization", "ational", "fulness", "ousness", "iveness",
        "tional", "biliti", "lessli", "entli", "ation",
        "alism", "aliti", "ousli", "iviti", "fulli",
        "enci", "anci", "abli", "izer", "ator",
        "icate", "ative", "alize", "iciti", "ical",
        "ness", "ments", "ment", "ingly", "edly",
        "ing", "ed", "ies", "es", "s"
    ]

    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return token[:-3] + "y"

            if suffix == "es" and token.endswith("ses"):
                return token[:-2]

            return token[: -len(suffix)]

    return token


# ============================================================
# SIMILARIDADES LEXICAIS
# ============================================================

def levenshtein_distance(a: str, b: str) -> int:
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (char_a != char_b)

            current_row.append(min(insertion, deletion, substitution))

        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm and not b_norm:
        return 1.0

    if not a_norm or not b_norm:
        return 0.0

    distance = levenshtein_distance(a_norm, b_norm)
    max_len = max(len(a_norm), len(b_norm))

    if max_len == 0:
        return 0.0

    return 1.0 - (distance / max_len)


def token_jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(split_tokens(a))
    tokens_b = set(split_tokens(b))

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a.union(tokens_b)

    if not union:
        return 0.0

    return len(tokens_a.intersection(tokens_b)) / len(union)


def stemmed_jaccard_similarity(a: str, b: str) -> float:
    tokens_a = {simple_stem(token) for token in split_tokens(a)}
    tokens_b = {simple_stem(token) for token in split_tokens(b)}

    tokens_a = {token for token in tokens_a if token}
    tokens_b = {token for token in tokens_b if token}

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a.union(tokens_b)

    if not union:
        return 0.0

    return len(tokens_a.intersection(tokens_b)) / len(union)


def char_bigrams(text: str) -> set[str]:
    text = normalize_text(text).replace(" ", "")

    if len(text) < 2:
        return {text} if text else set()

    return {text[index:index + 2] for index in range(len(text) - 1)}


def char_bigram_jaccard_similarity(a: str, b: str) -> float:
    bigrams_a = char_bigrams(a)
    bigrams_b = char_bigrams(b)

    if not bigrams_a or not bigrams_b:
        return 0.0

    union = bigrams_a.union(bigrams_b)

    if not union:
        return 0.0

    return len(bigrams_a.intersection(bigrams_b)) / len(union)


def prefix_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    min_len = min(len(a_norm), len(b_norm))

    if min_len == 0:
        return 0.0

    common = 0

    for index in range(min_len):
        if a_norm[index] == b_norm[index]:
            common += 1
        else:
            break

    return common / min_len


def containment_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    if a_norm in b_norm or b_norm in a_norm:
        return min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))

    return 0.0


# ============================================================
# TIPO DE URI
# ============================================================

def is_dbpedia_ontology_class(uri: str) -> int:
    if not uri.startswith("http://dbpedia.org/ontology/"):
        return 0

    token = uri_last_token(uri)

    if token and token[0].isupper():
        return 1

    return 0


def is_dbpedia_property(uri: str) -> int:
    if uri.startswith("http://dbpedia.org/property/"):
        return 1

    if uri.startswith("http://dbpedia.org/ontology/"):
        token = uri_last_token(uri)

        if token and token[0].islower():
            return 1

    return 0


def is_dbpedia_resource(uri: str) -> int:
    if uri.startswith("http://dbpedia.org/resource/"):
        return 1

    return 0


def candidate_namespace_feature(uri: str) -> float:
    namespace = uri_namespace(uri)

    if namespace == "ontology":
        return 1.0

    if namespace == "property":
        return 0.8

    if namespace == "resource":
        return 0.6

    return 0.0


# ============================================================
# LEITURA DOS CANDIDATOS
# ============================================================

def load_candidate_alignments() -> list[dict]:
    if not CANDIDATE_ALIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de candidatos não encontrado: {CANDIDATE_ALIGNMENTS_FILE}"
        )

    rows = []

    with open(CANDIDATE_ALIGNMENTS_FILE, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    print(f"[OK] Candidatos carregados: {len(rows):,}")
    return rows


def collect_candidate_uris(candidates: list[dict]) -> set[str]:
    uris = set()

    for row in candidates:
        uri = row.get("candidate_uri", "").strip()

        if uri:
            uris.add(uri)

    print(f"[OK] URIs candidatas únicas: {len(uris):,}")
    return uris


# ============================================================
# ÍNDICES LEVES POR STREAMING
# ============================================================

def load_property_frequency_index() -> dict:
    if not PROPERTY_FREQUENCY_INDEX.exists():
        print(f"[AVISO] Índice de frequência de propriedades não encontrado: {PROPERTY_FREQUENCY_INDEX}")
        return {}

    with open(PROPERTY_FREQUENCY_INDEX, "r", encoding=JSON_ENCODING) as file:
        index = json.load(file)

    print(f"[OK] property_frequency_index carregado: {len(index):,} propriedades")
    return index


def collect_instance_type_evidence(candidate_uris: set[str]) -> tuple[dict, dict]:
    """
    Retorna:
        resource_type_index:
            resource_uri -> set(class_uri)

        class_instance_count:
            class_uri -> número de instâncias observadas
    """
    resource_type_index = {}
    class_instance_count = {}

    if not INSTANCE_TYPES_CSV.exists():
        print(f"[AVISO] Arquivo não encontrado: {INSTANCE_TYPES_CSV}")
        return resource_type_index, class_instance_count

    with open(INSTANCE_TYPES_CSV, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            resource = row.get("resource", "").strip()
            class_uri = row.get("class", "").strip()

            if not resource or not class_uri:
                continue

            if resource in candidate_uris:
                resource_type_index.setdefault(resource, set()).add(class_uri)

            if class_uri in candidate_uris:
                class_instance_count[class_uri] = class_instance_count.get(class_uri, 0) + 1

            if row_number % 500000 == 0:
                print(f"[PROGRESSO] instance_types.csv: {row_number:,} linhas lidas")

            if MAX_INSTANCE_TYPE_ROWS is not None and row_number >= MAX_INSTANCE_TYPE_ROWS:
                break

    print(f"[OK] Recursos candidatos com tipos: {len(resource_type_index):,}")
    print(f"[OK] Classes candidatas com instâncias: {len(class_instance_count):,}")

    return resource_type_index, class_instance_count


def collect_property_evidence_from_file(
    file_path,
    candidate_uris: set[str],
    source_name: str,
    max_rows=None,
) -> dict:
    """
    Conta propriedades associadas a cada candidato quando ele aparece como subject.
    Espera CSV com colunas:
        resource, property, value/object
    """
    evidence = {}

    if not file_path.exists():
        print(f"[AVISO] Arquivo não encontrado: {file_path}")
        return evidence

    with open(file_path, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            resource = row.get("resource", "").strip()
            property_uri = row.get("property", "").strip()

            if not resource or not property_uri:
                continue

            if resource in candidate_uris:
                evidence.setdefault(resource, set()).add(property_uri)

            if row_number % 500000 == 0:
                print(f"[PROGRESSO] {source_name}: {row_number:,} linhas lidas")

            if max_rows is not None and row_number >= max_rows:
                break

    print(f"[OK] {source_name}: candidatos com propriedades = {len(evidence):,}")
    return evidence


def merge_property_sets(*dicts: dict) -> dict:
    merged = {}

    for current_dict in dicts:
        for uri, values in current_dict.items():
            merged.setdefault(uri, set()).update(values)

    return merged


# ============================================================
# FEATURES
# ============================================================

def compute_property_statistics(
    candidate_uri: str,
    literal_properties: dict,
    object_properties: dict,
    infobox_properties: dict,
    property_frequency_index: dict,
) -> dict:
    literal_set = literal_properties.get(candidate_uri, set())
    object_set = object_properties.get(candidate_uri, set())
    infobox_set = infobox_properties.get(candidate_uri, set())

    all_properties = set()
    all_properties.update(literal_set)
    all_properties.update(object_set)
    all_properties.update(infobox_set)

    frequencies = []

    for property_uri in all_properties:
        freq = to_float(property_frequency_index.get(property_uri, 0.0))
        frequencies.append(freq)

    if frequencies:
        average_frequency = sum(frequencies) / len(frequencies)
    else:
        average_frequency = 0.0

    if average_frequency > 0:
        discriminative_property_score = 1.0 / math.log(average_frequency + 2.0)
    else:
        discriminative_property_score = 0.0

    return {
        "literal_property_count": len(literal_set),
        "object_property_count": len(object_set),
        "infobox_property_count": len(infobox_set),
        "total_property_count": len(all_properties),
        "log_total_property_count": round(safe_log(len(all_properties)), 6),
        "average_property_frequency": round(average_frequency, 6),
        "log_average_property_frequency": round(safe_log(average_frequency), 6),
        "discriminative_property_score": round(clamp(discriminative_property_score), 6),
    }


def compute_features_for_candidate(
    row: dict,
    resource_type_index: dict,
    class_instance_count: dict,
    literal_properties: dict,
    object_properties: dict,
    infobox_properties: dict,
    property_frequency_index: dict,
) -> dict:
    reference_label = row.get("reference_label", "")
    candidate_label = row.get("candidate_label", "")
    candidate_uri = row.get("candidate_uri", "")
    candidate_uri_token = row.get("candidate_uri_token", "") or uri_last_token(candidate_uri)

    output = row.copy()

    # Lexical features using candidate label.
    output["levenshtein_label_similarity"] = round(
        levenshtein_similarity(reference_label, candidate_label), 6
    )
    output["token_jaccard_similarity"] = round(
        token_jaccard_similarity(reference_label, candidate_label), 6
    )
    output["stemmed_jaccard_similarity"] = round(
        stemmed_jaccard_similarity(reference_label, candidate_label), 6
    )
    output["char_bigram_jaccard_similarity"] = round(
        char_bigram_jaccard_similarity(reference_label, candidate_label), 6
    )
    output["prefix_similarity"] = round(
        prefix_similarity(reference_label, candidate_label), 6
    )

    # Lexical features using URI token.
    output["uri_levenshtein_similarity"] = round(
        levenshtein_similarity(reference_label, candidate_uri_token), 6
    )
    output["uri_token_jaccard_similarity"] = round(
        token_jaccard_similarity(reference_label, candidate_uri_token), 6
    )
    output["uri_stemmed_jaccard_similarity"] = round(
        stemmed_jaccard_similarity(reference_label, candidate_uri_token), 6
    )
    output["uri_char_bigram_jaccard_similarity"] = round(
        char_bigram_jaccard_similarity(reference_label, candidate_uri_token), 6
    )
    output["uri_prefix_similarity"] = round(
        prefix_similarity(reference_label, candidate_uri_token), 6
    )

    output["containment_similarity"] = round(
        max(
            containment_similarity(reference_label, candidate_label),
            containment_similarity(reference_label, candidate_uri_token),
        ),
        6,
    )

    output["exact_label_match"] = int(
        normalize_text(reference_label) == normalize_text(candidate_label)
    )
    output["exact_uri_token_match"] = int(
        normalize_text(reference_label) == normalize_text(candidate_uri_token)
    )

    # Namespace and type features.
    output["candidate_namespace_feature"] = candidate_namespace_feature(candidate_uri)
    output["is_dbpedia_ontology_class"] = is_dbpedia_ontology_class(candidate_uri)
    output["is_dbpedia_resource"] = is_dbpedia_resource(candidate_uri)
    output["is_dbpedia_property"] = is_dbpedia_property(candidate_uri)

    types = resource_type_index.get(candidate_uri, set())
    output["has_dbpedia_type"] = int(len(types) > 0)
    output["dbpedia_type_count"] = len(types)

    property_stats = compute_property_statistics(
        candidate_uri=candidate_uri,
        literal_properties=literal_properties,
        object_properties=object_properties,
        infobox_properties=infobox_properties,
        property_frequency_index=property_frequency_index,
    )

    output.update(property_stats)

    instances = class_instance_count.get(candidate_uri, 0)
    output["class_instance_count"] = instances
    output["class_instance_log_score"] = round(safe_log(instances), 6)

    output["initial_score_numeric"] = to_float(row.get("initial_score"))

    return output


def extract_features(candidates: list[dict]) -> list[dict]:
    candidate_uris = collect_candidate_uris(candidates)

    print("\n" + "=" * 80)
    print("[INFO] Carregando evidências da DBpedia em modo streaming")
    print("=" * 80)

    property_frequency_index = load_property_frequency_index()

    resource_type_index, class_instance_count = collect_instance_type_evidence(
        candidate_uris=candidate_uris,
    )

    literal_properties = collect_property_evidence_from_file(
        file_path=LITERAL_PROPERTIES_CSV,
        candidate_uris=candidate_uris,
        source_name="literal_properties.csv",
        max_rows=MAX_LITERAL_PROPERTY_ROWS,
    )

    object_properties = collect_property_evidence_from_file(
        file_path=OBJECT_PROPERTIES_CSV,
        candidate_uris=candidate_uris,
        source_name="object_properties.csv",
        max_rows=MAX_OBJECT_PROPERTY_ROWS,
    )

    infobox_properties = collect_property_evidence_from_file(
        file_path=INFOBOX_PROPERTIES_CSV,
        candidate_uris=candidate_uris,
        source_name="infobox_properties.csv",
        max_rows=MAX_INFOBOX_PROPERTY_ROWS,
    )

    print("\n" + "=" * 80)
    print("[INFO] Calculando features")
    print("=" * 80)

    feature_rows = []

    for index, row in enumerate(candidates, start=1):
        feature_rows.append(
            compute_features_for_candidate(
                row=row,
                resource_type_index=resource_type_index,
                class_instance_count=class_instance_count,
                literal_properties=literal_properties,
                object_properties=object_properties,
                infobox_properties=infobox_properties,
                property_frequency_index=property_frequency_index,
            )
        )

        if index % 1000 == 0:
            print(f"[PROGRESSO] {index:,} candidatos processados")

    print(f"[OK] Features calculadas para {len(feature_rows):,} candidatos")
    return feature_rows


# ============================================================
# SAÍDA
# ============================================================

def save_features(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Nenhuma feature para salvar.")

    fieldnames = list(rows[0].keys())

    with open(CANDIDATE_FEATURES_FILE, "w", encoding=CSV_ENCODING, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[OK] Features salvas em: {CANDIDATE_FEATURES_FILE}")


def save_manifest(candidates: list[dict], features: list[dict]) -> None:
    with open(FEATURE_EXTRACTION_MANIFEST_FILE, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Feature Extraction Manifest\n")
        file.write("===================================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        file.write(f"Input candidates file: {CANDIDATE_ALIGNMENTS_FILE}\n")
        file.write(f"Output features file: {CANDIDATE_FEATURES_FILE}\n\n")

        file.write(f"Candidates loaded: {len(candidates)}\n")
        file.write(f"Feature rows generated: {len(features)}\n\n")

        file.write("Streaming sources:\n")
        file.write(f"  instance types: {INSTANCE_TYPES_CSV}\n")
        file.write(f"  literal properties: {LITERAL_PROPERTIES_CSV}\n")
        file.write(f"  object properties: {OBJECT_PROPERTIES_CSV}\n")
        file.write(f"  infobox properties: {INFOBOX_PROPERTIES_CSV}\n")
        file.write(f"  property frequency index: {PROPERTY_FREQUENCY_INDEX}\n\n")

        file.write("Limits:\n")
        file.write(f"  MAX_INSTANCE_TYPE_ROWS: {MAX_INSTANCE_TYPE_ROWS}\n")
        file.write(f"  MAX_LITERAL_PROPERTY_ROWS: {MAX_LITERAL_PROPERTY_ROWS}\n")
        file.write(f"  MAX_OBJECT_PROPERTY_ROWS: {MAX_OBJECT_PROPERTY_ROWS}\n")
        file.write(f"  MAX_INFOBOX_PROPERTY_ROWS: {MAX_INFOBOX_PROPERTY_ROWS}\n")

    print(f"[OK] Manifesto salvo em: {FEATURE_EXTRACTION_MANIFEST_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DBpedia Feature Extractor - Experiment-aware Version")
    print("=" * 80)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Input candidates: {CANDIDATE_ALIGNMENTS_FILE}")
    print(f"Output features: {CANDIDATE_FEATURES_FILE}")
    print("=" * 80)

    try:
        ensure_directories()

        candidates = load_candidate_alignments()
        features = extract_features(candidates)

        save_features(features)
        save_manifest(candidates, features)

        print("\n" + "=" * 80)
        print("RESUMO")
        print("=" * 80)
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Candidatos: {len(candidates):,}")
        print(f"Linhas de features: {len(features):,}")
        print(f"Arquivo de saída: {CANDIDATE_FEATURES_FILE}")
        print("\n[OK] Extração de features concluída com sucesso.")

    except Exception as error:
        print(f"\n[ERRO] Falha na extração de features: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()