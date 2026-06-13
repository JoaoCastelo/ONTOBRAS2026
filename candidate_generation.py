"""
candidate_generation.py

Geração multifonte de candidatos para alinhamento entre entidades de uma
ontologia de referência e entidades da DBpedia.

Esta versão trabalha em modo streaming, evitando carregar grandes índices
JSON em memória.

Estratégias principais:
    1. Construção direta de URIs DBpedia:
        - dbo:Class
        - dbo:property
        - dbp:property
        - dbr:Resource

    2. Busca em labels.csv:
        - label exato
        - label expandido
        - token da URI

    3. Busca em instance_types.csv:
        - classes usadas como rdf:type
        - recursos associados a tipos

    4. Busca em property_frequency_index.json:
        - propriedades dbo/dbp por similaridade lexical

    5. Expansão lexical controlada:
        - hasMaker -> maker, manufacturer, producer, creator
        - region -> area, location, place
        - country -> nation, state

Autor: João Castelo
Projeto: Validação Fuzzy-LODAlign com DBpedia
"""

from pathlib import Path
from datetime import datetime
import csv
import json
import math
import re
import sys
from experiment_config import (
    EXPERIMENT_NAME,
    DBPEDIA_PROCESSED_DIR,
    DBPEDIA_INDEX_DIR,
    REFERENCE_TERMS_FILE,
    LABELS_CSV,
    INSTANCE_TYPES_CSV,
    PROPERTY_FREQUENCY_INDEX,
    CANDIDATE_ALIGNMENTS_FILE,
    CANDIDATE_GENERATION_MANIFEST_FILE,
    ensure_experiment_directories,
)

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PROCESSED_DIR = DBPEDIA_PROCESSED_DIR
INDEX_DIR = DBPEDIA_INDEX_DIR

OUTPUT_FILE = CANDIDATE_ALIGNMENTS_FILE
MANIFEST_FILE = CANDIDATE_GENERATION_MANIFEST_FILE
CSV_ENCODING = "utf-8"
JSON_ENCODING = "utf-8"

TOP_K = 30

# Limite máximo por fonte. Ajuda a evitar crescimento excessivo em casos muito genéricos.
MAX_CANDIDATES_PER_SOURCE = 10000

# Para teste rápido, use um número inteiro.
# Para execução completa, deixe como None.
MAX_LABEL_ROWS = None
MAX_INSTANCE_TYPE_ROWS = None


# ============================================================
# SINÔNIMOS CONTROLADOS
# ============================================================

CONTROLLED_SYNONYMS = {
    "maker": [
        "maker",
        "manufacturer",
        "producer",
        "creator",
        "made by",
        "produced by",
        "manufactured by",
    ],
    "has maker": [
        "maker",
        "manufacturer",
        "producer",
        "creator",
    ],
    "manufacturer": [
        "maker",
        "manufacturer",
        "producer",
        "creator",
    ],
    "producer": [
        "maker",
        "manufacturer",
        "producer",
        "creator",
    ],
    "creator": [
        "maker",
        "manufacturer",
        "producer",
        "creator",
        "author",
    ],
    "author": [
        "author",
        "writer",
        "creator",
    ],
    "region": [
        "region",
        "area",
        "location",
        "place",
    ],
    "country": [
        "country",
        "nation",
        "state",
    ],
    "food": [
        "food",
        "dish",
        "cuisine",
        "ingredient",
    ],
    "wine": [
        "wine",
        "wine region",
        "alcoholic beverage",
        "beverage",
    ],
    "camera": [
        "camera",
        "photographic camera",
        "device",
    ],
}


# Candidatos especiais úteis quando o termo expandido é amplo.
# Isso evita que propriedades importantes fiquem fora do conjunto de candidatos.
CONTROLLED_DIRECT_CANDIDATES = {
    "maker": [
        ("http://dbpedia.org/ontology/manufacturer", "manufacturer", "property", 1.15),
        ("http://dbpedia.org/ontology/producer", "producer", "property", 1.10),
        ("http://dbpedia.org/ontology/creator", "creator", "property", 1.05),
        ("http://dbpedia.org/property/maker", "maker", "property", 1.05),
        ("http://dbpedia.org/property/manufacturer", "manufacturer", "property", 1.00),
        ("http://dbpedia.org/property/producer", "producer", "property", 1.00),
    ],
    "has maker": [
        ("http://dbpedia.org/ontology/manufacturer", "manufacturer", "property", 1.15),
        ("http://dbpedia.org/ontology/producer", "producer", "property", 1.10),
        ("http://dbpedia.org/property/maker", "maker", "property", 1.05),
    ],
    "manufacturer": [
        ("http://dbpedia.org/ontology/manufacturer", "manufacturer", "property", 1.15),
        ("http://dbpedia.org/property/manufacturer", "manufacturer", "property", 1.00),
    ],
    "producer": [
        ("http://dbpedia.org/ontology/producer", "producer", "property", 1.10),
        ("http://dbpedia.org/property/producer", "producer", "property", 1.00),
    ],
    "region": [
        ("http://dbpedia.org/ontology/region", "region", "property", 1.10),
        ("http://dbpedia.org/property/region", "region", "property", 1.05),
    ],
    "country": [
        ("http://dbpedia.org/ontology/country", "country", "property", 1.10),
        ("http://dbpedia.org/property/country", "country", "property", 1.05),
    ],
}


# ============================================================
# TERMOS DE TESTE
# ============================================================

DEFAULT_REFERENCE_TERMS = [
    {
        "entity_id": "ref:Wine",
        "entity_type": "class",
        "label": "Wine",
    },
    {
        "entity_id": "ref:Food",
        "entity_type": "class",
        "label": "Food",
    },
    {
        "entity_id": "ref:Camera",
        "entity_type": "class",
        "label": "Camera",
    },
    {
        "entity_id": "ref:Chardonnay",
        "entity_type": "individual",
        "label": "Chardonnay",
    },
    {
        "entity_id": "ref:Region",
        "entity_type": "property",
        "label": "region",
    },
    {
        "entity_id": "ref:Country",
        "entity_type": "property",
        "label": "country",
    },
    {
        "entity_id": "ref:hasMaker",
        "entity_type": "property",
        "label": "hasMaker",
    },
]


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def ensure_directories() -> None:
    ensure_experiment_directories()
    print(f"[OK] Experimento ativo: {EXPERIMENT_NAME}")
    print(f"[OK] Pasta de alinhamentos verificada: {OUTPUT_FILE.parent}")
    print(f"[OK] Pasta de logs verificada: {MANIFEST_FILE.parent}")

def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparação lexical.

    Exemplos:
        hasMaker -> has maker
        wine_region -> wine region
        Cabernet-Sauvignon -> cabernet sauvignon
    """
    if text is None:
        return ""

    text = str(text).strip()

    # Separa camelCase.
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


def is_probably_class_uri(uri: str) -> bool:
    if not uri.startswith("http://dbpedia.org/ontology/"):
        return False

    token = uri_last_token(uri)

    return bool(token and token[0].isupper())


def is_probably_property_uri(uri: str) -> bool:
    if uri.startswith("http://dbpedia.org/property/"):
        return True

    if uri.startswith("http://dbpedia.org/ontology/"):
        token = uri_last_token(uri)

        if token and token[0].islower():
            return True

    return False


def is_probably_resource_uri(uri: str) -> bool:
    return uri.startswith("http://dbpedia.org/resource/")


def normalize_entity_type(entity_type: str) -> str:
    entity_type = normalize_text(entity_type)

    if entity_type in {"class", "concept"}:
        return "class"

    if entity_type in {
        "property",
        "objectproperty",
        "dataproperty",
        "object property",
        "data property",
        "annotation property",
    }:
        return "property"

    if entity_type in {"individual", "instance", "resource"}:
        return "individual"

    return entity_type or "unknown"


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
# SIMILARIDADES
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

    return {text[i:i + 2] for i in range(len(text) - 1)}


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


def combined_initial_score(reference_label: str, candidate_label: str, candidate_uri: str) -> float:
    """
    Score lexical inicial para ordenar candidatos.

    O ranking final é feito posteriormente em alignment_ranking.py.
    """
    candidate_token = uri_last_token(candidate_uri)

    lev_label = levenshtein_similarity(reference_label, candidate_label)
    jac_label = token_jaccard_similarity(reference_label, candidate_label)
    stem_label = stemmed_jaccard_similarity(reference_label, candidate_label)
    bigram_label = char_bigram_jaccard_similarity(reference_label, candidate_label)
    prefix_label = prefix_similarity(reference_label, candidate_label)

    lev_uri = levenshtein_similarity(reference_label, candidate_token)
    jac_uri = token_jaccard_similarity(reference_label, candidate_token)

    score = (
        0.25 * lev_label +
        0.20 * jac_label +
        0.15 * stem_label +
        0.15 * bigram_label +
        0.10 * prefix_label +
        0.10 * lev_uri +
        0.05 * jac_uri
    )

    return round(score, 6)


# ============================================================
# EXPANSÃO DOS TERMOS
# ============================================================

def remove_property_prefixes(text: str) -> list[str]:
    normalized = normalize_text(text)

    variants = [normalized]

    prefixes = [
        "has ",
        "is ",
        "was ",
        "were ",
        "contains ",
        "contain ",
        "uses ",
        "use ",
        "made of ",
        "part of ",
        "produced by ",
        "manufactured by ",
        "created by ",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            stripped = normalized[len(prefix):].strip()

            if stripped:
                variants.append(stripped)

    return variants


def add_controlled_synonyms(variants: list[str]) -> list[str]:
    seen = set(variants)
    output = list(variants)

    for variant in list(variants):
        tokens = split_tokens(variant)
        candidate_keys = [variant] + tokens

        for key in candidate_keys:
            key = normalize_text(key)

            if key in CONTROLLED_SYNONYMS:
                for synonym in CONTROLLED_SYNONYMS[key]:
                    normalized_synonym = normalize_text(synonym)

                    if normalized_synonym and normalized_synonym not in seen:
                        output.append(normalized_synonym)
                        seen.add(normalized_synonym)

    return output


def generate_term_variants(label: str) -> list[str]:
    normalized = normalize_text(label)

    variants = []
    seen = set()

    def add_variant(value: str) -> None:
        value = normalize_text(value)

        if value and value not in seen:
            variants.append(value)
            seen.add(value)

    add_variant(normalized)

    for item in remove_property_prefixes(label):
        add_variant(item)

    tokens = split_tokens(label)

    for token in tokens:
        if len(token) > 2:
            add_variant(token)

    stemmed_tokens = [simple_stem(token) for token in tokens if len(token) > 2]
    stemmed_tokens = [token for token in stemmed_tokens if token]

    for token in stemmed_tokens:
        add_variant(token)

    if stemmed_tokens:
        add_variant(" ".join(stemmed_tokens))

    stopwords = {
        "a", "an", "the", "of", "in", "on", "for", "to", "by", "with",
        "has", "is", "was", "were", "and", "or"
    }

    content_tokens = [token for token in tokens if token not in stopwords]

    if content_tokens:
        add_variant(" ".join(content_tokens))

        for token in content_tokens:
            add_variant(token)

    variants = add_controlled_synonyms(variants)

    final_variants = []
    final_seen = set()

    for variant in variants:
        variant = normalize_text(variant)

        if variant and variant not in final_seen:
            final_variants.append(variant)
            final_seen.add(variant)

    return final_variants[:50]


# ============================================================
# CONSTRUÇÃO DIRETA DE URIS DBPEDIA
# ============================================================

def to_camel_case(text: str) -> str:
    tokens = split_tokens(text)

    if not tokens:
        return ""

    return "".join(token.capitalize() for token in tokens)


def to_lower_camel_case(text: str) -> str:
    tokens = split_tokens(text)

    if not tokens:
        return ""

    first = tokens[0].lower()
    rest = "".join(token.capitalize() for token in tokens[1:])

    return first + rest


def dbpedia_ontology_class_uri(term: str) -> str:
    camel = to_camel_case(term)

    if not camel:
        return ""

    return f"http://dbpedia.org/ontology/{camel}"


def dbpedia_ontology_property_uri(term: str) -> str:
    lower_camel = to_lower_camel_case(term)

    if not lower_camel:
        return ""

    return f"http://dbpedia.org/ontology/{lower_camel}"


def dbpedia_property_uri(term: str) -> str:
    normalized = normalize_text(term).replace(" ", "_")

    if not normalized:
        return ""

    return f"http://dbpedia.org/property/{normalized}"


def dbpedia_resource_uri(term: str) -> str:
    camel = to_camel_case(term)

    if not camel:
        return ""

    return f"http://dbpedia.org/resource/{camel}"


def generate_constructed_dbpedia_uris(term: dict) -> list[dict]:
    """
    Gera candidatos diretamente por construção de URI.

    Essa fonte evita que candidatos como dbo:Wine ou dbo:manufacturer
    fiquem fora da lista apenas porque não apareceram em labels.csv.
    """
    candidates = []
    seen = set()

    reference_type = term["entity_type"]

    for expanded_term in term["expanded_terms"]:

        candidate_specs = []

        if reference_type == "class":
            candidate_specs.append(
                {
                    "uri": dbpedia_ontology_class_uri(expanded_term),
                    "label": to_camel_case(expanded_term),
                    "weight": 1.12,
                    "source": "dbpedia_ontology_class_construction",
                }
            )

        elif reference_type == "property":
            candidate_specs.append(
                {
                    "uri": dbpedia_ontology_property_uri(expanded_term),
                    "label": to_lower_camel_case(expanded_term),
                    "weight": 1.12,
                    "source": "dbpedia_ontology_property_construction",
                }
            )

            candidate_specs.append(
                {
                    "uri": dbpedia_property_uri(expanded_term),
                    "label": expanded_term,
                    "weight": 1.04,
                    "source": "dbpedia_property_construction",
                }
            )

            # Candidatos controlados adicionais.
            if expanded_term in CONTROLLED_DIRECT_CANDIDATES:
                for uri, label, candidate_type, weight in CONTROLLED_DIRECT_CANDIDATES[expanded_term]:
                    if candidate_type == "property":
                        candidate_specs.append(
                            {
                                "uri": uri,
                                "label": label,
                                "weight": weight,
                                "source": "controlled_direct_property_candidate",
                            }
                        )

        elif reference_type == "individual":
            candidate_specs.append(
                {
                    "uri": dbpedia_resource_uri(expanded_term),
                    "label": to_camel_case(expanded_term),
                    "weight": 1.10,
                    "source": "dbpedia_resource_construction",
                }
            )

        for spec in candidate_specs:
            candidate_uri = spec["uri"]
            candidate_label = spec["label"]
            evidence_weight = spec["weight"]
            evidence_source = spec["source"]

            if not candidate_uri:
                continue

            key = (term["entity_id"], candidate_uri)

            if key in seen:
                continue

            seen.add(key)

            candidates.append(
                make_candidate(
                    term=term,
                    candidate_uri=candidate_uri,
                    candidate_label=candidate_label,
                    strategy="dbpedia_uri_construction",
                    evidence_source=evidence_source,
                    matched_term=expanded_term,
                    evidence_weight=evidence_weight,
                )
            )

    return candidates


# ============================================================
# TERMOS DE REFERÊNCIA
# ============================================================

def create_example_reference_terms_file() -> None:
    """
    Cria arquivo de referência apenas no experimento piloto,
    caso ele ainda não exista.

    No experimento expandido, o arquivo deve existir previamente.
    """
    if REFERENCE_TERMS_FILE.exists():
        return

    if EXPERIMENT_NAME == "expanded":
        raise FileNotFoundError(
            f"Arquivo de termos expandidos não encontrado: {REFERENCE_TERMS_FILE}. "
            "Execute primeiro o setup_expanded_experiment_files.py."
        )

    with open(REFERENCE_TERMS_FILE, "w", encoding=CSV_ENCODING, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["entity_id", "entity_type", "label"]
        )
        writer.writeheader()

        for item in DEFAULT_REFERENCE_TERMS:
            writer.writerow(item)

    print(f"[OK] Arquivo de exemplo criado: {REFERENCE_TERMS_FILE}")

def load_reference_terms() -> list[dict]:
    create_example_reference_terms_file()

    terms = []

    with open(REFERENCE_TERMS_FILE, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"entity_id", "entity_type", "label"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"O arquivo {REFERENCE_TERMS_FILE} precisa conter as colunas: "
                f"{', '.join(sorted(required_columns))}"
            )

        for row in reader:
            entity_id = row.get("entity_id", "").strip()
            entity_type = normalize_entity_type(row.get("entity_type", ""))
            label = row.get("label", "").strip()

            if not entity_id or not label:
                continue

            expanded_terms = generate_term_variants(label)

            terms.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "label": label,
                    "normalized_label": normalize_text(label),
                    "expanded_terms": expanded_terms,
                }
            )

    print(f"[OK] Termos de referência carregados: {len(terms)}")
    return terms


# ============================================================
# SCORE DE COMPATIBILIDADE
# ============================================================

def type_compatibility_bonus(reference_entity_type: str, candidate_uri: str) -> float:
    """
    Bônus/penalidade conforme compatibilidade entre o tipo da entidade
    de referência e o tipo provável da URI candidata.
    """
    reference_entity_type = normalize_entity_type(reference_entity_type)

    if reference_entity_type == "class":
        if is_probably_class_uri(candidate_uri):
            return 0.25

        if is_probably_resource_uri(candidate_uri):
            return -0.03

        if is_probably_property_uri(candidate_uri):
            return -0.18

    if reference_entity_type == "property":
        if is_probably_property_uri(candidate_uri):
            return 0.25

        if is_probably_class_uri(candidate_uri):
            return -0.20

        if is_probably_resource_uri(candidate_uri):
            return -0.10

    if reference_entity_type == "individual":
        if is_probably_resource_uri(candidate_uri):
            return 0.25

        if is_probably_class_uri(candidate_uri):
            return -0.02

        if is_probably_property_uri(candidate_uri):
            return -0.18

    return 0.0


def source_weight_for_candidate(reference_entity_type: str, candidate_uri: str, base_weight: float) -> float:
    """
    Ajusta peso da evidência conforme compatibilidade entre fonte e tipo.
    """
    reference_entity_type = normalize_entity_type(reference_entity_type)

    adjusted = base_weight

    if reference_entity_type == "class":
        if is_probably_class_uri(candidate_uri):
            adjusted += 0.15

        elif is_probably_resource_uri(candidate_uri):
            adjusted -= 0.08

        elif is_probably_property_uri(candidate_uri):
            adjusted -= 0.15

    elif reference_entity_type == "property":
        if is_probably_property_uri(candidate_uri):
            adjusted += 0.15

        elif is_probably_class_uri(candidate_uri):
            adjusted -= 0.15

        elif is_probably_resource_uri(candidate_uri):
            adjusted -= 0.10

    elif reference_entity_type == "individual":
        if is_probably_resource_uri(candidate_uri):
            adjusted += 0.15

        elif is_probably_property_uri(candidate_uri):
            adjusted -= 0.15

    return max(0.0, min(adjusted, 1.30))


# ============================================================
# CANDIDATOS
# ============================================================

def make_candidate(
    term: dict,
    candidate_uri: str,
    candidate_label: str,
    strategy: str,
    evidence_source: str,
    matched_term: str,
    evidence_weight: float,
) -> dict:
    lexical_score = combined_initial_score(
        reference_label=term["label"],
        candidate_label=candidate_label,
        candidate_uri=candidate_uri,
    )

    adjusted_evidence_weight = source_weight_for_candidate(
        term["entity_type"],
        candidate_uri,
        evidence_weight,
    )

    compatibility_bonus = type_compatibility_bonus(
        term["entity_type"],
        candidate_uri,
    )

    construction_bonus = 0.0

    if strategy == "dbpedia_uri_construction":
        if term["entity_type"] == "class" and is_probably_class_uri(candidate_uri):
            construction_bonus = 0.15

        elif term["entity_type"] == "property" and is_probably_property_uri(candidate_uri):
            construction_bonus = 0.15

        elif term["entity_type"] == "individual" and is_probably_resource_uri(candidate_uri):
            construction_bonus = 0.15

    final_score = (
        0.64 * lexical_score +
        0.18 * adjusted_evidence_weight +
        compatibility_bonus +
        construction_bonus
    )

    final_score = max(0.0, min(final_score, 1.0))

    return {
        "reference_entity_id": term["entity_id"],
        "reference_entity_type": term["entity_type"],
        "reference_label": term["label"],
        "reference_normalized_label": term["normalized_label"],
        "reference_expanded_terms": "|".join(term["expanded_terms"]),

        "candidate_uri": candidate_uri,
        "candidate_label": candidate_label,
        "candidate_uri_token": uri_last_token(candidate_uri),
        "candidate_namespace": uri_namespace(candidate_uri),

        "strategies": strategy,
        "evidence_sources": evidence_source,
        "matched_terms": matched_term,

        "strategy_count": 1,
        "evidence_source_count": 1,
        "average_evidence_weight": round(adjusted_evidence_weight, 6),

        "initial_score": round(final_score, 6),
    }


def merge_candidates(candidates: list[dict]) -> list[dict]:
    merged = {}

    for candidate in candidates:
        key = (
            candidate["reference_entity_id"],
            candidate["candidate_uri"]
        )

        if key not in merged:
            merged[key] = candidate.copy()
            merged[key]["_strategies"] = set(candidate["strategies"].split("|"))
            merged[key]["_sources"] = set(candidate["evidence_sources"].split("|"))
            merged[key]["_matched"] = set(candidate["matched_terms"].split("|"))
            merged[key]["_weights"] = [float(candidate["average_evidence_weight"])]
            merged[key]["_scores"] = [float(candidate["initial_score"])]
        else:
            merged[key]["_strategies"].update(candidate["strategies"].split("|"))
            merged[key]["_sources"].update(candidate["evidence_sources"].split("|"))
            merged[key]["_matched"].update(candidate["matched_terms"].split("|"))
            merged[key]["_weights"].append(float(candidate["average_evidence_weight"]))
            merged[key]["_scores"].append(float(candidate["initial_score"]))

    output = []

    for _, candidate in merged.items():
        strategies = sorted(candidate["_strategies"])
        sources = sorted(candidate["_sources"])
        matched = sorted(candidate["_matched"])

        avg_weight = sum(candidate["_weights"]) / len(candidate["_weights"])
        max_score = max(candidate["_scores"])

        strategy_bonus = min(len(strategies) * 0.04, 0.20)
        source_bonus = min(len(sources) * 0.04, 0.20)

        final_score = min(max_score + strategy_bonus + source_bonus, 1.0)

        candidate["strategies"] = "|".join(strategies)
        candidate["evidence_sources"] = "|".join(sources)
        candidate["matched_terms"] = "|".join(matched)
        candidate["strategy_count"] = len(strategies)
        candidate["evidence_source_count"] = len(sources)
        candidate["average_evidence_weight"] = round(avg_weight, 6)
        candidate["initial_score"] = round(final_score, 6)

        del candidate["_strategies"]
        del candidate["_sources"]
        del candidate["_matched"]
        del candidate["_weights"]
        del candidate["_scores"]

        output.append(candidate)

    return output


def keep_top_k_per_reference(candidates: list[dict]) -> list[dict]:
    grouped = {}

    for candidate in candidates:
        ref = candidate["reference_entity_id"]
        grouped.setdefault(ref, []).append(candidate)

    output = []

    for _, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                float(row["initial_score"]),
                int(row["evidence_source_count"]),
                int(row["strategy_count"]),
            ),
            reverse=True,
        )

        output.extend(rows[:TOP_K])

    return output


# ============================================================
# ESTRATÉGIAS MULTIFONTE EM STREAMING
# ============================================================

def generate_from_labels_csv(reference_terms: list[dict]) -> list[dict]:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {LABELS_CSV}")

    print("\n" + "=" * 80)
    print("[FONTE] labels.csv")
    print("=" * 80)

    candidates = []

    expanded_lookup = {}

    for term in reference_terms:
        for expanded_term in term["expanded_terms"]:
            expanded_lookup.setdefault(expanded_term, []).append(term)

    with open(LABELS_CSV, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            resource = row.get("resource", "").strip()
            label = row.get("label", "").strip()

            if not resource or not label:
                continue

            normalized_label = normalize_text(label)
            normalized_token = normalize_text(uri_last_token(resource))

            matched_terms = set()

            if normalized_label in expanded_lookup:
                matched_terms.add(normalized_label)

            if normalized_token in expanded_lookup:
                matched_terms.add(normalized_token)

            if not matched_terms:
                continue

            for matched_term in matched_terms:
                for term in expanded_lookup[matched_term]:

                    strategies = []

                    if normalized_label == term["normalized_label"]:
                        strategies.append("exact_label")

                    if normalized_label in term["expanded_terms"]:
                        strategies.append("expanded_label")

                    if normalized_token in term["expanded_terms"]:
                        strategies.append("resource_token")

                    strategy_name = "|".join(sorted(set(strategies))) or "label_stream"

                    candidates.append(
                        make_candidate(
                            term=term,
                            candidate_uri=resource,
                            candidate_label=label,
                            strategy=strategy_name,
                            evidence_source="dbpedia_labels",
                            matched_term=matched_term,
                            evidence_weight=0.90,
                        )
                    )

            if row_number % 500000 == 0:
                print(
                    f"[PROGRESSO] labels.csv: "
                    f"{row_number:,} linhas lidas | "
                    f"{len(candidates):,} candidatos"
                )

            if len(candidates) >= MAX_CANDIDATES_PER_SOURCE:
                print("[INFO] Limite de candidatos da fonte labels atingido.")
                break

            if MAX_LABEL_ROWS is not None and row_number >= MAX_LABEL_ROWS:
                print(f"[INFO] Limite de linhas labels atingido: {MAX_LABEL_ROWS:,}")
                break

    print(f"[OK] Candidatos vindos de labels.csv: {len(candidates):,}")
    return candidates


def generate_from_instance_types_csv(reference_terms: list[dict]) -> list[dict]:
    if not INSTANCE_TYPES_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INSTANCE_TYPES_CSV}")

    print("\n" + "=" * 80)
    print("[FONTE] instance_types.csv")
    print("=" * 80)

    candidates = []
    seen_pairs = set()

    relevant_terms = [
        term for term in reference_terms
        if term["entity_type"] in {"class", "individual"}
    ]

    expanded_lookup = {}

    for term in relevant_terms:
        for expanded_term in term["expanded_terms"]:
            expanded_lookup.setdefault(expanded_term, []).append(term)

    with open(INSTANCE_TYPES_CSV, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            resource = row.get("resource", "").strip()
            class_uri = row.get("class", "").strip()

            if not resource or not class_uri:
                continue

            class_token = normalize_text(uri_last_token(class_uri))
            resource_token = normalize_text(uri_last_token(resource))

            possible_matches = set()

            if class_token in expanded_lookup:
                possible_matches.add(class_token)

            class_token_stemmed = " ".join(
                simple_stem(token) for token in split_tokens(class_token)
            )

            if class_token_stemmed in expanded_lookup:
                possible_matches.add(class_token_stemmed)

            for part in split_tokens(class_token):
                if part in expanded_lookup:
                    possible_matches.add(part)

            if resource_token in expanded_lookup:
                possible_matches.add(resource_token)

            if not possible_matches:
                continue

            for matched_term in possible_matches:
                for term in expanded_lookup[matched_term]:

                    if term["entity_type"] == "class":
                        candidate_uri = class_uri
                        candidate_label = uri_last_token(class_uri)
                        strategy = "ontology_class_token"
                        evidence_source = "dbpedia_instance_types_classes"
                        evidence_weight = 0.98

                    elif term["entity_type"] == "individual":
                        candidate_uri = resource
                        candidate_label = uri_last_token(resource)
                        strategy = "resource_from_instance_types"
                        evidence_source = "dbpedia_instance_types_resources"
                        evidence_weight = 0.95

                    else:
                        continue

                    pair_key = (term["entity_id"], candidate_uri)

                    if pair_key in seen_pairs:
                        continue

                    seen_pairs.add(pair_key)

                    candidates.append(
                        make_candidate(
                            term=term,
                            candidate_uri=candidate_uri,
                            candidate_label=candidate_label,
                            strategy=strategy,
                            evidence_source=evidence_source,
                            matched_term=matched_term,
                            evidence_weight=evidence_weight,
                        )
                    )

            if row_number % 500000 == 0:
                print(
                    f"[PROGRESSO] instance_types.csv: "
                    f"{row_number:,} linhas lidas | "
                    f"{len(candidates):,} candidatos"
                )

            if len(candidates) >= MAX_CANDIDATES_PER_SOURCE:
                print("[INFO] Limite de candidatos da fonte instance_types atingido.")
                break

            if MAX_INSTANCE_TYPE_ROWS is not None and row_number >= MAX_INSTANCE_TYPE_ROWS:
                print(f"[INFO] Limite de linhas instance_types atingido: {MAX_INSTANCE_TYPE_ROWS:,}")
                break

    print(f"[OK] Candidatos vindos de instance_types.csv: {len(candidates):,}")
    return candidates


def load_property_frequency_index() -> dict:
    if not PROPERTY_FREQUENCY_INDEX.exists():
        print(f"[AVISO] Índice não encontrado: {PROPERTY_FREQUENCY_INDEX}")
        return {}

    print(f"[LOAD] {PROPERTY_FREQUENCY_INDEX.name}")

    with open(PROPERTY_FREQUENCY_INDEX, "r", encoding=JSON_ENCODING) as file:
        return json.load(file)


def generate_from_property_frequency(reference_terms: list[dict]) -> list[dict]:
    print("\n" + "=" * 80)
    print("[FONTE] property_frequency_index.json")
    print("=" * 80)

    property_frequency_index = load_property_frequency_index()

    candidates = []

    property_terms = [
        term for term in reference_terms
        if term["entity_type"] == "property"
    ]

    if not property_terms:
        print("[INFO] Nenhum termo de referência do tipo propriedade.")
        return candidates

    for property_uri, frequency in property_frequency_index.items():
        property_token = normalize_text(uri_last_token(property_uri))

        if not property_token:
            continue

        for term in property_terms:
            best_score = 0.0
            best_match = ""

            for expanded_term in term["expanded_terms"]:
                score = max(
                    levenshtein_similarity(expanded_term, property_token),
                    token_jaccard_similarity(expanded_term, property_token),
                    stemmed_jaccard_similarity(expanded_term, property_token),
                    char_bigram_jaccard_similarity(expanded_term, property_token),
                    prefix_similarity(expanded_term, property_token),
                )

                if score > best_score:
                    best_score = score
                    best_match = expanded_term

            if best_score < 0.78:
                continue

            try:
                freq_value = float(frequency)
            except (ValueError, TypeError):
                freq_value = 0.0

            freq_bonus = min(math.log(freq_value + 1.0) / 20.0, 0.20)

            evidence_weight = 0.82 + freq_bonus

            candidates.append(
                make_candidate(
                    term=term,
                    candidate_uri=property_uri,
                    candidate_label=uri_last_token(property_uri),
                    strategy="property_token|property_frequency",
                    evidence_source="dbpedia_properties|dbpedia_property_frequency",
                    matched_term=best_match,
                    evidence_weight=evidence_weight,
                )
            )

            if len(candidates) >= MAX_CANDIDATES_PER_SOURCE:
                print("[INFO] Limite de candidatos da fonte property_frequency atingido.")
                print(f"[OK] Candidatos vindos de property_frequency: {len(candidates):,}")
                return candidates

    print(f"[OK] Candidatos vindos de property_frequency: {len(candidates):,}")
    return candidates


# ============================================================
# SAÍDA
# ============================================================

def save_candidates(candidates: list[dict]) -> None:
    if not candidates:
        raise ValueError("Nenhum candidato foi gerado.")

    fieldnames = [
        "reference_entity_id",
        "reference_entity_type",
        "reference_label",
        "reference_normalized_label",
        "reference_expanded_terms",

        "candidate_uri",
        "candidate_label",
        "candidate_uri_token",
        "candidate_namespace",

        "strategies",
        "evidence_sources",
        "matched_terms",

        "strategy_count",
        "evidence_source_count",
        "average_evidence_weight",

        "initial_score",
    ]

    with open(OUTPUT_FILE, "w", encoding=CSV_ENCODING, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in candidates:
            writer.writerow(row)

    print(f"\n[OK] Candidatos salvos em: {OUTPUT_FILE}")


def save_manifest(reference_terms: list[dict], candidates: list[dict]) -> None:
    with open(MANIFEST_FILE, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Multi-source Candidate Generation Manifest\n")
        file.write("=================================================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Processed directory: {PROCESSED_DIR}\n")
        file.write(f"Reference terms file: {REFERENCE_TERMS_FILE}\n")
        file.write(f"Output file: {OUTPUT_FILE}\n")
        file.write(f"Top-k: {TOP_K}\n")
        file.write(f"Max candidates per source: {MAX_CANDIDATES_PER_SOURCE}\n")
        file.write(f"Max label rows: {MAX_LABEL_ROWS}\n")
        file.write(f"Max instance type rows: {MAX_INSTANCE_TYPE_ROWS}\n\n")

        file.write("Controlled synonyms:\n")
        for key, values in CONTROLLED_SYNONYMS.items():
            file.write(f"  {key}: {values}\n")

        file.write("\nControlled direct candidates:\n")
        for key, values in CONTROLLED_DIRECT_CANDIDATES.items():
            file.write(f"  {key}: {values}\n")

        file.write("\n")
        file.write(f"Reference terms: {len(reference_terms)}\n")
        file.write(f"Generated candidates: {len(candidates)}\n\n")

        file.write("Expanded terms per reference entity:\n")
        for term in reference_terms:
            file.write(f"  {term['entity_id']} -> {term['expanded_terms']}\n")

        file.write("\nCandidates per reference term:\n")

        counter = {}

        for candidate in candidates:
            ref = candidate["reference_entity_id"]
            counter[ref] = counter.get(ref, 0) + 1

        for ref, count in counter.items():
            file.write(f"  {ref}: {count}\n")

    print(f"[OK] Manifesto salvo em: {MANIFEST_FILE}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DBpedia Multi-source Candidate Generator - Full Adjusted Version")
    print("=" * 80)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Reference terms file: {REFERENCE_TERMS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Manifest file: {MANIFEST_FILE}")
    print(f"Top-k: {TOP_K}")
    print("=" * 80)

    ensure_directories()

    try:
        reference_terms = load_reference_terms()

        print("\n[INFO] Termos de referência e expansões:")
        for term in reference_terms:
            print(
                f"  - {term['entity_id']} | {term['label']} | "
                f"tipo={term['entity_type']} | expansões={term['expanded_terms']}"
            )

        all_candidates = []

        print("\n" + "=" * 80)
        print("[FONTE] dbpedia_uri_construction")
        print("=" * 80)

        constructed_candidates = []

        for term in reference_terms:
            constructed_candidates.extend(
                generate_constructed_dbpedia_uris(term)
            )

        print(f"[OK] Candidatos construídos por URI: {len(constructed_candidates):,}")

        all_candidates.extend(constructed_candidates)

        all_candidates.extend(
            generate_from_labels_csv(reference_terms)
        )

        all_candidates.extend(
            generate_from_instance_types_csv(reference_terms)
        )

        all_candidates.extend(
            generate_from_property_frequency(reference_terms)
        )

        print("\n" + "=" * 80)
        print("[INFO] Fundindo candidatos duplicados")
        print("=" * 80)

        merged_candidates = merge_candidates(all_candidates)
        final_candidates = keep_top_k_per_reference(merged_candidates)

        print(f"[OK] Candidatos brutos: {len(all_candidates):,}")
        print(f"[OK] Candidatos após fusão: {len(merged_candidates):,}")
        print(f"[OK] Candidatos finais top-k: {len(final_candidates):,}")

        save_candidates(final_candidates)
        save_manifest(reference_terms, final_candidates)

        print("\n" + "=" * 80)
        print("RESUMO")
        print("=" * 80)
        print(f"Termos de referência: {len(reference_terms)}")
        print(f"Candidatos finais: {len(final_candidates)}")
        print(f"Arquivo de saída: {OUTPUT_FILE}")
        print("\n[OK] Geração multifonte de candidatos concluída com sucesso.")

    except Exception as error:
        print(f"\n[ERRO] Falha na geração multifonte de candidatos: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()