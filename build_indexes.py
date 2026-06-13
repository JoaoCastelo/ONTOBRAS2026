# Script for building local indexes

"""
build_indexes.py

Script para construir índices locais a partir dos arquivos CSV
pré-processados da DBpedia.

Entrada:
    experiments/data/dbpedia_processed

Saída:
    experiments/data/dbpedia_processed/indexes

Objetivo:
    Criar estruturas de busca rápida para geração de candidatos,
    extração de features e validação de alinhamentos.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import csv
import json
import re
import sys


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PROJECT_ROOT = Path(".")

PROCESSED_DIR = PROJECT_ROOT / "experiments" / "data" / "dbpedia_processed"
INDEX_DIR = PROCESSED_DIR / "indexes"

CSV_ENCODING = "utf-8"
JSON_ENCODING = "utf-8"

# Para teste rápido, você pode usar um número inteiro.
# Para processar tudo, deixe como None.
MAX_ROWS_PER_FILE = None
# Exemplo:
# MAX_ROWS_PER_FILE = 100000


# ============================================================
# ARQUIVOS DE ENTRADA
# ============================================================

FILES = {
    "labels": PROCESSED_DIR / "labels.csv",
    "instance_types": PROCESSED_DIR / "instance_types.csv",
    "literal_properties": PROCESSED_DIR / "literal_properties.csv",
    "object_properties": PROCESSED_DIR / "object_properties.csv",
    "infobox_properties": PROCESSED_DIR / "infobox_properties.csv",
    "sameas_links": PROCESSED_DIR / "sameas_links.csv",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ensure_directories() -> None:
    """
    Cria a pasta de índices, caso ela não exista.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Pasta de índices verificada: {INDEX_DIR}")


def normalize_text(text: str) -> str:
    """
    Normaliza rótulos e termos para busca lexical.

    Exemplo:
        "Cabernet Sauvignon" -> "cabernet sauvignon"
        "New_York_City" -> "new york city"
    """
    if text is None:
        return ""

    text = text.strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def uri_last_token(uri: str) -> str:
    """
    Extrai o último token de uma URI.

    Exemplo:
        http://dbpedia.org/resource/Chardonnay -> Chardonnay
        http://dbpedia.org/ontology/Wine -> Wine
    """
    if not uri:
        return ""

    uri = uri.rstrip("/#")
    return re.split(r"[/#]", uri)[-1]


def safe_read_csv(path: Path):
    """
    Lê um CSV de forma incremental.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding=CSV_ENCODING, newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            yield row_number, row

            if MAX_ROWS_PER_FILE is not None and row_number >= MAX_ROWS_PER_FILE:
                break


def save_json(data, path: Path) -> None:
    """
    Salva dados em JSON.
    """
    with open(path, "w", encoding=JSON_ENCODING) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_manifest(results: dict) -> None:
    """
    Salva um manifesto dos índices criados.
    """
    manifest_path = INDEX_DIR / "index_manifest.txt"

    with open(manifest_path, "w", encoding="utf-8") as file:
        file.write("DBpedia Index Manifest\n")
        file.write("======================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Processed directory: {PROCESSED_DIR}\n")
        file.write(f"Index directory: {INDEX_DIR}\n")
        file.write(f"Max rows per file: {MAX_ROWS_PER_FILE}\n\n")

        for name, info in results.items():
            file.write(f"[{info['status']}] {name}\n")
            file.write(f"  output: {info.get('output', '')}\n")
            file.write(f"  entries: {info.get('entries', 0)}\n")
            file.write(f"  error: {info.get('error', '')}\n\n")

    print(f"\n[OK] Manifesto salvo em: {manifest_path}")


def convert_defaultdict_to_dict(data):
    """
    Converte defaultdict/list/set em estruturas serializáveis em JSON.
    """
    if isinstance(data, defaultdict):
        data = dict(data)

    if isinstance(data, dict):
        return {key: convert_defaultdict_to_dict(value) for key, value in data.items()}

    if isinstance(data, set):
        return sorted(list(data))

    if isinstance(data, list):
        return [convert_defaultdict_to_dict(item) for item in data]

    return data


# ============================================================
# ÍNDICE 1: LABEL INDEX
# ============================================================

def build_label_indexes() -> dict:
    """
    Constrói dois índices:

    1. label_index:
        resource -> label

    2. normalized_label_index:
        normalized_label -> [resource1, resource2, ...]

    Esses índices serão usados para geração de candidatos.
    """
    input_path = FILES["labels"]

    label_index = {}
    normalized_label_index = defaultdict(list)

    print("\n" + "=" * 80)
    print("[INDEX] label_index e normalized_label_index")
    print(f"[INPUT] {input_path}")
    print("=" * 80)

    rows = 0

    for row_number, row in safe_read_csv(input_path):
        resource = row.get("resource", "").strip()
        label = row.get("label", "").strip()

        if not resource or not label:
            continue

        normalized_label = normalize_text(label)

        if not normalized_label:
            continue

        label_index[resource] = label
        normalized_label_index[normalized_label].append(resource)

        rows += 1

        if row_number % 500000 == 0:
            print(f"[PROGRESSO] labels: {row_number:,} linhas lidas | {rows:,} labels indexados")

    label_index_path = INDEX_DIR / "label_index.json"
    normalized_label_index_path = INDEX_DIR / "normalized_label_index.json"

    save_json(label_index, label_index_path)
    save_json(convert_defaultdict_to_dict(normalized_label_index), normalized_label_index_path)

    print(f"[OK] label_index salvo: {label_index_path}")
    print(f"[OK] normalized_label_index salvo: {normalized_label_index_path}")
    print(f"[OK] Total de labels indexados: {len(label_index):,}")

    return {
        "status": "OK",
        "output": f"{label_index_path}; {normalized_label_index_path}",
        "entries": len(label_index),
        "error": "",
    }


# ============================================================
# ÍNDICE 2: RESOURCE TYPE INDEX E CLASS INSTANCE INDEX
# ============================================================

def build_type_indexes() -> dict:
    """
    Constrói dois índices:

    1. resource_type_index:
        resource -> [class1, class2, ...]

    2. class_instance_index:
        class -> [resource1, resource2, ...]

    Esses índices são essenciais para usar a ABox da DBpedia
    como evidência para alinhamento.
    """
    input_path = FILES["instance_types"]

    resource_type_index = defaultdict(set)
    class_instance_index = defaultdict(set)

    print("\n" + "=" * 80)
    print("[INDEX] resource_type_index e class_instance_index")
    print(f"[INPUT] {input_path}")
    print("=" * 80)

    rows = 0

    for row_number, row in safe_read_csv(input_path):
        resource = row.get("resource", "").strip()
        class_uri = row.get("class", "").strip()

        if not resource or not class_uri:
            continue

        resource_type_index[resource].add(class_uri)
        class_instance_index[class_uri].add(resource)

        rows += 1

        if row_number % 500000 == 0:
            print(
                f"[PROGRESSO] instance_types: "
                f"{row_number:,} linhas lidas | "
                f"{len(resource_type_index):,} recursos | "
                f"{len(class_instance_index):,} classes"
            )

    resource_type_index = convert_defaultdict_to_dict(resource_type_index)
    class_instance_index = convert_defaultdict_to_dict(class_instance_index)

    resource_type_index_path = INDEX_DIR / "resource_type_index.json"
    class_instance_index_path = INDEX_DIR / "class_instance_index.json"

    save_json(resource_type_index, resource_type_index_path)
    save_json(class_instance_index, class_instance_index_path)

    print(f"[OK] resource_type_index salvo: {resource_type_index_path}")
    print(f"[OK] class_instance_index salvo: {class_instance_index_path}")
    print(f"[OK] Recursos tipados: {len(resource_type_index):,}")
    print(f"[OK] Classes encontradas: {len(class_instance_index):,}")

    return {
        "status": "OK",
        "output": f"{resource_type_index_path}; {class_instance_index_path}",
        "entries": len(resource_type_index),
        "error": "",
    }


# ============================================================
# ÍNDICE 3: ENTITY PROPERTY INDEX
# ============================================================

def build_entity_property_index() -> dict:
    """
    Constrói índice de propriedades por entidade:

        resource -> {
            literal_properties: [prop1, prop2, ...],
            object_properties: [prop1, prop2, ...],
            infobox_properties: [prop1, prop2, ...]
        }

    Esse índice será usado para extrair features estatísticas,
    estruturais e de caracterização das entidades.
    """
    entity_property_index = defaultdict(
        lambda: {
            "literal_properties": set(),
            "object_properties": set(),
            "infobox_properties": set(),
        }
    )

    print("\n" + "=" * 80)
    print("[INDEX] entity_property_index")
    print("=" * 80)

    sources = [
        ("literal_properties", FILES["literal_properties"], "property"),
        ("object_properties", FILES["object_properties"], "property"),
        ("infobox_properties", FILES["infobox_properties"], "property"),
    ]

    total_rows = 0

    for source_name, input_path, property_column in sources:
        print(f"\n[INPUT] {source_name}: {input_path}")

        for row_number, row in safe_read_csv(input_path):
            resource = row.get("resource", "").strip()
            prop = row.get(property_column, "").strip()

            if not resource or not prop:
                continue

            entity_property_index[resource][source_name].add(prop)
            total_rows += 1

            if row_number % 500000 == 0:
                print(
                    f"[PROGRESSO] {source_name}: "
                    f"{row_number:,} linhas lidas | "
                    f"{len(entity_property_index):,} entidades indexadas"
                )

    entity_property_index = convert_defaultdict_to_dict(entity_property_index)

    output_path = INDEX_DIR / "entity_property_index.json"
    save_json(entity_property_index, output_path)

    print(f"\n[OK] entity_property_index salvo: {output_path}")
    print(f"[OK] Entidades com propriedades: {len(entity_property_index):,}")
    print(f"[OK] Total de linhas analisadas: {total_rows:,}")

    return {
        "status": "OK",
        "output": str(output_path),
        "entries": len(entity_property_index),
        "error": "",
    }


# ============================================================
# ÍNDICE 4: PROPERTY FREQUENCY INDEX
# ============================================================

def build_property_frequency_index() -> dict:
    """
    Constrói índice de frequência de propriedades:

        property -> frequency

    Esse índice será usado como evidência estatística.
    Propriedades muito frequentes tendem a ser menos discriminativas,
    enquanto propriedades frequentes dentro de um domínio podem ser úteis.
    """
    property_counter = Counter()

    print("\n" + "=" * 80)
    print("[INDEX] property_frequency_index")
    print("=" * 80)

    sources = [
        ("literal_properties", FILES["literal_properties"]),
        ("object_properties", FILES["object_properties"]),
        ("infobox_properties", FILES["infobox_properties"]),
    ]

    total_rows = 0

    for source_name, input_path in sources:
        print(f"\n[INPUT] {source_name}: {input_path}")

        for row_number, row in safe_read_csv(input_path):
            prop = row.get("property", "").strip()

            if not prop:
                continue

            property_counter[prop] += 1
            total_rows += 1

            if row_number % 500000 == 0:
                print(
                    f"[PROGRESSO] {source_name}: "
                    f"{row_number:,} linhas lidas | "
                    f"{len(property_counter):,} propriedades únicas"
                )

    output_path = INDEX_DIR / "property_frequency_index.json"
    save_json(dict(property_counter.most_common()), output_path)

    print(f"\n[OK] property_frequency_index salvo: {output_path}")
    print(f"[OK] Propriedades únicas: {len(property_counter):,}")
    print(f"[OK] Total de ocorrências: {total_rows:,}")

    return {
        "status": "OK",
        "output": str(output_path),
        "entries": len(property_counter),
        "error": "",
    }


# ============================================================
# ÍNDICE 5: RESOURCE TOKEN INDEX
# ============================================================

def build_resource_token_index() -> dict:
    """
    Constrói índice baseado no último token da URI:

        normalized_token -> [resource1, resource2, ...]

    Exemplo:
        chardonnay -> http://dbpedia.org/resource/Chardonnay

    Esse índice é útil quando não há label disponível ou quando
    queremos combinar label com URI.
    """
    input_path = FILES["labels"]

    resource_token_index = defaultdict(list)

    print("\n" + "=" * 80)
    print("[INDEX] resource_token_index")
    print(f"[INPUT] {input_path}")
    print("=" * 80)

    rows = 0

    for row_number, row in safe_read_csv(input_path):
        resource = row.get("resource", "").strip()

        if not resource:
            continue

        token = uri_last_token(resource)
        normalized_token = normalize_text(token)

        if not normalized_token:
            continue

        resource_token_index[normalized_token].append(resource)
        rows += 1

        if row_number % 500000 == 0:
            print(
                f"[PROGRESSO] resource_token_index: "
                f"{row_number:,} linhas lidas | "
                f"{len(resource_token_index):,} tokens"
            )

    resource_token_index = convert_defaultdict_to_dict(resource_token_index)

    output_path = INDEX_DIR / "resource_token_index.json"
    save_json(resource_token_index, output_path)

    print(f"[OK] resource_token_index salvo: {output_path}")
    print(f"[OK] Tokens únicos: {len(resource_token_index):,}")

    return {
        "status": "OK",
        "output": str(output_path),
        "entries": len(resource_token_index),
        "error": "",
    }


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Executa a construção de todos os índices.
    """
    print("=" * 80)
    print("DBpedia Index Builder")
    print("=" * 80)
    print(f"Pasta processed: {PROCESSED_DIR}")
    print(f"Pasta indexes: {INDEX_DIR}")
    print(f"Limite de linhas por arquivo: {MAX_ROWS_PER_FILE}")
    print("=" * 80)

    ensure_directories()

    results = {}

    index_builders = {
        "label_indexes": build_label_indexes,
        "type_indexes": build_type_indexes,
        "entity_property_index": build_entity_property_index,
        "property_frequency_index": build_property_frequency_index,
        "resource_token_index": build_resource_token_index,
    }

    for name, builder in index_builders.items():
        try:
            result = builder()
            results[name] = result

        except Exception as error:
            print(f"[ERRO] Falha ao construir índice {name}: {error}")
            results[name] = {
                "status": "FAILED",
                "output": "",
                "entries": 0,
                "error": str(error),
            }

    save_manifest(results)

    total = len(results)
    successful = sum(1 for item in results.values() if item["status"] == "OK")
    failed = total - successful

    print("\n" + "=" * 80)
    print("RESUMO DA CONSTRUÇÃO DE ÍNDICES")
    print("=" * 80)
    print(f"Total de índices: {total}")
    print(f"Índices criados com sucesso: {successful}")
    print(f"Falhas: {failed}")

    if failed > 0:
        print("\n[AVISO] Alguns índices não foram criados.")
        sys.exit(1)

    print("\n[OK] Todos os índices foram criados com sucesso.")


if __name__ == "__main__":
    main()