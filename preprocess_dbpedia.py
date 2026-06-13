# Script for preprocessing DBpedia dumps

"""
preprocess_dbpedia.py

Script para pré-processar dumps da DBpedia em formato .ttl.bz2
e convertê-los para arquivos CSV mais simples de manipular.

Entrada:
    experiments/data/dbpedia_raw

Saída:
    experiments/data/dbpedia_processed
"""

from pathlib import Path
import bz2
import csv
import re
import sys
from datetime import datetime


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PROJECT_ROOT = Path(".")

RAW_DIR = PROJECT_ROOT / "experiments" / "data" / "dbpedia_raw"
PROCESSED_DIR = PROJECT_ROOT / "experiments" / "data" / "dbpedia_processed"

MAX_LINES_PER_FILE = None
# Para teste rápido, você pode usar:
# MAX_LINES_PER_FILE = 100000

CSV_ENCODING = "utf-8"
TTL_ENCODING = "utf-8"


# ============================================================
# ARQUIVOS DE ENTRADA E SAÍDA
# ============================================================

FILES = {
    "labels": {
        "input": "labels_lang=en.ttl.bz2",
        "output": "labels.csv",
        "columns": ["resource", "label"],
    },
    "instance_types": {
        "input": "instance-types_lang=en_specific.ttl.bz2",
        "output": "instance_types.csv",
        "columns": ["resource", "class"],
    },
    "mappingbased_literals": {
        "input": "mappingbased-literals_lang=en.ttl.bz2",
        "output": "literal_properties.csv",
        "columns": ["resource", "property", "value"],
    },
    "mappingbased_objects": {
        "input": "mappingbased-objects_lang=en.ttl.bz2",
        "output": "object_properties.csv",
        "columns": ["resource", "property", "object"],
    },
    "infobox_properties": {
        "input": "infobox-properties_lang=en.ttl.bz2",
        "output": "infobox_properties.csv",
        "columns": ["resource", "property", "value"],
    },
    "commons_sameas_links": {
        "input": "commons-sameas-links_lang=en.ttl.bz2",
        "output": "sameas_links.csv",
        "columns": ["resource", "same_as"],
    },
}


# ============================================================
# EXPRESSÕES REGULARES
# ============================================================

URI_PATTERN = re.compile(r"<([^>]*)>")
LITERAL_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"(?:@[a-zA-Z\-]+|\^\^<[^>]+>)?')


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ensure_directories() -> None:
    """
    Cria a pasta de saída se ela não existir.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Pasta de saída verificada: {PROCESSED_DIR}")


def clean_uri(uri: str) -> str:
    """
    Limpa uma URI da DBpedia, mantendo o identificador completo.

    Exemplo:
        http://dbpedia.org/resource/Chardonnay
    """
    return uri.strip()


def clean_literal(value: str) -> str:
    """
    Limpa literais RDF, removendo escapes simples.
    """
    value = value.replace('\\"', '"')
    value = value.replace("\\n", " ")
    value = value.replace("\\t", " ")
    value = value.replace("\\r", " ")
    return value.strip()


def extract_uris(line: str) -> list[str]:
    """
    Extrai URIs de uma linha RDF em Turtle/N-Triples.
    """
    return URI_PATTERN.findall(line)


def extract_literal(line: str) -> str | None:
    """
    Extrai o primeiro literal de uma linha RDF.
    """
    match = LITERAL_PATTERN.search(line)

    if match:
        return clean_literal(match.group(1))

    return None


def iter_bz2_lines(path: Path):
    """
    Itera sobre linhas de um arquivo .bz2 de forma incremental.
    """
    with bz2.open(path, "rt", encoding=TTL_ENCODING, errors="replace") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            yield line


def write_manifest(results: dict) -> None:
    """
    Salva um manifesto da etapa de pré-processamento.
    """
    manifest_path = PROCESSED_DIR / "preprocessing_manifest.txt"

    with open(manifest_path, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Preprocessing Manifest\n")
        file.write("==============================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Raw directory: {RAW_DIR}\n")
        file.write(f"Processed directory: {PROCESSED_DIR}\n")
        file.write(f"Max lines per file: {MAX_LINES_PER_FILE}\n\n")

        for name, info in results.items():
            file.write(f"[{info['status']}] {name}\n")
            file.write(f"  input:  {info['input']}\n")
            file.write(f"  output: {info['output']}\n")
            file.write(f"  rows:   {info['rows']}\n")
            file.write(f"  error:  {info.get('error', '')}\n\n")

    print(f"\n[OK] Manifesto salvo em: {manifest_path}")


# ============================================================
# PARSERS ESPECÍFICOS
# ============================================================

def parse_labels(line: str) -> tuple[str, str] | None:
    """
    Parser para labels_lang=en.ttl.bz2.

    Formato esperado:
        <resource> <predicate> "label"@en .
    """
    uris = extract_uris(line)
    literal = extract_literal(line)

    if len(uris) >= 1 and literal:
        return clean_uri(uris[0]), literal

    return None


def parse_instance_types(line: str) -> tuple[str, str] | None:
    """
    Parser para instance-types_lang=en_specific.ttl.bz2.

    Formato esperado:
        <resource> rdf:type <class> .
    """
    uris = extract_uris(line)

    if len(uris) >= 2:
        return clean_uri(uris[0]), clean_uri(uris[1])

    return None


def parse_literal_property(line: str) -> tuple[str, str, str] | None:
    """
    Parser para propriedades literais.

    Formato esperado:
        <resource> <property> "value" .
    """
    uris = extract_uris(line)
    literal = extract_literal(line)

    if len(uris) >= 2 and literal:
        return clean_uri(uris[0]), clean_uri(uris[1]), literal

    return None


def parse_object_property(line: str) -> tuple[str, str, str] | None:
    """
    Parser para propriedades objeto.

    Formato esperado:
        <resource> <property> <object> .
    """
    uris = extract_uris(line)

    if len(uris) >= 3:
        return clean_uri(uris[0]), clean_uri(uris[1]), clean_uri(uris[2])

    return None


def parse_sameas(line: str) -> tuple[str, str] | None:
    """
    Parser para links owl:sameAs.

    Formato esperado:
        <resource> owl:sameAs <same_as> .
    """
    uris = extract_uris(line)

    if len(uris) >= 2:
        return clean_uri(uris[0]), clean_uri(uris[1])

    return None


PARSERS = {
    "labels": parse_labels,
    "instance_types": parse_instance_types,
    "mappingbased_literals": parse_literal_property,
    "mappingbased_objects": parse_object_property,
    "infobox_properties": parse_literal_property,
    "commons_sameas_links": parse_sameas,
}


# ============================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# ============================================================

def preprocess_file(name: str, config: dict) -> dict:
    """
    Pré-processa um arquivo específico da DBpedia.
    """
    input_path = RAW_DIR / config["input"]
    output_path = PROCESSED_DIR / config["output"]
    columns = config["columns"]
    parser = PARSERS[name]

    print("\n" + "=" * 80)
    print(f"[PROCESSANDO] {name}")
    print(f"[ENTRADA] {input_path}")
    print(f"[SAÍDA] {output_path}")
    print("=" * 80)

    if not input_path.exists():
        message = f"Arquivo de entrada não encontrado: {input_path}"
        print(f"[ERRO] {message}")

        return {
            "status": "FAILED",
            "input": str(input_path),
            "output": str(output_path),
            "rows": 0,
            "error": message,
        }

    rows_written = 0
    lines_read = 0

    try:
        with open(output_path, "w", encoding=CSV_ENCODING, newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)

            for line in iter_bz2_lines(input_path):
                lines_read += 1

                parsed = parser(line)

                if parsed is not None:
                    writer.writerow(parsed)
                    rows_written += 1

                if lines_read % 500000 == 0:
                    print(
                        f"[PROGRESSO] {name}: "
                        f"{lines_read:,} linhas lidas | "
                        f"{rows_written:,} linhas gravadas"
                    )

                if MAX_LINES_PER_FILE is not None and lines_read >= MAX_LINES_PER_FILE:
                    print(f"[INFO] Limite de teste atingido: {MAX_LINES_PER_FILE:,} linhas")
                    break

        print(
            f"[OK] {name}: "
            f"{lines_read:,} linhas lidas | "
            f"{rows_written:,} linhas gravadas"
        )

        return {
            "status": "OK",
            "input": str(input_path),
            "output": str(output_path),
            "rows": rows_written,
            "error": "",
        }

    except Exception as error:
        print(f"[ERRO] Falha ao processar {name}: {error}")

        return {
            "status": "FAILED",
            "input": str(input_path),
            "output": str(output_path),
            "rows": rows_written,
            "error": str(error),
        }


def main() -> None:
    """
    Executa o pré-processamento dos dumps selecionados.
    """
    print("=" * 80)
    print("DBpedia Preprocessor")
    print("=" * 80)
    print(f"Pasta raw: {RAW_DIR}")
    print(f"Pasta processed: {PROCESSED_DIR}")
    print("=" * 80)

    ensure_directories()

    results = {}

    for name, config in FILES.items():
        result = preprocess_file(name, config)
        results[name] = result

    write_manifest(results)

    total = len(results)
    successful = sum(1 for item in results.values() if item["status"] == "OK")
    failed = total - successful

    print("\n" + "=" * 80)
    print("RESUMO DO PRÉ-PROCESSAMENTO")
    print("=" * 80)
    print(f"Total de arquivos: {total}")
    print(f"Processados com sucesso: {successful}")
    print(f"Falhas: {failed}")

    if failed > 0:
        print("\n[AVISO] Alguns arquivos não foram processados.")
        sys.exit(1)

    print("\n[OK] Todos os arquivos foram pré-processados com sucesso.")


if __name__ == "__main__":
    main()