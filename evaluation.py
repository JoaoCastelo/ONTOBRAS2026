"""
evaluation.py

Avaliação dos alinhamentos ranqueados contra um gold standard.

Esta versão usa experiment_config.py para alternar entre:
    EXPERIMENT_NAME = "pilot"
    EXPERIMENT_NAME = "expanded"

Entrada:
    ranked_alignments.csv
    best_alignments.csv
    gold_standard.csv ou gold_standard_expanded.csv

Saída:
    evaluation_metrics.csv
    evaluation_details.csv
    evaluation_manifest.txt

Autor: João Castelo
Projeto: Validação Fuzzy-LODAlign com DBpedia
"""

from datetime import datetime
import csv
import sys

from experiment_config import (
    EXPERIMENT_NAME,
    RANKED_ALIGNMENTS_FILE,
    BEST_ALIGNMENTS_FILE,
    GOLD_STANDARD_FILE,
    EVALUATION_METRICS_FILE,
    EVALUATION_DETAILS_FILE,
    EVALUATION_MANIFEST_FILE,
    ensure_experiment_directories,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_ENCODING = "utf-8"

TOP_K_VALUES = [1, 5, 10]


# ============================================================
# UTILITÁRIOS
# ============================================================

def ensure_directories() -> None:
    ensure_experiment_directories()
    print(f"[OK] Experimento ativo: {EXPERIMENT_NAME}")
    print(f"[OK] Ranked input: {RANKED_ALIGNMENTS_FILE}")
    print(f"[OK] Best input: {BEST_ALIGNMENTS_FILE}")
    print(f"[OK] Gold standard: {GOLD_STANDARD_FILE}")
    print(f"[OK] Metrics output: {EVALUATION_METRICS_FILE}")
    print(f"[OK] Details output: {EVALUATION_DETAILS_FILE}")
    print(f"[OK] Log: {EVALUATION_MANIFEST_FILE}")


def to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default

        return int(float(value))

    except (ValueError, TypeError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


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
    """
    Lê o gold standard.

    Formato esperado:
        reference_entity_id,reference_label,expected_candidate_uri,relation

    Permite mais de uma URI esperada para a mesma entidade de referência.
    """
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
# AVALIAÇÃO
# ============================================================

def group_ranked_by_reference(ranked_rows: list[dict]) -> dict:
    grouped = {}

    for row in ranked_rows:
        reference_id = row.get("reference_entity_id", "").strip()

        if not reference_id:
            continue

        grouped.setdefault(reference_id, []).append(row)

    for reference_id, rows in grouped.items():
        rows.sort(
            key=lambda row: to_int(row.get("rank")),
            reverse=False,
        )

    print(f"[OK] Entidades com candidatos ranqueados: {len(grouped)}")

    return grouped


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


def evaluate_ranked_alignments(ranked_rows: list[dict], gold: dict) -> tuple[list[dict], list[dict]]:
    grouped = group_ranked_by_reference(ranked_rows)

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
            top1 = ranked_candidates[0]

            detail_row["top1_candidate_uri"] = top1.get("candidate_uri", "")
            detail_row["top1_candidate_label"] = top1.get("candidate_label", "")
            detail_row["top1_score"] = top1.get("final_alignment_score", "")
            detail_row["top1_rank"] = top1.get("rank", "")
        else:
            detail_row["top1_candidate_uri"] = ""
            detail_row["top1_candidate_label"] = ""
            detail_row["top1_score"] = ""
            detail_row["top1_rank"] = ""

        details.append(detail_row)

    metrics = []

    for k in TOP_K_VALUES:
        metrics.append(
            {
                "experiment_name": EXPERIMENT_NAME,
                "metric": f"Precision@{k}",
                "value": round(hit_sums[k] / total_references, 6),
            }
        )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "metric": "MRR",
            "value": round(reciprocal_rank_sum / total_references, 6),
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "metric": "Coverage",
            "value": round((total_references - missing_predictions) / total_references, 6),
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "metric": "Evaluated references",
            "value": total_references,
        }
    )

    metrics.append(
        {
            "experiment_name": EXPERIMENT_NAME,
            "metric": "Missing predictions",
            "value": missing_predictions,
        }
    )

    return metrics, details


# ============================================================
# MANIFESTO
# ============================================================

def save_manifest(
    ranked_rows: list[dict],
    best_rows: list[dict],
    gold: dict,
    metrics: list[dict],
    details: list[dict],
) -> None:
    with open(EVALUATION_MANIFEST_FILE, "w", encoding=CSV_ENCODING) as file:
        file.write("DBpedia Alignment Evaluation Manifest\n")
        file.write("=====================================\n\n")
        file.write(f"Generated at: {datetime.now()}\n")
        file.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        file.write(f"Ranked alignments file: {RANKED_ALIGNMENTS_FILE}\n")
        file.write(f"Best alignments file: {BEST_ALIGNMENTS_FILE}\n")
        file.write(f"Gold standard file: {GOLD_STANDARD_FILE}\n")
        file.write(f"Metrics output file: {EVALUATION_METRICS_FILE}\n")
        file.write(f"Details output file: {EVALUATION_DETAILS_FILE}\n\n")

        file.write(f"Ranked rows: {len(ranked_rows)}\n")
        file.write(f"Best rows: {len(best_rows)}\n")
        file.write(f"Gold references: {len(gold)}\n")
        file.write(f"Gold expected mappings: {sum(len(v) for v in gold.values())}\n")
        file.write(f"Metric rows: {len(metrics)}\n")
        file.write(f"Detail rows: {len(details)}\n\n")

        file.write("Metrics:\n")
        for row in metrics:
            file.write(f"  {row['metric']}: {row['value']}\n")

    print(f"[OK] Manifesto salvo: {EVALUATION_MANIFEST_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("DBpedia Alignment Evaluation - Experiment-aware Version")
    print("=" * 80)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Ranked alignments: {RANKED_ALIGNMENTS_FILE}")
    print(f"Best alignments: {BEST_ALIGNMENTS_FILE}")
    print(f"Gold standard: {GOLD_STANDARD_FILE}")
    print(f"Metrics output: {EVALUATION_METRICS_FILE}")
    print("=" * 80)

    try:
        ensure_directories()

        ranked_rows = read_csv(RANKED_ALIGNMENTS_FILE)
        best_rows = read_csv(BEST_ALIGNMENTS_FILE)
        gold = load_gold_standard()

        metrics, details = evaluate_ranked_alignments(
            ranked_rows=ranked_rows,
            gold=gold,
        )

        write_csv(metrics, EVALUATION_METRICS_FILE)
        write_csv(details, EVALUATION_DETAILS_FILE)

        save_manifest(
            ranked_rows=ranked_rows,
            best_rows=best_rows,
            gold=gold,
            metrics=metrics,
            details=details,
        )

        print("\n" + "=" * 80)
        print("RESUMO DA AVALIAÇÃO")
        print("=" * 80)

        for row in metrics:
            print(f"{row['metric']}: {row['value']}")

        print("\n[OK] Avaliação concluída com sucesso.")

    except Exception as error:
        print(f"\n[ERRO] Falha na avaliação: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()