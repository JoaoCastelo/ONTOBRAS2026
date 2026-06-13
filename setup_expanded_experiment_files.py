from pathlib import Path
import shutil


PROJECT_ROOT = Path("C:/Users/user/Desktop/CodigoDoutorado")

SRC_DIR = PROJECT_ROOT / "experiments" / "src"

REFERENCE_SOURCE_FILE = SRC_DIR / "reference_terms_expanded.csv"
GOLD_SOURCE_FILE = SRC_DIR / "gold_standard_expanded.csv"

EXPANDED_REFERENCE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "reference_ontologies"
    / "expanded"
)

EXPANDED_RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "expanded"
)

EXPANDED_ALIGNMENTS_DIR = EXPANDED_RESULTS_DIR / "alignments"
EXPANDED_METRICS_DIR = EXPANDED_RESULTS_DIR / "metrics"

REFERENCE_TARGET_FILE = EXPANDED_REFERENCE_DIR / "reference_terms_expanded.csv"
GOLD_TARGET_FILE = EXPANDED_REFERENCE_DIR / "gold_standard_expanded.csv"


def ensure_directories() -> None:
    directories = [
        EXPANDED_REFERENCE_DIR,
        EXPANDED_RESULTS_DIR,
        EXPANDED_ALIGNMENTS_DIR,
        EXPANDED_METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Pasta verificada/criada: {directory}")


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Arquivo de origem não encontrado: {source}")

    shutil.copy2(source, target)
    print(f"[OK] Arquivo copiado:")
    print(f"     Origem : {source}")
    print(f"     Destino: {target}")


def main() -> None:
    print("=" * 80)
    print("Setup do experimento expandido")
    print("=" * 80)

    ensure_directories()

    copy_file(
        source=REFERENCE_SOURCE_FILE,
        target=REFERENCE_TARGET_FILE,
    )

    copy_file(
        source=GOLD_SOURCE_FILE,
        target=GOLD_TARGET_FILE,
    )

    print("\n" + "=" * 80)
    print("[OK] Experimento expandido configurado com sucesso.")
    print("=" * 80)

    print("\nArquivos instalados:")
    print(f"- {REFERENCE_TARGET_FILE}")
    print(f"- {GOLD_TARGET_FILE}")

    print("\nPastas de resultado criadas:")
    print(f"- {EXPANDED_ALIGNMENTS_DIR}")
    print(f"- {EXPANDED_METRICS_DIR}")


if __name__ == "__main__":
    main()