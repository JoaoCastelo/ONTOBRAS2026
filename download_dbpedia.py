# Script for downloading DBpedia dumps

"""
download_dbpedia.py

Script para baixar dumps essenciais da DBpedia para experimentos de
alinhamento de ontologias usando Linked Open Data.

Este script baixa arquivos RDF compactados (.ttl.bz2) para a pasta:

experiments/data/dbpedia_raw
"""

from pathlib import Path
from urllib.parse import urlparse
import requests
import sys
import time


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PROJECT_ROOT = Path(".")
DBPEDIA_RAW_DIR = PROJECT_ROOT / "experiments" / "data" / "dbpedia_raw"

# Versão fixa para garantir reprodutibilidade do experimento.
# Depois podemos trocar para uma versão mais recente via DBpedia Databus.
DBPEDIA_VERSION = "2020.10.01"
LANGUAGE = "en"

# Tamanho do bloco de download em bytes.
CHUNK_SIZE = 1024 * 1024  # 1 MB

# Número de tentativas em caso de falha.
MAX_RETRIES = 3

# Tempo de espera entre tentativas.
RETRY_SLEEP_SECONDS = 5


# ============================================================
# DUMPS SELECIONADOS
# ============================================================

"""
Dumps escolhidos para a primeira validação:

1. labels
   - usado para similaridade lexical e geração de candidatos.

2. instance-types
   - usado para saber a qual classe da DBpedia cada recurso pertence.

3. mappingbased-literals
   - usado para extrair valores literais associados às entidades.

4. mappingbased-objects
   - usado para extrair relações entre recursos da DBpedia.

5. infobox-properties
   - usado como fonte complementar de propriedades extraídas da Wikipedia.

6. infobox-property-definitions
   - usado para descrição/definição das propriedades de infobox.

7. commons-sameas-links
   - usado opcionalmente para enriquecer links equivalentes.
"""

DBPEDIA_DUMPS = {
    "labels": (
        f"https://downloads.dbpedia.org/repo/dbpedia/generic/labels/"
        f"{DBPEDIA_VERSION}/labels_lang={LANGUAGE}.ttl.bz2"
    ),

    "instance_types": (
        f"https://downloads.dbpedia.org/repo/dbpedia/mappings/instance-types/"
        f"{DBPEDIA_VERSION}/instance-types_lang={LANGUAGE}_specific.ttl.bz2"
    ),

    "mappingbased_literals": (
        f"https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-literals/"
        f"{DBPEDIA_VERSION}/mappingbased-literals_lang={LANGUAGE}.ttl.bz2"
    ),

    "mappingbased_objects": (
        f"https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-objects/"
        f"{DBPEDIA_VERSION}/mappingbased-objects_lang={LANGUAGE}.ttl.bz2"
    ),

    "infobox_properties": (
        f"https://downloads.dbpedia.org/repo/dbpedia/generic/infobox-properties/"
        f"{DBPEDIA_VERSION}/infobox-properties_lang={LANGUAGE}.ttl.bz2"
    ),

    "infobox_property_definitions": (
        f"https://downloads.dbpedia.org/repo/dbpedia/generic/infobox-property-definitions/"
        f"{DBPEDIA_VERSION}/infobox-property-definitions_lang={LANGUAGE}.ttl.bz2"
    ),

    "commons_sameas_links": (
        f"https://downloads.dbpedia.org/repo/dbpedia/generic/commons-sameas-links/"
        f"{DBPEDIA_VERSION}/commons-sameas-links_lang={LANGUAGE}.ttl.bz2"
    ),
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def create_directories() -> None:
    """
    Cria a pasta de destino dos dumps, se ela ainda não existir.
    """
    DBPEDIA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Pasta de destino verificada: {DBPEDIA_RAW_DIR}")


def get_filename_from_url(url: str) -> str:
    """
    Extrai o nome do arquivo a partir da URL.
    """
    parsed_url = urlparse(url)
    return Path(parsed_url.path).name


def get_remote_file_size(url: str) -> int | None:
    """
    Obtém o tamanho remoto do arquivo, quando o servidor informa Content-Length.
    Retorna None se não for possível obter o tamanho.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)

        if response.status_code >= 400:
            print(f"[AVISO] HEAD retornou status {response.status_code} para: {url}")
            return None

        content_length = response.headers.get("Content-Length")

        if content_length is None:
            return None

        return int(content_length)

    except requests.RequestException as error:
        print(f"[AVISO] Não foi possível obter tamanho remoto: {error}")
        return None


def format_size(size_bytes: int | None) -> str:
    """
    Formata tamanho em bytes para MB/GB.
    """
    if size_bytes is None:
        return "tamanho desconhecido"

    size_mb = size_bytes / (1024 * 1024)

    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"

    return f"{size_mb:.2f} MB"


def should_skip_file(destination: Path, remote_size: int | None) -> bool:
    """
    Decide se o arquivo já existe e pode ser pulado.

    Se o tamanho remoto for conhecido, compara com o tamanho local.
    Se o tamanho remoto for desconhecido, apenas verifica se o arquivo existe.
    """
    if not destination.exists():
        return False

    local_size = destination.stat().st_size

    if remote_size is None:
        print(f"[SKIP] Arquivo já existe: {destination.name}")
        return True

    if local_size == remote_size:
        print(f"[SKIP] Arquivo já baixado completamente: {destination.name}")
        return True

    print(
        f"[AVISO] Arquivo local incompleto ou diferente: {destination.name} "
        f"local={format_size(local_size)} remoto={format_size(remote_size)}"
    )
    return False


def download_file(url: str, destination: Path) -> bool:
    """
    Baixa um arquivo com suporte simples a retomada parcial.

    Retorna True se o download foi concluído com sucesso.
    Retorna False se houve erro.
    """
    remote_size = get_remote_file_size(url)

    print("\n" + "=" * 80)
    print(f"[INFO] URL: {url}")
    print(f"[INFO] Destino: {destination}")
    print(f"[INFO] Tamanho remoto: {format_size(remote_size)}")

    if should_skip_file(destination, remote_size):
        return True

    headers = {}

    # Retomada parcial: se o arquivo já existe, tenta continuar de onde parou.
    existing_size = destination.stat().st_size if destination.exists() else 0

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"[INFO] Tentando retomar download a partir de {format_size(existing_size)}")

    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if response.status_code in (403, 404):
                print(f"[ERRO] Arquivo não encontrado ou acesso negado: {response.status_code}")
                return False

            if response.status_code not in (200, 206):
                print(f"[ERRO] Status HTTP inesperado: {response.status_code}")
                return False

            mode = "ab" if response.status_code == 206 else "wb"

            downloaded = existing_size if mode == "ab" else 0
            last_print_time = time.time()

            with open(destination, mode) as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_print_time >= 2:
                            if remote_size:
                                percent = (downloaded / remote_size) * 100
                                print(
                                    f"[PROGRESSO] {destination.name}: "
                                    f"{format_size(downloaded)} / {format_size(remote_size)} "
                                    f"({percent:.2f}%)"
                                )
                            else:
                                print(
                                    f"[PROGRESSO] {destination.name}: "
                                    f"{format_size(downloaded)} baixados"
                                )

                            last_print_time = now

        final_size = destination.stat().st_size

        if remote_size is not None and final_size != remote_size:
            print(
                f"[ERRO] Tamanho final diferente do esperado em {destination.name}: "
                f"final={format_size(final_size)} esperado={format_size(remote_size)}"
            )
            return False

        print(f"[OK] Download concluído: {destination.name}")
        return True

    except requests.RequestException as error:
        print(f"[ERRO] Falha de rede ao baixar {url}: {error}")
        return False

    except OSError as error:
        print(f"[ERRO] Falha ao salvar arquivo {destination}: {error}")
        return False


def download_with_retries(name: str, url: str) -> bool:
    """
    Executa o download com múltiplas tentativas.
    """
    filename = get_filename_from_url(url)
    destination = DBPEDIA_RAW_DIR / filename

    print("\n" + "#" * 80)
    print(f"[DUMP] {name}")
    print("#" * 80)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[INFO] Tentativa {attempt}/{MAX_RETRIES}")

        success = download_file(url, destination)

        if success:
            return True

        if attempt < MAX_RETRIES:
            print(f"[INFO] Aguardando {RETRY_SLEEP_SECONDS} segundos antes de tentar novamente...")
            time.sleep(RETRY_SLEEP_SECONDS)

    print(f"[FALHA] Não foi possível baixar o dump: {name}")
    return False


def save_manifest(results: dict[str, bool]) -> None:
    """
    Salva um manifesto simples com o status dos downloads.
    """
    manifest_path = DBPEDIA_RAW_DIR / "download_manifest.txt"

    with open(manifest_path, "w", encoding="utf-8") as file:
        file.write("DBpedia Download Manifest\n")
        file.write("=========================\n\n")
        file.write(f"DBpedia version: {DBPEDIA_VERSION}\n")
        file.write(f"Language: {LANGUAGE}\n")
        file.write(f"Destination: {DBPEDIA_RAW_DIR}\n\n")

        for name, url in DBPEDIA_DUMPS.items():
            status = "OK" if results.get(name, False) else "FAILED"
            filename = get_filename_from_url(url)

            file.write(f"[{status}] {name}\n")
            file.write(f"  file: {filename}\n")
            file.write(f"  url:  {url}\n\n")

    print(f"\n[OK] Manifesto salvo em: {manifest_path}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Função principal do script.
    """
    print("=" * 80)
    print("DBpedia Dump Downloader")
    print("=" * 80)
    print(f"Projeto: {PROJECT_ROOT}")
    print(f"Destino: {DBPEDIA_RAW_DIR}")
    print(f"Versão DBpedia: {DBPEDIA_VERSION}")
    print(f"Idioma: {LANGUAGE}")
    print("=" * 80)

    create_directories()

    results = {}

    for name, url in DBPEDIA_DUMPS.items():
        success = download_with_retries(name, url)
        results[name] = success

    save_manifest(results)

    total = len(results)
    successful = sum(1 for value in results.values() if value)
    failed = total - successful

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Total de dumps: {total}")
    print(f"Downloads concluídos: {successful}")
    print(f"Falhas: {failed}")

    if failed > 0:
        print("\n[AVISO] Alguns arquivos não foram baixados.")
        print("Verifique sua conexão ou se as URLs ainda estão disponíveis.")
        sys.exit(1)

    print("\n[OK] Todos os dumps foram baixados com sucesso.")


if __name__ == "__main__":
    main()