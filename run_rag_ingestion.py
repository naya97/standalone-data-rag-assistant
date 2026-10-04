import logging
from pathlib import Path

from src.rag.week7.ingest import IngestionError, run_ingestion

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / "week7"
DB_PATH = BASE_DIR / "data" / "db" / "local_storage.sqlite"
LOG_PATH = BASE_DIR / "logs" / "week7" / "ingestion.log"


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        summary = run_ingestion(raw_dir=RAW_DIR, db_path=DB_PATH)
    except IngestionError as exc:
        logger.critical("Ingestion Pipeline Terminated Due To Error: %s", exc)
        raise SystemExit(1) from exc

    print("\n--- Week 7 Ingestion Summary ---")
    print(f"Files found:     {summary['total_files']}")
    print(f"Processed:       {summary['processed']}")
    print(f"Failed:          {summary['failed']}")
    print(f"Total chunks:    {summary['total_chunks']}")
    print(f"By document type: {summary['by_type']}")
    print(f"By source format: {summary['by_format']}")
    if summary["failures"]:
        print("Failures:")
        for f in summary["failures"]:
            print(f"  - {f['file']} [{f['phase']}]: {f['error']}")


if __name__ == "__main__":
    main()
