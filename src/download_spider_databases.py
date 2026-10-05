from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download


REPO_ID = "HAL-9001/spider-databases"
FILENAME = "spider_data.zip"
OUTPUT_DIR = Path("data/spider")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Spider SQLite databases...")

    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )

    print("Archive:", archive_path)
    print("Extracting...")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(OUTPUT_DIR)

    database_dir = OUTPUT_DIR / "spider_data" / "database"

    if not database_dir.exists():
        raise RuntimeError(
            f"Database directory not found: {database_dir}"
        )

    sqlite_files = list(database_dir.rglob("*.sqlite"))

    print()
    print("Done.")
    print("Database directory:", database_dir)
    print("SQLite databases:", len(sqlite_files))


if __name__ == "__main__":
    main()