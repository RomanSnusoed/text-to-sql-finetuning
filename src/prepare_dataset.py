from pathlib import Path

from datasets import Dataset, load_dataset


SEED = 42

# На первом проходе намеренно используем subset.
# Когда pipeline заработает, размер можно увеличить.
TRAIN_SIZE = 1200
VAL_SIZE = 150
TEST_SIZE = 250

OUTPUT_DIR = Path("data/processed")


def load_schema_map():
    """Load Spider database schemas and map them by database ID."""
    schemas = load_dataset(
        "richardr1126/spider-schema",
        split="train",
    )

    # Dataset currently calls the schema field something like
    # "Schema (values (type))". Find it defensively.
    schema_column = next(
        column
        for column in schemas.column_names
        if column.lower().startswith("schema")
    )

    schema_map = {}

    for row in schemas:
        schema_map[row["db_id"]] = row[schema_column]

    return schema_map


def format_example(example, schema_map):
    """Convert a Spider example into prompt/completion format."""

    db_id = example["db_id"]

    if db_id not in schema_map:
        raise KeyError(f"Schema not found for database: {db_id}")

    schema = schema_map[db_id]

    prompt = (
        "You are a Text-to-SQL assistant.\n"
        "Generate one valid SQL query for the given database schema "
        "and question.\n"
        "Return SQL only. Do not explain the answer.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Question:\n{example['question']}\n\n"
        "SQL:\n"
    )

    return {
        "db_id": db_id,
        "question": example["question"],
        "schema": schema,
        "prompt": prompt,
        "completion": example["query"].strip(),
    }


def convert_split(split, schema_map):
    rows = [
        format_example(example, schema_map)
        for example in split
    ]

    return Dataset.from_list(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Spider...")
    spider = load_dataset("xlangai/spider")

    print("Loading database schemas...")
    schema_map = load_schema_map()

    print(f"Available schemas: {len(schema_map)}")
    print(f"Original train examples: {len(spider['train'])}")
    print(f"Original validation examples: {len(spider['validation'])}")

    # Shuffle only the original training split.
    shuffled_train = spider["train"].shuffle(seed=SEED)

    train_raw = shuffled_train.select(range(TRAIN_SIZE))

    val_raw = shuffled_train.select(
        range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)
    )

    # Spider's official held-out validation databases become our test set.
    test_raw = spider["validation"].select(
        range(min(TEST_SIZE, len(spider["validation"])))
    )

    train = convert_split(train_raw, schema_map)
    validation = convert_split(val_raw, schema_map)
    test = convert_split(test_raw, schema_map)

    train.to_json(
        OUTPUT_DIR / "train.jsonl",
        orient="records",
        lines=True,
    )

    validation.to_json(
        OUTPUT_DIR / "validation.jsonl",
        orient="records",
        lines=True,
    )

    test.to_json(
        OUTPUT_DIR / "test.jsonl",
        orient="records",
        lines=True,
    )

    print()
    print("Dataset prepared successfully.")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print("\nExample:")
    print("-" * 80)
    print(train[0]["prompt"])
    print(train[0]["completion"])


if __name__ == "__main__":
    main()