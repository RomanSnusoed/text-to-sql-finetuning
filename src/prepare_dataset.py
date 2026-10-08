import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Spider Text-to-SQL datasets."
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=6500,
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/v2-full"),
    )

    return parser.parse_args()


def load_schema_map():
    """Load Spider database schemas and map them by database ID."""
    schemas = load_dataset(
        "richardr1126/spider-schema",
        split="train",
    )

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
    """Convert a Spider example into schema-grounded prompt/completion format."""

    db_id = example["db_id"]

    if db_id not in schema_map:
        raise KeyError(f"Schema not found for database: {db_id}")

    schema = schema_map[db_id]

    prompt = (
        "You are an expert Text-to-SQL system for SQLite.\n"
        "Convert the user's question into exactly one executable SQLite query.\n\n"

        "Rules:\n"
        "- Use ONLY tables and columns that appear in the provided database schema.\n"
        "- Never invent table names or column names.\n"
        "- Respect the relationships and structure shown in the schema.\n"
        "- Use valid SQLite syntax.\n"
        "- Return exactly one SQL query.\n"
        "- Return SQL only: no Markdown, explanations, comments, or code fences.\n\n"

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
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Spider...")
    spider = load_dataset("xlangai/spider")

    print("Loading database schemas...")
    schema_map = load_schema_map()

    print(f"Available schemas: {len(schema_map)}")
    print(f"Original train examples: {len(spider['train'])}")
    print(f"Original validation examples: {len(spider['validation'])}")

    shuffled_train = spider["train"].shuffle(seed=SEED)

    requested_total = args.train_size + args.val_size

    if requested_total > len(shuffled_train):
        raise ValueError(
            f"Requested {requested_total} train+validation examples, "
            f"but Spider train contains only {len(shuffled_train)}."
        )

    train_raw = shuffled_train.select(
        range(args.train_size)
    )

    val_raw = shuffled_train.select(
        range(
            args.train_size,
            args.train_size + args.val_size,
        )
    )

    test_raw = spider["validation"].select(
        range(
            min(
                args.test_size,
                len(spider["validation"]),
            )
        )
    )

    train = convert_split(train_raw, schema_map)
    validation = convert_split(val_raw, schema_map)
    test = convert_split(test_raw, schema_map)

    train.to_json(
        output_dir / "train.jsonl",
        orient="records",
        lines=True,
    )

    validation.to_json(
        output_dir / "validation.jsonl",
        orient="records",
        lines=True,
    )

    test.to_json(
        output_dir / "test.jsonl",
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