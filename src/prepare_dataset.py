import argparse
import json
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
        default=Path("data/v5-structured-schema"),
    )

    parser.add_argument(
        "--tables-file",
        type=Path,
        default=Path("data/spider/spider_data/tables.json"),
    )

    return parser.parse_args()


def load_schema_map(tables_file):
    """Build structured schemas from the official Spider tables.json."""

    with tables_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        databases = json.load(file)

    schema_map = {}

    for database in databases:
        db_id = database["db_id"]

        table_names = database["table_names_original"]
        column_names = database["column_names_original"]
        column_types = database["column_types"]

        primary_keys = set(database["primary_keys"])
        foreign_keys = database["foreign_keys"]

        columns_by_table = {
            table_index: []
            for table_index in range(len(table_names))
        }

        for column_index, column_info in enumerate(column_names):
            table_index, column_name = column_info

            # Spider contains the special "*"
            # column with table index -1.
            if table_index == -1:
                continue

            column_type = column_types[column_index]

            annotations = []

            if column_index in primary_keys:
                annotations.append("PRIMARY KEY")

            annotation_text = ""

            if annotations:
                annotation_text = (
                    " [" + ", ".join(annotations) + "]"
                )

            columns_by_table[table_index].append(
                f"  - {column_name} ({column_type})"
                f"{annotation_text}"
            )

        schema_lines = []

        for table_index, table_name in enumerate(table_names):
            schema_lines.append(
                f"TABLE {table_name}"
            )

            schema_lines.extend(
                columns_by_table[table_index]
            )

            schema_lines.append("")

        if foreign_keys:
            schema_lines.append("RELATIONSHIPS")

            for source_index, target_index in foreign_keys:
                source_table_index, source_column = (
                    column_names[source_index]
                )

                target_table_index, target_column = (
                    column_names[target_index]
                )

                source_table = table_names[
                    source_table_index
                ]

                target_table = table_names[
                    target_table_index
                ]

                schema_lines.append(
                    f"  - {source_table}.{source_column}"
                    f" = "
                    f"{target_table}.{target_column}"
                )

        schema_map[db_id] = "\n".join(
            schema_lines
        ).strip()

    return schema_map


def format_example(example, schema_map):
    """Convert a Spider example into prompt/completion format."""

    db_id = example["db_id"]

    if db_id not in schema_map:
        raise KeyError(
            f"Schema not found for database: {db_id}"
        )

    schema = schema_map[db_id]

    # IMPORTANT:
    # Keep the instruction prompt identical to v3.
    # Only the schema representation changes in v5.
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
        format_example(
            example,
            schema_map,
        )
        for example in split
    ]

    return Dataset.from_list(rows)


def main():
    args = parse_args()

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.tables_file.exists():
        raise FileNotFoundError(
            f"Spider tables.json not found: "
            f"{args.tables_file}"
        )

    print("Loading Spider...")
    spider = load_dataset(
        "xlangai/spider"
    )

    print(
        "Loading structured database schemas..."
    )

    schema_map = load_schema_map(
        args.tables_file
    )

    print(
        f"Available schemas: {len(schema_map)}"
    )

    print(
        f"Original train examples: "
        f"{len(spider['train'])}"
    )

    print(
        f"Original validation examples: "
        f"{len(spider['validation'])}"
    )

    shuffled_train = spider[
        "train"
    ].shuffle(
        seed=SEED
    )

    requested_total = (
        args.train_size
        + args.val_size
    )

    if requested_total > len(shuffled_train):
        raise ValueError(
            f"Requested {requested_total} "
            f"train+validation examples, "
            f"but Spider train contains only "
            f"{len(shuffled_train)}."
        )

    train_raw = shuffled_train.select(
        range(args.train_size)
    )

    val_raw = shuffled_train.select(
        range(
            args.train_size,
            args.train_size
            + args.val_size,
        )
    )

    test_raw = spider[
        "validation"
    ].select(
        range(
            min(
                args.test_size,
                len(
                    spider[
                        "validation"
                    ]
                ),
            )
        )
    )

    train = convert_split(
        train_raw,
        schema_map,
    )

    validation = convert_split(
        val_raw,
        schema_map,
    )

    test = convert_split(
        test_raw,
        schema_map,
    )

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
    print(
        "Dataset prepared successfully."
    )

    print(
        f"Train:      {len(train)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    print(
        f"Test:       {len(test)}"
    )

    print("\nExample:")
    print("-" * 80)
    print(train[0]["prompt"])
    print(train[0]["completion"])


if __name__ == "__main__":
    main()