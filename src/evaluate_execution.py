import json
import sqlite3
from pathlib import Path

from tqdm import tqdm


RESULTS_FILE = Path("results/model_comparison.jsonl")
DATABASE_ROOT = Path("data/spider/spider_data/database")


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 6)

    return value


def normalize_rows(rows, order_matters):
    normalized = [
        tuple(normalize_value(value) for value in row)
        for row in rows
    ]

    if not order_matters:
        normalized = sorted(
            normalized,
            key=lambda row: repr(row),
        )

    return normalized


def execute_sql(db_path, sql):
    if not db_path.exists():
        return None, f"Database not found: {db_path}"

    try:
        absolute_path = db_path.resolve().as_posix()

        connection = sqlite3.connect(
            f"file:{absolute_path}?mode=ro",
            uri=True,
            timeout=5,
        )

        cursor = connection.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        connection.close()

        return rows, None

    except Exception as error:
        return None, str(error)


def compare_execution(db_path, reference_sql, predicted_sql):
    reference_rows, reference_error = execute_sql(
        db_path,
        reference_sql,
    )

    if reference_error:
        return None, f"REFERENCE ERROR: {reference_error}"

    predicted_rows, predicted_error = execute_sql(
        db_path,
        predicted_sql,
    )

    if predicted_error:
        return False, predicted_error

    # If the gold query explicitly orders results,
    # ordering is part of the intended answer.
    order_matters = "order by" in reference_sql.lower()

    reference_rows = normalize_rows(
        reference_rows,
        order_matters,
    )

    predicted_rows = normalize_rows(
        predicted_rows,
        order_matters,
    )

    return reference_rows == predicted_rows, None


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(RESULTS_FILE)

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    base_correct = 0
    tuned_correct = 0

    base_errors = 0
    tuned_errors = 0

    evaluated = 0

    failed_examples = []

    print("=" * 80)
    print("SQL EXECUTION EVALUATION")
    print("=" * 80)
    print("Examples:", len(rows))

    for row in tqdm(rows):
        db_id = row["db_id"]

        db_path = (
            DATABASE_ROOT
            / db_id
            / f"{db_id}.sqlite"
        )

        base_match, base_error = compare_execution(
            db_path,
            row["reference_sql"],
            row["base_sql"],
        )

        tuned_match, tuned_error = compare_execution(
            db_path,
            row["reference_sql"],
            row["finetuned_sql"],
        )

        # If the reference itself cannot execute,
        # don't count the example.
        if base_match is None or tuned_match is None:
            continue

        evaluated += 1

        if base_match:
            base_correct += 1

        if tuned_match:
            tuned_correct += 1

        if base_error:
            base_errors += 1

        if tuned_error:
            tuned_errors += 1

        if base_match != tuned_match:
            failed_examples.append(
                {
                    "question": row["question"],
                    "reference": row["reference_sql"],
                    "base": row["base_sql"],
                    "finetuned": row["finetuned_sql"],
                    "base_correct": base_match,
                    "finetuned_correct": tuned_match,
                    "base_error": base_error,
                    "finetuned_error": tuned_error,
                }
            )

    if evaluated == 0:
        raise RuntimeError("No examples could be evaluated.")

    base_accuracy = 100 * base_correct / evaluated
    tuned_accuracy = 100 * tuned_correct / evaluated

    print()
    print("=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'Base':>12}"
        f"{'Fine-tuned':>15}"
    )

    print("-" * 57)

    print(
        f"{'Execution accuracy':<30}"
        f"{base_accuracy:>11.1f}%"
        f"{tuned_accuracy:>14.1f}%"
    )

    print(
        f"{'Execution errors':<30}"
        f"{base_errors:>12}"
        f"{tuned_errors:>15}"
    )

    print()
    print("Evaluated:", evaluated)

    print("\nChanged outcomes:")
    print("=" * 80)

    for example in failed_examples[:10]:
        print("\nQUESTION:")
        print(example["question"])

        print("\nREFERENCE:")
        print(example["reference"])

        print(
            "\nBASE:",
            "CORRECT" if example["base_correct"] else "WRONG",
        )
        print(example["base"])

        if example["base_error"]:
            print("Error:", example["base_error"])

        print(
            "\nFINE-TUNED:",
            "CORRECT" if example["finetuned_correct"] else "WRONG",
        )
        print(example["finetuned"])

        if example["finetuned_error"]:
            print("Error:", example["finetuned_error"])

        print("-" * 80)


if __name__ == "__main__":
    main()