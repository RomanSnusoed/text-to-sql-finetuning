import argparse
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm


DEFAULT_DATABASE_ROOT = Path(
    "data/spider/spider_data/database"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Text-to-SQL predictions by SQLite execution."
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--database-root",
        type=Path,
        default=DEFAULT_DATABASE_ROOT,
    )

    return parser.parse_args()


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 6)

    return value


def normalize_rows(
    rows,
    order_matters,
):
    normalized = [
        tuple(
            normalize_value(value)
            for value in row
        )
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
        return (
            None,
            f"Database not found: {db_path}",
        )

    connection = None

    try:
        absolute_path = (
            db_path.resolve().as_posix()
        )

        connection = sqlite3.connect(
            f"file:{absolute_path}?mode=ro",
            uri=True,
            timeout=5,
        )

        cursor = connection.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        return rows, None

    except Exception as error:
        return None, str(error)

    finally:
        if connection is not None:
            connection.close()


def compare_execution(
    db_path,
    reference_sql,
    predicted_sql,
):
    reference_rows, reference_error = (
        execute_sql(
            db_path,
            reference_sql,
        )
    )

    if reference_error:
        return (
            None,
            f"REFERENCE ERROR: {reference_error}",
        )

    predicted_rows, predicted_error = (
        execute_sql(
            db_path,
            predicted_sql,
        )
    )

    if predicted_error:
        return False, predicted_error

    order_matters = (
        "order by"
        in reference_sql.lower()
    )

    reference_rows = normalize_rows(
        reference_rows,
        order_matters,
    )

    predicted_rows = normalize_rows(
        predicted_rows,
        order_matters,
    )

    return (
        reference_rows == predicted_rows,
        None,
    )


def save_json(path, data):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = parse_args()

    results_dir = (
        Path("results")
        / args.experiment_id
    )

    if args.predictions_file:
        predictions_file = (
            args.predictions_file
        )
    else:
        predictions_file = (
            results_dir
            / "predictions.jsonl"
        )

    metrics_file = (
        results_dir
        / "execution_metrics.json"
    )

    if not predictions_file.exists():
        raise FileNotFoundError(
            predictions_file
        )

    with predictions_file.open(
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

    reference_errors = 0
    evaluated = 0

    changed_examples = []

    print("=" * 80)
    print("SQL EXECUTION EVALUATION")
    print("=" * 80)

    print(
        "Experiment:",
        args.experiment_id,
    )

    print("Examples:", len(rows))

    for row in tqdm(rows):
        db_id = row["db_id"]

        db_path = (
            args.database_root
            / db_id
            / f"{db_id}.sqlite"
        )

        base_match, base_error = (
            compare_execution(
                db_path,
                row["reference_sql"],
                row["base_sql"],
            )
        )

        tuned_match, tuned_error = (
            compare_execution(
                db_path,
                row["reference_sql"],
                row["finetuned_sql"],
            )
        )

        if (
            base_match is None
            or tuned_match is None
        ):
            reference_errors += 1
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
            changed_examples.append(
                {
                    "db_id": db_id,
                    "question": (
                        row["question"]
                    ),
                    "reference_sql": (
                        row[
                            "reference_sql"
                        ]
                    ),
                    "base_sql": (
                        row["base_sql"]
                    ),
                    "finetuned_sql": (
                        row[
                            "finetuned_sql"
                        ]
                    ),
                    "base_correct": (
                        base_match
                    ),
                    "finetuned_correct": (
                        tuned_match
                    ),
                    "base_error": (
                        base_error
                    ),
                    "finetuned_error": (
                        tuned_error
                    ),
                }
            )

    if evaluated == 0:
        raise RuntimeError(
            "No examples could be evaluated."
        )

    base_accuracy = (
        100
        * base_correct
        / evaluated
    )

    tuned_accuracy = (
        100
        * tuned_correct
        / evaluated
    )

    absolute_improvement = (
        tuned_accuracy
        - base_accuracy
    )

    if base_accuracy > 0:
        relative_improvement = (
            100
            * absolute_improvement
            / base_accuracy
        )
    else:
        relative_improvement = None

    if base_errors > 0:
        error_reduction = (
            100
            * (base_errors - tuned_errors)
            / base_errors
        )
    else:
        error_reduction = None

    metrics = {
        "experiment_id": (
            args.experiment_id
        ),
        "evaluated_examples": (
            evaluated
        ),
        "reference_errors": (
            reference_errors
        ),
        "base_execution_accuracy": (
            base_accuracy / 100
        ),
        "finetuned_execution_accuracy": (
            tuned_accuracy / 100
        ),
        "absolute_improvement_percentage_points": (
            absolute_improvement
        ),
        "relative_improvement_percent": (
            relative_improvement
        ),
        "base_correct": (
            base_correct
        ),
        "finetuned_correct": (
            tuned_correct
        ),
        "base_execution_errors": (
            base_errors
        ),
        "finetuned_execution_errors": (
            tuned_errors
        ),
        "execution_error_reduction_percent": (
            error_reduction
        ),
    }

    save_json(
        metrics_file,
        metrics,
    )

    changed_file = (
        results_dir
        / "changed_outcomes.jsonl"
    )

    with changed_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in changed_examples:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

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
        f"{'Correct queries':<30}"
        f"{base_correct:>12}"
        f"{tuned_correct:>15}"
    )

    print(
        f"{'Execution errors':<30}"
        f"{base_errors:>12}"
        f"{tuned_errors:>15}"
    )

    print()
    print(
        "Absolute improvement:",
        f"{absolute_improvement:+.1f} pp",
    )

    if relative_improvement is not None:
        print(
            "Relative improvement:",
            f"{relative_improvement:+.1f}%",
        )

    print("Evaluated:", evaluated)

    print(
        "\nExecution metrics saved to:"
    )
    print(metrics_file)

    print(
        "\nChanged outcomes saved to:"
    )
    print(changed_file)


if __name__ == "__main__":
    main()