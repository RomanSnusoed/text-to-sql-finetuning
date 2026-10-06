import argparse
import json
from pathlib import Path

import sqlglot
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned Text-to-SQL models."
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    return parser.parse_args()


def clean_sql(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("sql\n"):
            text = text[4:].strip()

    return text


def canonical_sql(sql):
    try:
        parsed = sqlglot.parse_one(
            sql,
            read="sqlite",
        )

        return parsed.sql(
            dialect="sqlite"
        )

    except Exception:
        return None


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    return clean_sql(
        tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
    )


def evaluate_outputs(rows, field):
    valid = 0
    canonical_matches = 0

    for row in rows:
        predicted = canonical_sql(
            row[field]
        )

        reference = canonical_sql(
            row["reference_sql"]
        )

        if predicted is not None:
            valid += 1

        if (
            predicted is not None
            and reference is not None
            and predicted == reference
        ):
            canonical_matches += 1

    total = len(rows)

    return {
        "valid_sql": valid,
        "valid_sql_pct": 100 * valid / total,
        "canonical_exact": canonical_matches,
        "canonical_exact_pct": (
            100 * canonical_matches / total
        ),
    }


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

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        results_dir
        / "predictions.jsonl"
    )

    metrics_file = (
        results_dir
        / "generation_metrics.json"
    )

    dataset = load_dataset(
        "json",
        data_files=args.test_file,
        split="train",
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(
                min(
                    args.limit,
                    len(dataset),
                )
            )
        )

    print("=" * 80)
    print("BASE vs FINE-TUNED TEXT-TO-SQL")
    print("=" * 80)

    print("Experiment:", args.experiment_id)
    print("Examples:", len(dataset))
    print("Adapter:", args.adapter_path)
    print("Test file:", args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),
            bnb_4bit_use_double_quant=True,
        )
    )

    print("\nLoading base model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
            dtype=torch.bfloat16,
        )
    )

    model.eval()

    results = []

    print(
        "\nGenerating BASE predictions..."
    )

    for example in tqdm(dataset):
        prediction = generate(
            model,
            tokenizer,
            example["prompt"],
            args.max_new_tokens,
        )

        results.append(
            {
                "db_id": example["db_id"],
                "question": (
                    example["question"]
                ),
                "reference_sql": (
                    example["completion"]
                ),
                "base_sql": prediction,
            }
        )

    print(
        "\nAttaching trained LoRA adapter..."
    )

    model = PeftModel.from_pretrained(
        model,
        str(args.adapter_path),
    )

    model.eval()

    print(
        "\nGenerating FINE-TUNED predictions..."
    )

    for index, example in enumerate(
        tqdm(dataset)
    ):
        prediction = generate(
            model,
            tokenizer,
            example["prompt"],
            args.max_new_tokens,
        )

        results[index][
            "finetuned_sql"
        ] = prediction

    with predictions_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in results:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    base_metrics = evaluate_outputs(
        results,
        "base_sql",
    )

    tuned_metrics = evaluate_outputs(
        results,
        "finetuned_sql",
    )

    generation_metrics = {
        "experiment_id": (
            args.experiment_id
        ),
        "evaluated_examples": (
            len(results)
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "base": base_metrics,
        "finetuned": tuned_metrics,
    }

    save_json(
        metrics_file,
        generation_metrics,
    )

    print("\n" + "=" * 80)
    print("GENERATION RESULTS")
    print("=" * 80)

    print(
        f"\n{'Metric':<30}"
        f"{'Base':>12}"
        f"{'Fine-tuned':>15}"
    )

    print("-" * 57)

    print(
        f"{'Valid SQL':<30}"
        f"{base_metrics['valid_sql_pct']:>11.1f}%"
        f"{tuned_metrics['valid_sql_pct']:>14.1f}%"
    )

    print(
        f"{'Canonical exact match':<30}"
        f"{base_metrics['canonical_exact_pct']:>11.1f}%"
        f"{tuned_metrics['canonical_exact_pct']:>14.1f}%"
    )

    print(
        "\nPredictions saved to:"
    )
    print(predictions_file)

    print(
        "\nGeneration metrics saved to:"
    )
    print(metrics_file)


if __name__ == "__main__":
    main()