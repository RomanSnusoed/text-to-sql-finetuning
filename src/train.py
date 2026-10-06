import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Text-to-SQL experiments."
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default="v2-full-context768",
    )

    parser.add_argument(
        "--train-file",
        type=str,
        default="data/v2-full/train.jsonl",
    )

    parser.add_argument(
        "--val-file",
        type=str,
        default="data/v2-full/validation.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/v2-full-context768"),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny 5-step training job.",
    )

    return parser.parse_args()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def make_json_safe(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def main():
    args = parse_args()

    print("=" * 80)
    print("TEXT-TO-SQL QLoRA TRAINING")
    print("=" * 80)

    print("Experiment:", args.experiment_id)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training configuration.")

    print("GPU:", torch.cuda.get_device_name(0))

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print("VRAM:", round(vram_gb, 2), "GB")

    # ------------------------------------------------------------------
    # 1. PATHS
    # ------------------------------------------------------------------

    if args.smoke:
        model_output_dir = Path("models/smoke-test-v2")
    else:
        model_output_dir = args.output_dir

    results_dir = Path("results") / args.experiment_id

    # ------------------------------------------------------------------
    # 2. DATASET
    # ------------------------------------------------------------------

    print("\nLoading datasets...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    full_train_size = len(train_dataset)
    full_val_size = len(eval_dataset)

    if args.smoke:
        print("\nSMOKE TEST MODE")

        train_dataset = train_dataset.select(
            range(min(64, len(train_dataset)))
        )

        eval_dataset = eval_dataset.select(
            range(min(16, len(eval_dataset)))
        )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(eval_dataset))

    # ------------------------------------------------------------------
    # 3. TOKENIZER
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 4. QUANTIZATION
    # ------------------------------------------------------------------

    print("\nPreparing 4-bit NF4 quantization...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------
    # 5. LORA
    # ------------------------------------------------------------------

    print("Preparing LoRA configuration...")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # ------------------------------------------------------------------
    # 6. TRAINING CONFIG
    # ------------------------------------------------------------------

    if args.smoke:
        epochs = 1
        max_steps = 5
        warmup_steps = 1
    else:
        epochs = args.epochs
        max_steps = -1
        warmup_steps = 15

    training_config = SFTConfig(
        output_dir=str(model_output_dir),

        num_train_epochs=epochs,
        max_steps=max_steps,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        max_length=args.max_length,

        completion_only_loss=True,

        logging_steps=5,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",

        eos_token="<|im_end|>",

        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 7. TRAINER
    # ------------------------------------------------------------------

    print("\nCreating SFTTrainer...")

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_config,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        quantization_config=quantization_config,
        peft_config=peft_config,
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
    )

    trainable_percent = (
        100 * trainable_params / total_params
    )

    print("\nTrainable parameters:")
    trainer.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 8. EXPERIMENT CONFIG
    # ------------------------------------------------------------------

    if not args.smoke:
        experiment_config = {
            "experiment_id": args.experiment_id,
            "model": MODEL_NAME,
            "method": "4-bit QLoRA SFT",
            "dataset": "Spider",

            "train_file": args.train_file,
            "validation_file": args.val_file,

            "training_examples": full_train_size,
            "validation_examples": full_val_size,

            "max_length": args.max_length,
            "epochs": args.epochs,

            "batch_size": 1,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),

            "learning_rate": args.learning_rate,

            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,

            "quantization": "4-bit NF4",
            "compute_dtype": "bfloat16",

            "gradient_checkpointing": True,

            "seed": args.seed,

            "model_output_dir": str(
                model_output_dir
            ),
        }

        save_json(
            results_dir / "config.json",
            experiment_config,
        )

        print(
            "\nExperiment config saved to:",
            results_dir / "config.json",
        )

    # ------------------------------------------------------------------
    # 9. TRAIN
    # ------------------------------------------------------------------

    print("\nStarting training...\n")

    train_result = trainer.train()

    print("\nTraining finished.")

    # ------------------------------------------------------------------
    # 10. FINAL EVALUATION
    # ------------------------------------------------------------------

    print("\nEvaluating validation loss...")

    eval_results = trainer.evaluate()

    print("\nEvaluation results:")

    for key, value in eval_results.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------------------
    # 11. SAVE MODEL + METRICS
    # ------------------------------------------------------------------

    if not args.smoke:
        print("\nSaving LoRA adapter...")

        trainer.save_model(
            str(model_output_dir)
        )

        tokenizer.save_pretrained(
            str(model_output_dir)
        )

        training_metrics = {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_parameter_percent": (
                trainable_percent
            ),
        }

        for key, value in train_result.metrics.items():
            training_metrics[key] = (
                make_json_safe(value)
            )

        for key, value in eval_results.items():
            training_metrics[key] = (
                make_json_safe(value)
            )

        save_json(
            results_dir / "training_metrics.json",
            training_metrics,
        )

        print(
            "\nTraining metrics saved to:",
            results_dir / "training_metrics.json",
        )

        print("\nModel saved to:")
        print(model_output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()