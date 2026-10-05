import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = Path("models/text-to-sql-qwen-lora")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny training job first to verify the pipeline.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("TEXT-TO-SQL QLoRA TRAINING")
    print("=" * 80)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
        "GB",
    )

    # ------------------------------------------------------------------
    # 1. DATASET
    # ------------------------------------------------------------------

    print("\nLoading datasets...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

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
    # 2. TOKENIZER
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. 4-BIT QUANTIZATION
    # ------------------------------------------------------------------

    print("\nPreparing 4-bit NF4 quantization...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------------

    print("Preparing LoRA configuration...")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",

        # QLoRA-style: adapt all linear transformer layers.
        target_modules="all-linear",
    )

    # ------------------------------------------------------------------
    # 5. TRAINING CONFIG
    # ------------------------------------------------------------------

    if args.smoke:
        epochs = 1
        max_steps = 5
        output_dir = "models/smoke-test"
    else:
        epochs = 2
        max_steps = -1
        output_dir = str(OUTPUT_DIR)

    training_config = SFTConfig(
        output_dir=output_dir,

        # ----------------------------
        # Training
        # ----------------------------
        num_train_epochs=epochs,
        max_steps=max_steps,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        warmup_steps=1 if args.smoke else 15,

        # ----------------------------
        # Memory
        # ----------------------------
        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        max_length=768,

        # Our dataset has:
        # prompt + completion
        completion_only_loss=True,

        # ----------------------------
        # Logging / evaluation
        # ----------------------------
        logging_steps=5,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # No external tracking service needed.
        report_to="none",

        # Qwen2.5 chat EOS token.
        eos_token="<|im_end|>",

        seed=42,
    )

    # ------------------------------------------------------------------
    # 6. TRAINER
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

    print("\nTrainable parameters:")

    trainer.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 7. TRAIN
    # ------------------------------------------------------------------

    print("\nStarting training...\n")

    train_result = trainer.train()

    print("\nTraining finished.")

    # ------------------------------------------------------------------
    # 8. FINAL EVALUATION
    # ------------------------------------------------------------------

    print("\nEvaluating validation loss...")

    eval_results = trainer.evaluate()

    print("\nEvaluation results:")

    for key, value in eval_results.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------------------
    # 9. SAVE LoRA ADAPTER
    # ------------------------------------------------------------------

    if not args.smoke:
        print("\nSaving LoRA adapter...")

        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        print("\nSaved to:")
        print(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()