

# Text-to-SQL Fine-Tuning with QLoRA

A local LLM fine-tuning project for translating natural-language questions and database schemas into executable SQL.

The project uses **Qwen2.5-Coder**, **QLoRA**, the **Spider Text-to-SQL dataset**, SQLGlot, and SQLite execution-based evaluation.

The main goal is to measure how controlled changes to training data, context length, prompting, schema representation, LoRA configuration, and model size affect real SQL execution accuracy.

---

## Best Result

The strongest experiment so far is:

```text
Model:              Qwen2.5-Coder-1.5B-Instruct
Training examples:  6,500
Validation:           500
Context length:      1,024
LoRA rank:              16
LoRA alpha:             32
Epochs:                  2
```

Execution accuracy improved from:

```text
Base model:          46.4%
Best fine-tuned:     66.4%
```

That corresponds to:

```text
+20.0 percentage points
+43.1% relative improvement
+50 additional correct queries
```

On the 250-query development benchmark:

```text
Base model:       116 / 250 correct
Best fine-tuned:  166 / 250 correct
```

> This project uses a custom Spider subset and a custom SQLite execution evaluator. The results are not directly comparable to the official Spider leaderboard.

---

## Experiment Results

| Experiment                       | Model                        |           Train |           Val |        Context | LoRA                 | Execution Accuracy |             Correct | Execution Errors |
| -------------------------------- | ---------------------------- | --------------: | ------------: | -------------: | -------------------- | -----------------: | ------------------: | ---------------: |
| Base                             | Qwen2.5-Coder-1.5B           |              — |            — |             — | —                   |              46.4% |           116 / 250 |              101 |
| v1 — Baseline QLoRA             | Qwen2.5-Coder-1.5B           |           1,200 |           150 |            768 | r16 / α32           |              59.6% |           149 / 250 |               59 |
| v2 — Full training split        | Qwen2.5-Coder-1.5B           |           6,500 |           500 |            768 | r16 / α32           |              64.0% |           160 / 250 |               36 |
| **v3 — Context 1024**     | **Qwen2.5-Coder-1.5B** | **6,500** | **500** | **1024** | **r16 / α32** |    **66.4%** | **166 / 250** |     **44** |
| v4 — Strict schema-aware prompt | Qwen2.5-Coder-1.5B           |           6,500 |           500 |           1024 | r16 / α32           |              63.2% |           158 / 250 |               41 |
| v5 — Structured PK/FK schema    | Qwen2.5-Coder-1.5B           |           6,500 |           500 |           1024 | r16 / α32           |              62.8% |           157 / 250 |               52 |
| v6 — Larger LoRA rank           | Qwen2.5-Coder-1.5B           |           6,500 |           500 |           1024 | r32 / α64           |              64.4% |           161 / 250 |               50 |
| v7 — Qwen 3B                    | Qwen2.5-Coder-3B             |           6,500 |           500 |           1024 | r16 / α32           |              60.4% |           151 / 250 |               49 |

Full experiment metadata is stored in:

```text
results/experiments.csv
```

---

## Key Findings

The experiments produced several useful results.

### More training data helped

Increasing the training set from:

```text
1,200 → 6,500 examples
```

improved execution accuracy from:

```text
59.6% → 64.0%
```

### Larger context helped

Increasing the maximum sequence length from:

```text
768 → 1024
```

produced the best result:

```text
64.0% → 66.4%
```

### Stricter prompting did not automatically improve accuracy

A more restrictive schema-aware prompt reduced execution accuracy:

```text
66.4% → 63.2%
```

The generated SQL became more constrained, but semantic correctness did not improve.

### More explicit schema metadata did not help the small model

Adding explicit primary-key and foreign-key structure produced:

```text
62.8%
```

This suggests that simply adding more schema information can increase prompt complexity without improving reasoning.

### Larger LoRA rank was not better

Increasing:

```text
r16 / alpha32
```

to:

```text
r32 / alpha64
```

resulted in:

```text
64.4%
```

which remained below the v3 result.

### A larger base model was not automatically better

Qwen2.5-Coder-3B was trained with the same main QLoRA configuration and reached:

```text
60.4%
```

compared with:

```text
66.4%
```

for the 1.5B v3 experiment.

This project therefore shows why model changes need to be evaluated empirically rather than assuming that a larger model or more complex configuration will necessarily perform better.

---

## Example

### Question

```text
Show name, country, age for all singers ordered by age
from the oldest to the youngest.
```

### Base model

```sql
SELECT Name, Country, Age
FROM singer
ORDER BY Age ASC;
```

The ordering direction is incorrect.

### Fine-tuned model

```sql
SELECT name, country, age
FROM singer
ORDER BY age DESC;
```

The fine-tuned model correctly interprets "oldest to youngest."

---

## Project Pipeline

```text
Spider Dataset
      ↓
Database Schema + Natural-Language Question
      ↓
Prompt / Completion Formatting
      ↓
Qwen2.5-Coder
      ↓
4-bit NF4 Quantization
      ↓
QLoRA Supervised Fine-Tuning
      ↓
LoRA Adapter
      ↓
SQL Generation
      ↓
SQLGlot Validation
      ↓
SQLite Execution
      ↓
Reference Result Comparison
```

The primary evaluation metric is **execution accuracy**.

Training loss and token-level accuracy are used only as training diagnostics.

---

## Dataset

The project uses the **Spider Text-to-SQL dataset**.

Each example contains:

```text
Database schema
+
Natural-language question
+
Reference SQL
```

For the main experiments, the original Spider training split is divided into:

```text
Training:     6,500 examples
Validation:     500 examples
```

The first 250 examples from the Spider validation split are used as a consistent development benchmark:

```text
Development benchmark: 250 queries
```

These same 250 queries were reused across v1–v7 so that experiments could be compared under identical conditions.

Because this benchmark has been used repeatedly for model selection, it should be considered a **development set rather than an untouched final test set**.

The remaining Spider validation examples can be reserved for a future final evaluation.

---

## Fine-Tuning Method

The project uses **Supervised Fine-Tuning (SFT)** with **QLoRA**.

The pretrained model is quantized to 4-bit precision and kept frozen while small trainable LoRA adapters are added to the model.

### Quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### Baseline LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

For the 1.5B model, approximately:

```text
18.5 million parameters
```

are trainable, representing roughly:

```text
1.18% of the model
```

---

## Best Training Configuration

The current best configuration is v3:

```text
Base model:                 Qwen2.5-Coder-1.5B-Instruct
Training examples:          6,500
Validation examples:          500
Epochs:                         2
Batch size:                     1
Gradient accumulation:          8
Learning rate:               2e-4
Max sequence length:          1024
LoRA rank:                      16
LoRA alpha:                     32
LoRA dropout:                 0.05
Precision:                    BF16
Base model quantization:  4-bit NF4
Gradient checkpointing:    Enabled
Seed:                           42
```

---

## Evaluation

The project uses three complementary evaluation methods.

### 1. SQL Parse Validity

Generated queries are parsed using **SQLGlot**.

This detects malformed SQL before execution.

### 2. Canonical SQL Comparison

SQLGlot is used to normalize generated and reference SQL.

This metric is useful for diagnostics, but exact structural equivalence is not the primary goal.

For example:

```sql
salary > 50000
```

and:

```sql
50000 < salary
```

can represent the same condition despite having different textual structures.

### 3. Execution Accuracy

The primary metric is execution accuracy.

```text
Reference SQL
      ↓
    SQLite
      ↓
Reference Result


Generated SQL
      ↓
    SQLite
      ↓
Generated Result
      ↓
Compare with Reference Result
```

A prediction is counted as correct when the generated query produces the same result as the reference SQL.

SQLite databases are opened in read-only mode during evaluation.

---

## Error Analysis

The best v3 model reaches:

```text
166 / 250 correct
66.4% execution accuracy
```

The remaining 84 incorrect examples consist of:

```text
Execution errors:          44
Executable but incorrect:  40
```

Observed failure modes include:

- hallucinated tables
- hallucinated columns
- incorrect schema linking
- incorrect joins
- wrong table selection
- aggregation mistakes
- incorrect `GROUP BY` logic
- nested-query mistakes
- incorrect set operations such as `INTERSECT` and `EXCEPT`
- incorrect literal values
- filter mistakes
- relational reasoning errors

Examples include generating identifiers such as:

```text
petage
pet_type
MakerID
```

when the real schema contains differently named columns.

Another important finding is:

```text
valid SQL != correct SQL
```

A query can parse successfully and execute without errors while still answering the question incorrectly.

---

## Hardware

Training was performed locally on:

```text
GPU:     NVIDIA GeForce RTX 5070 Laptop GPU
VRAM:    ~8 GB
OS:      Windows
Python:  3.11
```

QLoRA allows the experiments to run on consumer hardware without requiring a high-memory datacenter GPU.

---

## Project Structure

```text
text-to-sql-finetuning/
│
├── data/
│
├── models/
│
├── results/
│   ├── experiments.csv
│   ├── v1-1200-context768/
│   ├── v2-full-context768/
│   ├── v3-full-context1024/
│   ├── v4-schema-aware/
│   ├── v5-structured-schema/
│   ├── v6-lora-r32/
│   └── v7-qwen3b/
│
├── src/
│   ├── prepare_dataset.py
│   ├── test_base_model.py
│   ├── train.py
│   ├── compare_models.py
│   ├── download_spider_databases.py
│   └── evaluate_execution.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

Large model checkpoints, downloaded databases, virtual environments, and generated datasets are excluded from Git where appropriate.

---

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Prepare Dataset

```powershell
python src\prepare_dataset.py
```

The default full-data configuration creates:

```text
6,500 training examples
500 validation examples
250 development examples
```

---

## Train

Run a small smoke test first:

```powershell
python src\train.py --smoke
```

Example command for the best v3 configuration:

```powershell
python src\train.py `
  --experiment-id v3-full-context1024 `
  --train-file data\v2-full\train.jsonl `
  --val-file data\v2-full\validation.jsonl `
  --output-dir models\v3-full-context1024 `
  --max-length 1024 `
  --epochs 2 `
  --learning-rate 2e-4 `
  --lora-r 16 `
  --lora-alpha 32
```

---

## Generate Predictions

```powershell
python src\compare_models.py `
  --experiment-id v3-full-context1024 `
  --adapter-path models\v3-full-context1024 `
  --test-file data\v2-full\test.jsonl
```

For a different base model, such as Qwen2.5-Coder-3B:

```powershell
python src\compare_models.py `
  --model-name Qwen/Qwen2.5-Coder-3B-Instruct `
  --experiment-id v7-qwen3b `
  --adapter-path models\v7-qwen3b `
  --test-file data\v2-full\test.jsonl
```

---

## Execution Evaluation

```powershell
python src\evaluate_execution.py `
  --experiment-id v3-full-context1024
```

The evaluator executes both reference and generated SQL against the corresponding Spider SQLite database and compares the returned results.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL
- PEFT
- bitsandbytes
- QLoRA
- LoRA
- SQLGlot
- SQLite
- Spider
- Qwen2.5-Coder

---

## Current Status

The experiment series is currently paused after **v7**.

The strongest model remains:

```text
Qwen2.5-Coder-1.5B-Instruct
+
QLoRA
+
6,500 Spider training examples
+
1024-token context
```

with:

```text
66.4% execution accuracy
166 / 250 correct queries
```

Further improvements would likely require stronger schema grounding, better handling of relational structure, and evaluation on a separate untouched benchmark rather than continued optimization against the same 250-query development set.

---

## Key Takeaway

The project demonstrates an end-to-end parameter-efficient LLM fine-tuning workflow on consumer hardware.

The best QLoRA experiment improved execution accuracy from:

```text
46.4% → 66.4%
```

while training only a small fraction of the full model parameters.

The experiment history also shows an important practical result:

> More parameters, more prompt constraints, more schema metadata, and larger LoRA adapters do not automatically improve downstream task performance.

Each change needs to be evaluated against the metric that actually matters for the application — in this case, whether the generated SQL produces the correct database result.
