# Text-to-SQL Fine-Tuning with QLoRA

Fine-tuning a code-oriented language model to translate natural-language questions and database schemas into executable SQLite queries.

The project compares the original **Qwen2.5-Coder-1.5B-Instruct** model against a **QLoRA fine-tuned version** on held-out Spider Text-to-SQL examples.

## Results

The project uses a sequence of controlled experiments to measure how individual changes affect Text-to-SQL performance.

| Experiment                 | Training Examples | Context | Execution Accuracy |             Correct | Execution Errors |
| -------------------------- | ----------------: | ------: | -----------------: | ------------------: | ---------------: |
| Base Qwen2.5-Coder-1.5B    |                — |      — |              46.4% |           116 / 250 |              101 |
| v1 — QLoRA baseline       |             1,200 |     768 |              59.6% |           149 / 250 |               59 |
| v2 — Full Spider training |             6,500 |     768 |    **64.0%** | **160 / 250** |     **36** |

### Current best result

Training the same QLoRA configuration on 6,500 Spider examples improved execution accuracy from:

```text
Base model: 46.4%
v1 QLoRA:   59.6%
v2 QLoRA:   64.0%
```

Compared with the original base model, v2 achieved:

- **+17.6 percentage points** execution accuracy
- **+37.9% relative improvement**
- **+44 additional correct queries**
- execution errors reduced from **101 to 36**

Compared with v1, increasing the training set from 1,200 to 6,500 examples improved execution accuracy by **+4.4 percentage points**.

The final benchmark contains **250 held-out Spider queries** executed against the corresponding SQLite databases.

> Note: this project uses a custom held-out Spider subset and custom execution evaluator. Results are not directly comparable to the official Spider leaderboard.

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

Incorrect: the model sorts from youngest to oldest.

### Fine-tuned model

```sql
SELECT name, country, age
FROM singer
ORDER BY age DESC;
```

Correct.

The fine-tuned model also reduced schema hallucinations such as generating nonexistent columns, incorrect joins, or unsupported SQL expressions.

---

## Project Overview

The goal of this project is to demonstrate an end-to-end parameter-efficient LLM fine-tuning workflow on consumer hardware.

The pipeline covers:

```text
Spider Dataset
      ↓
Schema + Natural Language Question
      ↓
Prompt / Completion Formatting
      ↓
Qwen2.5-Coder-1.5B-Instruct
      ↓
4-bit NF4 Quantization
      ↓
QLoRA Fine-Tuning
      ↓
LoRA Adapter
      ↓
SQL Generation
      ↓
SQLGlot Validation
      ↓
SQLite Execution
      ↓
Base vs Fine-Tuned Evaluation
```

Rather than evaluating only string similarity, generated SQL is executed against the actual Spider SQLite databases and compared with the result of the reference query.

---

## Model

**Base model**

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

The model was chosen because it is:

* optimized for code-generation tasks
* small enough for local experimentation
* suitable for parameter-efficient fine-tuning
* capable of running in 4-bit precision on a consumer GPU

---

## Dataset

The project uses the **Spider Text-to-SQL dataset**.

Each training example contains:

```text
Database schema
+
Natural-language question
+
Ground-truth SQL query
```

Example:

```text
Schema:
school(...)
school_details(...)
school_performance(...)

Question:
What is the average enrollment of schools?

SQL:
SELECT avg(Enrollment) FROM school
```

### Current experiment

```text
Training:     1,200 examples
Validation:     150 examples
Test:           250 held-out examples
```

The final test examples come from databases not used for model training.

---

## Fine-Tuning Method

The model was trained using **Supervised Fine-Tuning (SFT)** with **QLoRA**.

Instead of updating all ~1.5 billion parameters, the pretrained model is quantized and frozen while small trainable low-rank adapters are added to its linear layers.

### Quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### LoRA

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

Only:

```text
18,464,768 parameters
```

were trainable, approximately:

```text
1.18% of the model
```

---

## Training Configuration

```text
Epochs:                   2
Training examples:        1,200
Validation examples:      150
Batch size:               1
Gradient accumulation:    8
Learning rate:            2e-4
Max sequence length:      768
Precision:                BF16
Base model precision:     4-bit NF4
Gradient checkpointing:   Enabled
```

### Training Result

```text
Training loss:          0.1643
Validation loss:        0.1706
Validation token acc:   95.07%
Training time:          ~24m 55s
```

Token-level accuracy is reported only as a training diagnostic and is **not** treated as the main Text-to-SQL metric.

---

## Hardware

Training was performed locally on:

```text
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
VRAM: ~8 GB
OS: Windows
Python: 3.11
```

QLoRA made it possible to fine-tune the model without requiring a high-memory datacenter GPU.

---

## Evaluation

Three forms of evaluation are used.

### 1. SQL Parse Validity

Generated queries are parsed using **SQLGlot**.

This detects malformed SQL before execution.

### 2. Canonical SQL Comparison

SQLGlot is also used to normalize SQL before structural string comparison.

This is more useful than raw exact-string matching but still cannot reliably determine semantic equivalence.

For example:

```sql
salary > 50000
```

and:

```sql
50000 < salary
```

can represent the same operation despite being different strings.

### 3. Execution Accuracy

The primary metric is **execution accuracy**.

For each example:

```text
Reference SQL ──→ SQLite ──→ Reference result

Base SQL ───────→ SQLite ──→ Result
                                  ↓
                              Compare

Fine-tuned SQL → SQLite ──→ Result
                                  ↓
                              Compare
```

A prediction is counted as correct when executing it produces the same result as the reference query.

The databases are opened in read-only mode during evaluation.

---

## Final Benchmark

Evaluation on 250 held-out queries:

```text
Base model
Execution accuracy: 46.4%
Correct queries:     116 / 250
Execution errors:    101

Fine-tuned model
Execution accuracy: 59.6%
Correct queries:     149 / 250
Execution errors:     59
```

### Improvement

```text
Execution Accuracy
46.4% → 59.6%

+13.2 percentage points
+28.4% relative improvement
```

Execution errors decreased from:

```text
101 → 59
```

which represents approximately a:

```text
41.6% reduction
```

---

## Error Analysis

Fine-tuning noticeably improved several common failure modes.

### Incorrect ordering

Base:

```sql
ORDER BY Age ASC
```

Fine-tuned:

```sql
ORDER BY age DESC
```

### Unnecessary or hallucinated joins

Base models frequently generated joins such as:

```sql
JOIN singer ...
```

even when the answer could be retrieved directly from the target table.

The fine-tuned model more frequently learned the simpler schema-correct query.

### Nonexistent columns

Examples of baseline execution failures included:

```text
no such column: T2.Average
no such column: T2.Name
```

Fine-tuning reduced the total number of execution errors by more than 40%.

### Remaining limitations

The fine-tuned model still fails on some complex cases involving:

* multi-table joins
* nested queries
* `INTERSECT`
* `EXCEPT`
* schema linking
* ambiguous natural-language questions
* complex aggregation

Fine-tuning can also introduce regressions: some queries solved correctly by the base model are answered incorrectly by the adapted model.

---

## Project Structure

```text
text-to-sql-finetuning/
│
├── data/
│   └── processed/
│
├── models/
│   └── text-to-sql-qwen-lora/
│
├── results/
│   └── model_comparison.jsonl
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
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

Large model files, downloaded databases, and generated datasets are excluded from Git.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Prepare Dataset

```bash
python src/prepare_dataset.py
```

This generates the processed train, validation, and test JSONL files.

---

## Train

Run a short smoke test first:

```bash
python src/train.py --smoke
```

Run full training:

```bash
python src/train.py
```

The resulting LoRA adapter is saved to:

```text
models/text-to-sql-qwen-lora/
```

---

## Download Spider Databases

```bash
python src/download_spider_databases.py
```

These SQLite databases are required for execution-based evaluation.

---

## Benchmark

Generate predictions for the held-out test subset:

```bash
python src/compare_models.py --limit 250
```

Then execute the generated and reference queries:

```bash
python src/evaluate_execution.py
```

---

# Roadmap

The current experiment intentionally uses a relatively small training subset and a compact 1.5B model.

Several improvements are planned.

## 1. Train on the Full Spider Training Set

Current:

```text
1,200 training examples
```

Future:

```text
Full Spider training split
```

Increasing dataset coverage should expose the model to more schemas, join patterns, aggregations, nested queries, and SQL operators.

---

## 2. Increase Training Context Length

Current:

```text
max_length = 768
```

Planned experiments:

```text
1024
1536
```

Large database schemas may currently be truncated, which can remove table or foreign-key information needed for correct SQL generation.

---

## 3. Schema-Aware Prompting

Introduce stricter generation instructions:

```text
Use only tables present in the schema.
Use only columns present in the schema.
Do not invent tables or columns.
Return exactly one SQLite query.
Do not provide explanations.
```

The schema representation can also be extended to explicitly identify:

* primary keys
* foreign keys
* table relationships
* column types

---

## 4. Hyperparameter Experiments

Current configuration:

```text
LoRA rank:        16
LoRA alpha:       32
Learning rate:    2e-4
Epochs:           2
```

Candidate experiment:

```text
LoRA rank:        32
LoRA alpha:       64
Learning rate:    1e-4
Epochs:           3
```

Experiments will be compared using execution accuracy rather than training loss alone.

---

## 5. Larger Base Model

Evaluate:

```text
Qwen2.5-Coder-3B-Instruct
```

using 4-bit QLoRA.

A larger model may improve:

* complex schema reasoning
* joins
* nested queries
* set operations
* schema linking

while remaining potentially feasible on limited GPU memory.

---

## 6. Schema Validation

Add a validation stage between generation and execution:

```text
Generated SQL
      ↓
SQLGlot parser
      ↓
Extract referenced tables / columns
      ↓
Compare against database schema
      ↓
Reject invalid schema references
```

This directly targets one of the most common failure modes: hallucinated tables and columns.

---

## 7. Execution-Guided SQL Repair

Implement an iterative inference pipeline:

```text
Question + Schema
      ↓
Fine-Tuned Model
      ↓
Generated SQL
      ↓
SQLite Execution
      ↓
Execution Error?
      ↓
Error + Schema + SQL
      ↓
Model Repair
      ↓
Execute Again
```

Example:

```text
SQLite error:
no such column: T3.concert_name
```

The model receives the execution error and available schema and gets one opportunity to repair its SQL.

Future evaluation can therefore compare:

```text
Base Model
        ↓
Fine-Tuned Model
        ↓
Fine-Tuned + Execution-Guided Repair
```

---

## Target Architecture

```text
Natural Language Question
          +
    Database Schema
          ↓
 Fine-Tuned Qwen
          ↓
    Generated SQL
          ↓
   SQLGlot Validation
          ↓
 Schema Consistency Check
          ↓
     SQLite Execution
          ↓
       Success?
       /      \
     Yes       No
      ↓         ↓
   Result    Repair Agent
                  ↓
             Corrected SQL
                  ↓
               SQLite
```

This transforms the project from a pure fine-tuning experiment into a more robust **Text-to-SQL inference system**.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL
* bitsandbytes
* QLoRA / LoRA
* SQLGlot
* SQLite
* Qwen2.5-Coder

---

## Key Takeaway

This project demonstrates that parameter-efficient fine-tuning can significantly improve the task-specific behavior of a small language model using consumer hardware.

With approximately **1.18% of model parameters trainable**, QLoRA increased held-out SQL execution accuracy from:

```text
46.4% → 59.6%
```

while reducing SQL execution errors by approximately:

```text
41.6%
```

The next stage is focused on full-dataset training, stronger schema grounding, and execution-guided error correction.

## Results

The project uses a sequence of controlled experiments to measure how individual changes affect Text-to-SQL performance.

| Experiment                 | Training Examples | Context | Execution Accuracy |             Correct | Execution Errors |
| -------------------------- | ----------------: | ------: | -----------------: | ------------------: | ---------------: |
| Base Qwen2.5-Coder-1.5B    |                — |      — |              46.4% |           116 / 250 |              101 |
| v1 — QLoRA baseline       |             1,200 |     768 |              59.6% |           149 / 250 |               59 |
| v2 — Full Spider training |             6,500 |     768 |    **64.0%** | **160 / 250** |     **36** |

### Current best result

Training the same QLoRA configuration on 6,500 Spider examples improved execution accuracy from:

```text
Base model: 46.4%
v1 QLoRA:   59.6%
v2 QLoRA:   64.0%
```

Compared with the original base model, v2 achieved:

- **+17.6 percentage points** execution accuracy
- **+37.9% relative improvement**
- **+44 additional correct queries**
- execution errors reduced from **101 to 36**

Compared with v1, increasing the training set from 1,200 to 6,500 examples improved execution accuracy by **+4.4 percentage points**.

The final benchmark contains **250 held-out Spider queries** executed against the corresponding SQLite databases.

> Note: this project uses a custom held-out Spider subset and custom execution evaluator. Results are not directly comparable to the official Spider leaderboard.

---

## Results

The project uses a sequence of controlled experiments to measure how individual changes affect Text-to-SQL performance.

| Experiment                 | Training Examples | Context | Execution Accuracy |             Correct | Execution Errors |
| -------------------------- | ----------------: | ------: | -----------------: | ------------------: | ---------------: |
| Base Qwen2.5-Coder-1.5B    |                — |      — |              46.4% |           116 / 250 |              101 |
| v1 — QLoRA baseline       |             1,200 |     768 |              59.6% |           149 / 250 |               59 |
| v2 — Full Spider training |             6,500 |     768 |    **64.0%** | **160 / 250** |     **36** |

### Current best result

Training the same QLoRA configuration on 6,500 Spider examples improved execution accuracy from:

```text
Base model: 46.4%
v1 QLoRA:   59.6%
v2 QLoRA:   64.0%
```

Compared with the original base model, v2 achieved:

- **+17.6 percentage points** execution accuracy
- **+37.9% relative improvement**
- **+44 additional correct queries**
- execution errors reduced from **101 to 36**

Compared with v1, increasing the training set from 1,200 to 6,500 examples improved execution accuracy by **+4.4 percentage points**.

The final benchmark contains **250 held-out Spider queries** executed against the corresponding SQLite databases.

> Note: this project uses a custom held-out Spider subset and custom execution evaluator. Results are not directly comparable to the official Spider leaderboard.

---
