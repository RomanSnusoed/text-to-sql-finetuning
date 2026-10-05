import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    # QLoRA-style 4-bit configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("\nModel loaded successfully.")
    print(
        "Model memory footprint:",
        round(model.get_memory_footprint() / 1024**3, 2),
        "GB",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Text-to-SQL assistant. "
                "Return SQL only. Do not explain the answer."
            ),
        },
        {
            "role": "user",
            "content": """
Database schema:

customers(
    customer_id INTEGER,
    name TEXT,
    country TEXT
)

orders(
    order_id INTEGER,
    customer_id INTEGER,
    total REAL
)

Question:
What are the names of customers from Germany?
""".strip(),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    print("\nGenerating SQL with BASE model...")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Remove the prompt tokens and decode only generated output.
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\nBASE MODEL OUTPUT:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)


if __name__ == "__main__":
    main()