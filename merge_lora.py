"""Merge the trained LoRA adapter into the base model and save the result
as a standard HuggingFace model directory ready for GGUF conversion.

Usage:
    python merge_lora.py [--adapter PATH] [--output PATH]

Defaults:
    --adapter  ./checkpoints/final_checkpoint
    --output   ./fyp-webagent-merged

Requirements:
    pip install peft transformers torch
"""

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: Path, output_path: Path) -> None:
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model : {base_model_id}")
    print(f"Adapter    : {adapter_path}")
    print(f"Output     : {output_path}")
    print()

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter weights into base model...")
    merged = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path} ...")
    merged.save_pretrained(output_path, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_path)

    print()
    print("Done. Next steps:")
    print(f"  1. Clone llama.cpp and run:")
    print(f"       python convert_hf_to_gguf.py {output_path.resolve()} \\")
    print(f"           --outfile fyp-webagent-q4.gguf --outtype q4_k_m")
    print(f"  2. Create a Modelfile containing:  FROM ./fyp-webagent-q4.gguf")
    print(f"  3. ollama create fyp-webagent -f Modelfile")
    print(f"  4. ollama serve")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("./checkpoints/final_checkpoint"),
        help="Path to the PEFT LoRA adapter directory (default: ./checkpoints/final_checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./fyp-webagent-merged"),
        help="Directory to save the merged HuggingFace model (default: ./fyp-webagent-merged)",
    )
    args = parser.parse_args()
    merge(args.adapter, args.output)
