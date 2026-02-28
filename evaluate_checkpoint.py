import json
import os
import re
import gc
import random
import argparse
import sys
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from pipeline_in_steps import (
    MODEL_NAME,
    MAX_TRAJECTORY_STEPS,
    MAX_HISTORY_STEPS,
    PAGE_LOAD_WAIT_SECONDS,
    convert_html_to_markdown,
    initialize_browser_session,
    get_observation_with_retry,
    format_history_for_prompt,
    generate_action_from_observation,
    parse_action_from_response,
    is_stop_action,
    save_screenshot,
)
from judge_integration import judge_trajectory, print_judgment


def load_model_with_checkpoint(checkpoint_dir: str):
    """Load the base model with 4-bit quantization and apply the LoRA checkpoint."""
    print(f"Loading base model: {MODEL_NAME}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print(f"Loading LoRA adapter from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    model.config.use_cache = True
    
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded — Total params: {total_params:,}, Adapter params: {adapter_params:,}")
    
    return tokenizer, model


def load_base_model():
    """Load the base model with 4-bit quantization but WITHOUT any LoRA adapter."""
    print(f"Loading base model (no adapter): {MODEL_NAME}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    model.config.use_cache = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model loaded — Total params: {total_params:,}")

    return tokenizer, model


def unload_model(model):
    """
    Free GPU memory occupied by a model.
    The caller MUST drop their own reference after this call:
        model = unload_model(model)
    """
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return None


def _instruction_bucket(instruction: str) -> str:
    """Assign an instruction to a coarse action-type bucket based on its first verb."""
    text = instruction.lower().strip()
    patterns = [
        ("search",       r"\b(search|look for)\b"),
        ("count",        r"\b(how many|count|total number)\b"),
        ("fill_submit",  r"\b(fill|submit|enter|type)\b"),
        ("navigate",     r"\b(go to|navigate|open|click|visit)\b"),
        ("find_locate",  r"\b(find|locate|identify|determine|what is|what are)\b"),
    ]
    for bucket, pattern in patterns:
        if re.search(pattern, text):
            return bucket
    return "other"


def sample_tasks(df, sample_size: int, seed: int):
    """
    Draw a stratified random sample of tasks from the dataset.

    Stratification is by instruction verb type (find/navigate/search/count/fill/other).
    Within each stratum, tasks are sampled proportionally to stratum size.
    Falls back to a plain random sample if any stratum is too small.
    """
    import pandas as pd

    random.seed(seed)
    df = df.copy().reset_index(drop=True)
    df["_bucket"] = df["instruction"].apply(_instruction_bucket)

    bucket_counts = df["_bucket"].value_counts()
    print("Instruction bucket distribution:")
    for bucket, count in bucket_counts.items():
        print(f"  {bucket:<15} {count:>5} tasks")

    # Proportional allocation: round down, then top up with largest-remainder method
    total = len(df)
    raw_allocs = {b: (count / total) * sample_size for b, count in bucket_counts.items()}
    floored = {b: int(v) for b, v in raw_allocs.items()}
    remainder = sample_size - sum(floored.values())
    # Give extra slots to buckets with largest fractional parts
    frac_parts = sorted(raw_allocs.items(), key=lambda x: -(x[1] - int(x[1])))
    for i, (b, _) in enumerate(frac_parts):
        if i < remainder:
            floored[b] += 1

    sampled_frames = []
    for bucket, n in floored.items():
        if n == 0:
            continue
        bucket_df = df[df["_bucket"] == bucket]
        n_draw = min(n, len(bucket_df))
        sampled_frames.append(bucket_df.sample(n=n_draw, random_state=seed))

    sample = pd.concat(sampled_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample = sample.drop(columns=["_bucket"])

    # If we ended up short (due to small buckets), pad with random rows not already selected
    if len(sample) < sample_size:
        remaining = df[~df.index.isin(sample.index)].drop(columns=["_bucket"])
        shortfall = sample_size - len(sample)
        extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed)
        sample = pd.concat([sample, extra]).reset_index(drop=True)

    print(f"\nSampled {len(sample)} tasks (seed={seed})")
    return sample


def save_results(results: list, path: str):
    """Save a list of trajectory result dicts to a JSON file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    serialisable = []
    for r in results:
        if r is None:
            serialisable.append(None)
            continue
        entry = {k: v for k, v in r.items() if k != "judgment"}
        if r.get("judgment"):
            j = r["judgment"]
            entry["judgment"] = {
                "success": j.success,
                "efficiency": j.efficiency,
                "self_correction": j.self_correction,
            }
        serialisable.append(entry)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to: {path}")


def _mean(values):
    return sum(values) / len(values) if values else None


def _extract_scores(results):
    """Return (successes, efficiencies, self_corrections, steps) from a results list."""
    successes, efficiencies, self_corrections, steps = [], [], [], []
    for r in results:
        if r is None:
            continue
        j = r.get("judgment")
        if j:
            if j.success is not None:         successes.append(j.success)
            if j.efficiency is not None:       efficiencies.append(j.efficiency)
            if j.self_correction is not None:  self_corrections.append(j.self_correction)
        steps.append(r["num_steps"])
    return successes, efficiencies, self_corrections, steps


def print_comparison(checkpoint_results: list, base_results: list):
    """Print a side-by-side summary comparing checkpoint vs base model."""
    cs, ce, csc, csteps = _extract_scores(checkpoint_results)
    bs, be, bsc, bsteps = _extract_scores(base_results)

    n_cp = len(checkpoint_results)
    n_base = len(base_results)
    cp_binary = len([s for s in cs if s > 0.5])
    base_binary = len([s for s in bs if s > 0.5])
    cp_rate = cp_binary / n_cp * 100 if n_cp else 0.0
    base_rate = base_binary / n_base * 100 if n_base else 0.0
    rate_delta = cp_rate - base_rate

    def fmt(val):
        return f"{val:.4f}" if val is not None else "  N/A  "

    def delta(a, b):
        if a is None or b is None:
            return "  N/A"
        d = a - b
        sign = "+" if d >= 0 else ""
        mark = " ✓" if d > 0.01 else (" ✗" if d < -0.01 else "")
        return f"{sign}{d:.4f}{mark}"

    cm_success = _mean(cs)
    bm_success = _mean(bs)
    cm_eff     = _mean(ce)
    bm_eff     = _mean(be)
    cm_sc      = _mean(csc)
    bm_sc      = _mean(bsc)
    cm_steps   = _mean(csteps)
    bm_steps   = _mean(bsteps)

    rate_delta_str = f"{rate_delta:+.1f}pp" + (" ✓" if rate_delta > 0 else (" ✗" if rate_delta < 0 else ""))

    print(f"\n{'='*65}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*65}")
    header = f"{'Metric':<22} {'Checkpoint':>12} {'Base Model':>12} {'Delta':>12}"
    print(header)
    print("-" * 65)
    print(f"{'Task Success Rate':<22} {f'{cp_rate:.1f}%':>12} {f'{base_rate:.1f}%':>12} {rate_delta_str:>12}")
    print(f"  (score > 0.5)  {f'{cp_binary}/{n_cp}':>22} {f'{base_binary}/{n_base}':>12}")
    print("-" * 65)
    print(f"{'Avg Success Score':<22} {fmt(cm_success):>12} {fmt(bm_success):>12} {delta(cm_success, bm_success):>12}")
    print(f"{'Avg Efficiency':<22} {fmt(cm_eff):>12} {fmt(bm_eff):>12} {delta(cm_eff, bm_eff):>12}")
    print(f"{'Avg Self-Correction':<22} {fmt(cm_sc):>12} {fmt(bm_sc):>12} {delta(cm_sc, bm_sc):>12}")

    c_step_str = f"{cm_steps:.1f}" if cm_steps is not None else "N/A"
    b_step_str = f"{bm_steps:.1f}" if bm_steps is not None else "N/A"
    step_delta = f"{(cm_steps - bm_steps):+.1f}" if (cm_steps is not None and bm_steps is not None) else "N/A"
    print(f"{'Avg Steps':<22} {c_step_str:>12} {b_step_str:>12} {step_delta:>12}")
    print(f"{'='*65}")
    print(f"Tasks evaluated (checkpoint): {len([r for r in checkpoint_results if r])} / {n_cp}")
    print(f"Tasks evaluated (base)      : {len([r for r in base_results if r])} / {n_base}")
    print(f"{'='*65}\n")


def run_eval_trajectory(task_data: dict, model, tokenizer):
    """
    Run a single evaluation trajectory: observe → markdown → prompt → generate → execute.
    No RL training or gradient computation.
    """
    start_url = task_data["website"]
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    trajectory_markdown_observations = []
    trajectory_action_jsons = []
    trajectory_responses = []
    history = []
    client = None

    try:
        client = initialize_browser_session(start_url)
        current_observation = get_observation_with_retry(client)
        print("Initial observation received.")
        save_screenshot(current_observation, 0, "eval_initial")

        for step in range(MAX_TRAJECTORY_STEPS):
            print(f"\n--- Eval Step {step + 1}/{MAX_TRAJECTORY_STEPS} ---")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Converting HTML to Markdown...")
            markdown_content = convert_html_to_markdown(current_observation.__dict__)
            if not markdown_content:
                print("Markdown conversion failed. Stopping.")
                break
            print("Markdown content generated.")

            trajectory_markdown_observations.append(markdown_content)
            save_screenshot(current_observation, step + 1, "eval_obs")

            print("Generating action...")
            response_text, agent_prompt, prompt_text = generate_action_from_observation(
                tokenizer,
                model,
                task_data["instruction"],
                current_observation,
                markdown_content,
                history,
            )
            trajectory_responses.append(response_text)

            predicted_action, json_text = parse_action_from_response(
                response_text, agent_prompt
            )

            if not predicted_action:
                print("LLM response did not contain a valid JSON block. Stopping.")
                print("Full response:", response_text)
                break

            print(f"Action (JSON): {json_text}")
            print(f"Parsed function calls: {predicted_action.function_calls}")
            trajectory_action_jsons.append(json_text)

            history.append((markdown_content, json_text))
            if len(history) > MAX_HISTORY_STEPS:
                history = history[-MAX_HISTORY_STEPS:]

            if is_stop_action(json_text):
                print("Stop action received. Ending trajectory.")
                break

            print("Executing action...")
            from utils import BrowserStatus

            status = client.action(predicted_action.function_calls)
            if status == BrowserStatus.ERROR:
                print("Failed to execute action. Stopping.")
                break
            print("Action executed successfully.")

            current_observation = get_observation_with_retry(client)
            print("Next observation received.")
            save_screenshot(current_observation, step + 2, "eval_after")

        # Judge the trajectory
        print("\n--- Judging Trajectory ---")
        judgment = judge_trajectory(
            instruction=task_data["instruction"],
            observations=trajectory_markdown_observations,
            actions=trajectory_action_jsons,
            criteria=task_data.get("criteria", ""),
            steps=task_data.get("steps", ""),
        )
        print("Judgment received:")
        print_judgment(judgment)

        return {
            "task_instruction": task_data["instruction"],
            "website": start_url,
            "num_steps": len(trajectory_action_jsons),
            "trajectory_markdown_observations": trajectory_markdown_observations,
            "trajectory_action_jsons": trajectory_action_jsons,
            "trajectory_responses": trajectory_responses,
            "judgment": judgment,
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if client and client.session_id:
            client.close()
            print("Browser session closed.")


def print_summary(results: list):
    """Print aggregate evaluation results."""
    successful = [r for r in results if r is not None and r.get("judgment")]
    if not successful:
        print("\nNo successful trajectories to summarize.")
        return

    successes = []
    efficiencies = []
    self_corrections = []

    for r in successful:
        j = r["judgment"]
        if j.success is not None:
            successes.append(j.success)
        if j.efficiency is not None:
            efficiencies.append(j.efficiency)
        if j.self_correction is not None:
            self_corrections.append(j.self_correction)

    binary_successes = [s for s in successes if s > 0.5]
    success_rate = len(binary_successes) / len(results) * 100 if results else 0.0

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks attempted : {len(results)}")
    print(f"Successful runs       : {len(successful)}")
    print(f"Failed runs           : {len(results) - len(successful)}")
    print()
    print(f"Task Success Rate     : {len(binary_successes)}/{len(results)} ({success_rate:.1f}%)  [judge score > 0.5]")
    if successes:
        print(f"Avg Success Score     : {sum(successes)/len(successes):.4f}")
    if efficiencies:
        print(f"Avg Efficiency Score  : {sum(efficiencies)/len(efficiencies):.4f}")
    if self_corrections:
        print(f"Avg Self-Correction   : {sum(self_corrections)/len(self_corrections):.4f}")
    print()

    avg_steps = sum(r["num_steps"] for r in successful) / len(successful)
    print(f"Avg Steps per Task    : {avg_steps:.1f}")
    print(f"{'='*60}")

    # Per-task breakdown
    print("\nPer-Task Results:")
    print(f"{'#':<4} {'Success':<9} {'Effic.':<9} {'Self-Corr.':<12} {'Steps':<6} Instruction")
    print("-" * 90)
    for i, r in enumerate(results):
        if r is None:
            print(f"{i+1:<4} {'FAILED':<9} {'-':<9} {'-':<12} {'-':<6} -")
            continue
        j = r.get("judgment")
        s = f"{j.success:.2f}" if (j and j.success is not None) else "-"
        e = f"{j.efficiency:.2f}" if (j and j.efficiency is not None) else "-"
        sc = f"{j.self_correction:.2f}" if (j and j.self_correction is not None) else "-"
        instr = r["task_instruction"][:50] + "..." if len(r["task_instruction"]) > 50 else r["task_instruction"]
        print(f"{i+1:<4} {s:<9} {e:<9} {sc:<12} {r['num_steps']:<6} {instr}")


def _run_tasks(task_rows, model, tokenizer, label=""):
    """Run eval trajectories for a list of task row dicts and return results."""
    results = []
    total = len(task_rows)
    for i, task_row in enumerate(task_rows):
        print(f"\n{'='*60}")
        print(f"{label}TASK {i+1}/{total}")
        print(f"Instruction: {task_row['instruction']}")
        print(f"Website    : {task_row['website']}")
        print(f"{'='*60}")

        result = run_eval_trajectory(task_row, model, tokenizer)
        results.append(result)

        if result:
            print(f"\n--- Task {i+1} Complete: {result['num_steps']} steps ---")
        else:
            print(f"\n--- Task {i+1} FAILED ---")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA checkpoint on the test dataset"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/checkpoint_trajectory_125",
        help="Path to the LoRA checkpoint directory",
    )
    # --- Stratified sampling (preferred) ---
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of tasks to draw via stratified random sampling. "
             "If set, overrides --num_tasks / --start_idx.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling (default: 42)",
    )
    # --- Legacy sequential selection ---
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="(Legacy) Number of sequential tasks to evaluate when --sample_size is not set",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="(Legacy) Starting index in the dataset when --sample_size is not set",
    )
    # --- Comparison mode ---
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Also evaluate the base model (no adapter) on the same tasks and print a comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save per-model JSON result files (default: eval_results)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/insta-150k-test.csv",
        help="Path to the test CSV dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        import pipeline_in_steps
        pipeline_in_steps.DEBUG_PIPELINE = True
        print("[DEBUG] Debug mode enabled")

    # ------------------------------------------------------------------ #
    # Load dataset and select tasks
    # ------------------------------------------------------------------ #
    df = pd.read_csv(args.dataset)
    print(f"Dataset loaded: {len(df)} tasks total")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.sample_size is not None:
        print(f"\nStratified sampling: {args.sample_size} tasks | seed={args.seed}")
        sample_df = sample_tasks(df, args.sample_size, args.seed)
        task_rows = sample_df.to_dict(orient="records")
        selection_label = f"sample_size={args.sample_size}_seed={args.seed}"
    else:
        end_idx = min(args.start_idx + args.num_tasks, len(df))
        task_rows = df.iloc[args.start_idx:end_idx].to_dict(orient="records")
        selection_label = f"tasks={args.start_idx}-{end_idx}"

    print(f"\n{'='*60}")
    print("CHECKPOINT EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint : {args.checkpoint_dir}")
    print(f"Dataset    : {args.dataset}")
    print(f"Selection  : {selection_label}")
    print(f"Compare    : {'yes (checkpoint vs base model)' if args.compare else 'no'}")
    print()

    # ------------------------------------------------------------------ #
    # Evaluate checkpoint
    # ------------------------------------------------------------------ #
    tokenizer, checkpoint_model = load_model_with_checkpoint(args.checkpoint_dir)
    checkpoint_results = _run_tasks(task_rows, checkpoint_model, tokenizer, label="[CHECKPOINT] ")
    print_summary(checkpoint_results)

    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_results_{timestamp}.json")
    save_results(checkpoint_results, checkpoint_path)

    # ------------------------------------------------------------------ #
    # Optionally evaluate base model on the same tasks
    # ------------------------------------------------------------------ #
    if args.compare:
        print(f"\n{'='*60}")
        print("BASE MODEL EVALUATION")
        print(f"{'='*60}")
        print("Unloading checkpoint model to free GPU memory...")
        checkpoint_model = unload_model(checkpoint_model)

        tokenizer, base_model = load_base_model()
        base_results = _run_tasks(task_rows, base_model, tokenizer, label="[BASE] ")
        print_summary(base_results)

        base_path = os.path.join(args.output_dir, f"base_results_{timestamp}.json")
        save_results(base_results, base_path)

        unload_model(base_model)

        # Side-by-side comparison
        print_comparison(checkpoint_results, base_results)
