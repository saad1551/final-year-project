import json
import os
import time
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

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks attempted : {len(results)}")
    print(f"Successful runs       : {len(successful)}")
    print(f"Failed runs           : {len(results) - len(successful)}")
    print()
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
        j = r["judgment"]
        s = f"{j.success:.2f}" if j.success is not None else "-"
        e = f"{j.efficiency:.2f}" if j.efficiency is not None else "-"
        sc = f"{j.self_correction:.2f}" if j.self_correction is not None else "-"
        instr = r["task_instruction"][:50] + "..." if len(r["task_instruction"]) > 50 else r["task_instruction"]
        print(f"{i+1:<4} {s:<9} {e:<9} {sc:<12} {r['num_steps']:<6} {instr}")


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
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of test tasks to evaluate",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index in the test dataset",
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

    # Set debug flag in pipeline module
    if args.debug:
        import pipeline_in_steps
        pipeline_in_steps.DEBUG_PIPELINE = True
        print("[DEBUG] Debug mode enabled")

    print(f"{'='*60}")
    print("CHECKPOINT EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint  : {args.checkpoint_dir}")
    print(f"Dataset     : {args.dataset}")
    print(f"Tasks       : {args.num_tasks} (starting at index {args.start_idx})")
    print()

    # Load model with checkpoint
    tokenizer, model = load_model_with_checkpoint(args.checkpoint_dir)

    # Load test dataset
    df = pd.read_csv(args.dataset)
    print(f"Dataset loaded: {len(df)} tasks")

    results = []

    for i in range(args.num_tasks):
        task_idx = args.start_idx + i
        if task_idx >= len(df):
            print(f"Reached end of dataset at index {task_idx}")
            break

        task_row = df.iloc[task_idx].to_dict()

        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{args.num_tasks} (Dataset Index: {task_idx})")
        print(f"Instruction: {task_row['instruction']}")
        print(f"Website: {task_row['website']}")
        print(f"{'='*60}")

        result = run_eval_trajectory(task_row, model, tokenizer)
        results.append(result)

        if result:
            print(f"\n--- Task {i+1} Complete: {result['num_steps']} steps ---")
        else:
            print(f"\n--- Task {i+1} FAILED ---")

    # Print aggregate summary
    print_summary(results)
