import argparse
import json
import os
import time
import random
import numpy as np
from glob import glob
from collections import defaultdict

import evaluate
from tqdm import tqdm
from llm_utils import get_sglang_llm
from transformers import AutoTokenizer
from utils import set_seed, prepare_data, load_jsonl
from typing import Dict, Any, Optional, List

import asyncio
import subprocess
import os
from agent import make_agent
from tools.search_utils import make_search_client
from config_loader import load_config_and_set_env

# Try to import prettytable, fall back to plain output if not installed
try:
    from prettytable import PrettyTable
    PRETTYTABLE_AVAILABLE = True
except ImportError:
    PRETTYTABLE_AVAILABLE = False
    print("[WARNING] prettytable not found. Results will be displayed in JSON format. Install with: pip install prettytable")

class CompatibleLLMResponse:
    def __init__(self, text: str, input_len: Optional[int] = None, 
                 input_tokens: Optional[List[int]] = None,
                 output_len: Optional[int] = None,
                 output_tokens: Optional[List[int]] = None,
                 output_logprobs: Optional[List[float]] = None,
                 output_versions: Optional[List[int]] = None):
        self.text = text
        self.input_len = input_len
        self.input_tokens = input_tokens or []
        self.output_len = output_len
        self.output_tokens = output_tokens or []  
        self.output_logprobs = output_logprobs or []
        self.output_versions = output_versions or []

def compute_average(results, metric):
    """Calculate average of all results"""
    values = []
    for v in results.values():
        if isinstance(v[metric], list) and len(v[metric]) > 0:
            values.extend(v[metric])
        elif isinstance(v[metric], (int, float)):
            values.append(v[metric])
    
    if not values:
        return np.nan
    return np.mean(values)

def compute_max(results, metric, n):
    """Calculate max value for pass@k"""
    ret = []
    for k, v in results.items():
        if isinstance(v[metric], list):
            # Ensure enough samples
            if len(v[metric]) >= n:
                ret.append(v[metric][:n])
            else:
                # Use existing data if not enough samples
                ret.append(v[metric])
        else:
            # Single value case
            ret.append([v[metric]])
    
    if not ret:
        return np.nan
    
    # Calculate maximum for each question, then take average
    max_scores = []
    for question_results in ret:
        if question_results:
            max_scores.append(max(question_results))
    
    return np.mean(max_scores) if max_scores else np.nan

def aggregate_multiple_runs(data_name, base_dir, args, n_sampling, tokenizer=None):
    """Aggregate results from multiple runs"""
    # Build file pattern to find all relevant result files
    eval_dir = f"agent_eval_{args.max_tokens_per_call}"
    cur_dir = os.path.join(base_dir, eval_dir, data_name)
    
    # Each sampling file has n_sampling=1 for Pass@k evaluation
    file_n_sampling = 1
    
    if args.parallel_mode == "seed":
        pattern = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_seed*_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}_s{args.start}_e{args.end}_n{file_n_sampling}.jsonl"
    else:
        pattern = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_split*_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}_s{args.start}_e{args.end}_n{file_n_sampling}.jsonl"
    
    file_pattern = os.path.join(cur_dir, pattern)
    files = glob(file_pattern)
    
    if not files:
        return {}
    
    aggregated_results = defaultdict(lambda: defaultdict(list))
    metrics = ["F1", "EM", "CEM"]
    gen_lens = []
    doc_lens = []
    num_searchs = []
    num_accesses = []
    
    total_samples = 0
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if not data:  # Skip empty lines
                        continue
                    
                    question_id = data.get("id", str(total_samples))
                    
                    for metric in metrics:
                        if metric in data:
                            aggregated_results[question_id][metric].append(data[metric])
                    
                    if "history" in data and tokenizer is not None:
                        history = data["history"]
                        
                        # Extract generated text from unified agent v2 format
                        gen_text = "".join([h["text"] for h in history if h["type"] == "llm_response" and "text" in h])
                        
                        # Extract document text from unified agent v2 format
                        doc_text = ""
                        for h in history:
                            if h["type"] == "search_result":
                                # Extract documents from search results
                                if "documents" in h and isinstance(h["documents"], list):
                                    doc_text += " ".join([str(doc) for doc in h["documents"]])
                        
                        # Count search operations (unified agent v2 format)
                        num_search = len([h for h in history if h["type"] == "search_result"])
                        
                        # Count page access operations (unified agent v2 format)
                        num_access = len([h for h in history if h["type"] == "page_access"])
                        
                        # Debug info for search/access counting (show for first few samples)
                        if total_samples < 3:  # Only show for first few samples
                            history_types = [h.get("type", "unknown") for h in history]
                            print(f"  [DEBUG] Sample {total_samples}: history types = {set(history_types)}, count = {len(history)}")
                            print(f"    Calculated: num_search = {num_search}, num_access = {num_access}")
                            # Show first few history entries for debugging
                            for i, h in enumerate(history[:3]):
                                print(f"    History[{i}]: type='{h.get('type', 'none')}', text_preview='{h.get('text', '')[:100]}...'")
                        
                        # Calculate token length or character length
                        try:
                            if gen_text:
                                gen_len = tokenizer([gen_text], return_length=True)['length'][0]
                                gen_lens.append(gen_len)
                            if doc_text:
                                doc_len = tokenizer([doc_text], return_length=True)['length'][0]
                                doc_lens.append(doc_len)
                        except Exception:
                            gen_lens.append(len(gen_text))
                            doc_lens.append(len(doc_text))
                        
                        num_searchs.append(num_search)
                        num_accesses.append(num_access)
                    
                    total_samples += 1
                    
        except Exception:
            continue
    
    if args.llm_as_judge:
        metrics.append("MBE")
        if args.parallel_mode == "seed":
            judge_pattern = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_seed*_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}_s{args.start}_e{args.end}_n{file_n_sampling}-llm_as_judge_{args.judge_prompt}_use_openai_{args.use_openai}.jsonl"
        else:
            judge_pattern = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_split*_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}_s{args.start}_e{args.end}_n{file_n_sampling}-llm_as_judge_{args.judge_prompt}_use_openai_{args.use_openai}.jsonl"
        
        judge_file_pattern = os.path.join(cur_dir, judge_pattern)
        judge_files = glob(judge_file_pattern)
        
        for judge_file_path in judge_files:
            try:
                with open(judge_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if not data:  # Skip empty lines
                            continue
                        
                        question_id = data.get("id", "unknown")
                        
                        if "MBE" in data:
                            aggregated_results[question_id]["MBE"].append(data["MBE"])
                            
            except Exception:
                continue
    
    final_results = {}
    if aggregated_results:
        final_results["num"] = len(aggregated_results)
        for metric in metrics:
            final_results[metric] = compute_average(aggregated_results, metric)
            final_results[f"{metric}.Max@{n_sampling}"] = compute_max(aggregated_results, metric, n_sampling)
        if gen_lens:
            final_results["avg_gen_len"] = np.mean(gen_lens)
        if doc_lens:
            final_results["avg_doc_len"] = np.mean(doc_lens)
        if num_searchs:
            final_results["avg_num_searchs"] = np.mean(num_searchs)
        if num_accesses:
            final_results["avg_num_access"] = np.mean(num_accesses)
    
    return final_results

def format_results_table(all_results):
    if not all_results:
        return "No results to display"
    
    if not PRETTYTABLE_AVAILABLE:
        return json.dumps(all_results, indent=2)
    
    table = PrettyTable()
    
    # Define the desired field order
    first_result = next(iter(all_results.values()))
    
    # Base fields in desired order
    ordered_fields = [
        "num", 
        "avg_gen_len", 
        "avg_doc_len", 
        "avg_num_searchs", 
        "avg_num_access",
        "F1", 
        "EM", 
        "CEM", 
        "MBE"
    ]
    
    # Add Max@k fields dynamically based on what's available
    max_fields = []
    for key in first_result.keys():
        if key.startswith(("F1.Max@", "EM.Max@", "CEM.Max@", "MBE.Max@")):
            max_fields.append(key)
    
    # Sort Max@k fields to maintain consistent order (F1, EM, CEM, MBE)
    max_fields.sort(key=lambda x: (
        0 if x.startswith("F1.Max@") else
        1 if x.startswith("EM.Max@") else  
        2 if x.startswith("CEM.Max@") else
        3 if x.startswith("MBE.Max@") else 4
    ))
    
    # Only include fields that actually exist in the results
    field_names = ["dataset"] + [field for field in ordered_fields + max_fields if field in first_result]
    table.field_names = field_names
    
    for dataset_name, result in all_results.items():
        formatted_values = []
        for field in field_names[1:]:
            value = result.get(field, "-")
            if isinstance(value, (int, float)) and not np.isnan(value):
                formatted_values.append(f"{value:.3f}")
            else:
                formatted_values.append(str(value))
        table.add_row([dataset_name] + formatted_values)
    
    # Add average row if multiple datasets
    if len(all_results) > 1:
        formatted_values = []
        for field in field_names[1:]:
            values = [v.get(field, np.nan) for v in all_results.values() if isinstance(v.get(field), (int, float))]
            if values and not all(np.isnan(values)):
                avg_value = np.nanmean(values)
                formatted_values.append(f"{avg_value:.3f}")
            else:
                formatted_values.append("-")
        table.add_row(["Average"] + formatted_values)
    
    return str(table)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="hotpotqa_500", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen__Qwen3-1.7B/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="none", type=str)
    parser.add_argument("--agent-type", default="areal-search-reasoning-v2", type=str)
    parser.add_argument("--search-client-type", default="search-r1", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-splits", default=1, type=int)
    parser.add_argument("--split-id", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max-tokens-per-call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--parallel-mode", type=str, default="seed", choices=["seed", "split"])
    parser.add_argument("--use-jina", action="store_true", help="use jieba to get webpage content")
    parser.add_argument("--jina-api-key", type=str, help="jina api key")
    parser.add_argument("--concurrent", type=int, default=64, help="concurrent requests of evaluation")
    parser.add_argument("--llm_as_judge", action="store_true", help="Enable LLM-as-judge evaluation")
    parser.add_argument("--judge-prompt", type=str, default="default", help="Judge prompt type for LLM-as-judge")
    parser.add_argument("--use-openai", default=False, type=eval, choices=[True, False], help="Use OpenAI for LLM-as-judge")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    # Pass@k related parameters (now the only evaluation mode)
    parser.add_argument("--pass-at-k", type=int, default=1, help="Number of samples for pass@k evaluation (default=1)")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing results without running new evaluation")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    ) 
    args.top_k = -1 if args.temperature == 0 else args.top_k

    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    args.data_parallel_size = len(available_gpus) // args.tensor_parallel_size
    return args

async def main(args):

    print("Loading configuration...")
    load_config_and_set_env()

    data_list = args.data_names.split(",")
    
    # If only aggregating existing results, skip new evaluation
    if args.aggregate_only:
        print("Aggregating existing results mode...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        except Exception:
            tokenizer = None
        
        all_results = {}
        for data_name in data_list:
            result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
            if result:
                all_results[data_name] = result
        
        if all_results:
            print("\n" + "="*60)
            print("Aggregated Results:")
            print("="*60)
            print(format_results_table(all_results))
        else:
            print("No aggregatable result files found")
        return

    # Pass@k evaluation mode (now the only mode)
    print(f"Pass@{args.pass_at_k} evaluation mode")
    
    # Generate random seeds for sampling diversity
    base_seed = args.seed
    random.seed(base_seed)  # Ensure reproducibility of seed generation
    sampling_seeds = [base_seed]  # Always include the original seed
    if args.pass_at_k > 1:
        # Generate additional random seeds
        while len(sampling_seeds) < args.pass_at_k:
            new_seed = random.randint(0, int(1e7))
            if new_seed not in sampling_seeds:
                sampling_seeds.append(new_seed)
    
    original_n_sampling = args.n_sampling
    args.n_sampling = 1  # Always use n_sampling=1 for Pass@k
    
    for i, current_seed in enumerate(sampling_seeds):
        print(f"Running sampling {i+1}/{args.pass_at_k}...")
        args.seed = current_seed
        
        llm = None
        try:
            llm = get_sglang_llm(args)
            semaphore = asyncio.Semaphore(args.concurrent)
            tasks = [eval_agent(semaphore, llm, data_name, args) for data_name in data_list]
            results = await asyncio.gather(*tasks)
            
        except Exception as e:
            print(f"Sampling {i+1} failed: {e}")
            continue
        
        finally:
            if llm is not None:
                try:
                    if hasattr(llm, 'shutdown'):
                        llm.shutdown()
                    elif hasattr(llm, 'close'):
                        llm.close()
                    del llm
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"Warning: Error while releasing GPU memory: {e}")
        
        print(f"Sampling {i+1} completed")
    
    args.n_sampling = original_n_sampling
    args.seed = base_seed  # Restore original seed
    print(f"\nAggregating {args.pass_at_k} sampling results...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception:
        tokenizer = None
    
    all_results = {}
    for data_name in data_list:
        result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
        if result:
            all_results[data_name] = result

    if args.llm_as_judge:
        print("\n" + "="*50)
        print("Starting LLM-as-judge evaluation for all datasets...")
        print("="*50)
        
        print("Releasing GPU memory from sglang LLM...")
        try:
            if 'llm' in locals() and llm is not None:
                if hasattr(llm, 'shutdown'):
                    llm.shutdown()
                elif hasattr(llm, 'close'):
                    llm.close()
                del llm
                print("LLM instance released successfully")
            else:
                print("LLM instance was not found or already released")
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU cache cleared successfully")
                
        except Exception as e:
            print(f"Warning: Error while releasing GPU memory: {e}")

        # Prepare seeds for LLM-as-judge (use the same random seeds generated earlier)
        if 'sampling_seeds' in locals():
            seeds_str = ",".join([str(s) for s in sampling_seeds])
        else:
            # Fallback for aggregate_only mode - generate seeds using the same logic
            base_seed = args.seed
            random.seed(base_seed)
            sampling_seeds = [base_seed]
            if args.pass_at_k > 1:
                while len(sampling_seeds) < args.pass_at_k:
                    new_seed = random.randint(0, 999999)
                    if new_seed not in sampling_seeds:
                        sampling_seeds.append(new_seed)
            seeds_str = ",".join([str(s) for s in sampling_seeds])

        # Run LLM-as-judge once for all datasets and seeds
        try:
            cmd = [
                "python", "llm_as_judge.py",
                "--data_names", ",".join(data_list),
                "--data_dir", args.data_dir,
                "--model_name_or_path", args.model_name_or_path,
                "--output_dir", args.output_dir,
                "--prompt_type", args.prompt_type,
                "--agent-type", args.agent_type,
                "--search-client-type", args.search_client_type,
                "--split", args.split,
                "--num_test_sample", str(args.num_test_sample),
                "--seeds", seeds_str,  # Pass all seeds at once
                "--n-splits", str(args.n_splits),
                "--split-id", str(args.split_id),
                "--start", str(args.start),
                "--end", str(args.end),
                "--temperature", str(args.temperature),
                "--n_sampling", "1",  # Always 1 for Pass@k evaluation
                "--top_p", str(args.top_p),
                "--top_k", str(args.top_k),
                "--max-tokens-per-call", str(args.max_tokens_per_call),
                "--parallel-mode", args.parallel_mode,
                "--tensor_parallel_size", str(args.tensor_parallel_size),
                "--judge-prompt", args.judge_prompt,
                "--use-openai", str(args.use_openai)
            ]
            
            # Add optional flags
            if args.shuffle:
                cmd.append("--shuffle")
            if args.save_outputs:
                cmd.append("--save_outputs")
            if args.overwrite:
                cmd.append("--overwrite")
            if args.use_safetensors:
                cmd.append("--use_safetensors")
            if args.apply_chat_template:
                cmd.append("--apply_chat_template")
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=".")
            if result.returncode != 0:
                print(f"LLM-as-judge failed with return code: {result.returncode}")
            else:
                print(f"LLM-as-judge completed successfully for all datasets and seeds")
                
        except Exception as e:
            print(f"Error running LLM-as-judge: {e}")
    
    if args.llm_as_judge:
        print(f"\nRe-aggregating results to include LLM-as-judge MBE scores...")
        
        if 'tokenizer' not in locals() or tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            except Exception:
                tokenizer = None
        
        all_results = {}
        for data_name in data_list:
            result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
            if result:
                all_results[data_name] = result

    if all_results:
        print("\n" + "="*60)
        print(f"Pass@{args.pass_at_k} Final Results:")
        print("="*60)
        print(format_results_table(all_results))
        
        eval_dir = f"agent_eval_{args.max_tokens_per_call}"
        result_path = os.path.join(args.output_dir, eval_dir, f"aggregate_results_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_t{args.temperature:.1f}.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {result_path}")
    else:
        print("\nNo results to display")

async def process_single_llm_query(llm, tokenizer, prompt: str, sampling_params: Dict, args, qid=None) -> CompatibleLLMResponse:
    # Build sampling kwargs based on new agent's requirements
    sampling_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=sampling_params.get("max_new_tokens", args.max_tokens_per_call),
        n=1,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else [tokenizer.pad_token_id, tokenizer.eos_token_id]
        ),
    )
    
    # Handle agent's stop strings
    if sampling_params.get("stop") and isinstance(sampling_params["stop"], list):
        stop_strings = sampling_params["stop"]
        
        if stop_strings == ["</think>"]:
            # Thinking mode: stop at </think>
            tokens = tokenizer.encode("</think>", add_special_tokens=False)
            existing_stops = sampling_kwargs.get("stop_token_ids", [])
            sampling_kwargs["stop_token_ids"] = existing_stops + tokens
            sampling_kwargs["stop"] = ["</think>"]
    
    try:
        output = await llm.async_generate(prompt, sampling_kwargs)
    except ValueError as e:
        print("ValueError when handling query {}".format(qid))
        raise e

    # Create compatible response object for agent v2
    text = output['text'] if isinstance(output, dict) else output
    
    # Post-process: truncate at first complete tool call for tool generation mode
    if sampling_params.get("stop") and sampling_params["stop"] != ["</think>"]:
        text = truncate_at_first_complete_tool_call(text)
    
    # Try to extract additional information if available
    input_tokens = tokenizer.encode(prompt) if tokenizer else None
    output_tokens = tokenizer.encode(text) if tokenizer and text else None
    
    print("[DEBUG] input length:", len(input_tokens), flush=True)

    return CompatibleLLMResponse(
        text=text,
        input_len=len(input_tokens) if input_tokens else None,
        input_tokens=input_tokens,
        output_len=len(output_tokens) if output_tokens else None,
        output_tokens=output_tokens,
        output_logprobs=None,  # Not available from current LLM
        output_versions=None   # Not available from current LLM
    )

async def process_single_search_query(search_client, query: str, topk: int = 5) -> Any:
    req_meta = {
        "queries": [query],
        "topk": topk,
        "return_scores": False
    }
    results = await search_client.query_async(req_meta)
    return results if results else None

async def process_single_access_query(search_client, url: str) -> Any:
    results = await search_client.access_async([url])
    return results if results else None

def truncate_at_first_complete_tool_call(text: str) -> str:
    """Truncate text at the first complete tool call"""
    import re
    
    patterns = [
        r'(<search>.*?</search>)',
        r'(<access>.*?</access>)', 
        r'(<answer>.*?</answer>)'
    ]
    
    earliest_end = len(text)
    found_tool_call = False
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tool_call_end = match.end()
            if tool_call_end < earliest_end:
                earliest_end = tool_call_end
                found_tool_call = True
    
    return text[:earliest_end] if found_tool_call else text

def convert_agent_tool_calls_to_dict(agent_tool_calls):
    """Convert agent tool calls to dict format"""
    import re
    
    dict_tool_calls = []
    
    for tool_call_str in agent_tool_calls:
        # Parse <search>...</search>
        search_match = re.search(r'<search>(.*?)</search>', tool_call_str, re.DOTALL)
        if search_match:
            dict_tool_calls.append({"type": "search", "query": search_match.group(1).strip()})
            continue
            
        # Parse <access>...</access>
        access_match = re.search(r'<access>(.*?)</access>', tool_call_str, re.DOTALL)
        if access_match:
            dict_tool_calls.append({"type": "access", "url": access_match.group(1).strip()})
            continue
            
        # Parse <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', tool_call_str, re.DOTALL)
        if answer_match:
            dict_tool_calls.append({"type": "answer", "content": answer_match.group(1).strip()})
            continue
    
    return dict_tool_calls


async def process_single_work_item(semaphore, agent_type, llm, tokenizer, search_client, args, out_dir, process):
    """Process a single work item using agent v2"""
    async with semaphore:
        if "history" not in process:
            process["history"] = []
            process["running"] = True
            process["num_turns"] = 0

        # Create fresh agent instance for thread safety
        agent = make_agent(agent_type)
        agent.initialize_with_prompt(process)
        
        # Set tokenizer for V1 agents that need it
        if hasattr(agent, 'set_tokenizer'):
            agent.set_tokenizer(tokenizer)
        
        
        while process["running"] and agent.num_turns < agent.max_turns:
            # Check if agent is finished
            if agent.is_finished:
                process["running"] = False
                break
            
            try:
                # Get LLM query from agent
                prompt_or_messages, sampling_params = agent.prepare_llm_query()

                assert not agent.is_finished

                if isinstance(prompt_or_messages, str):
                    prompt = prompt_or_messages
                
                    # Process LLM query
                    llm_response = await process_single_llm_query(llm, tokenizer, prompt, sampling_params, args, qid=process["id"])
                    completion_text = llm_response.text
                
                # Let agent consume LLM response and get tool calls
                tool_calls_raw = agent.consume_llm_response(llm_response, completion_text)
                tool_calls = convert_agent_tool_calls_to_dict(tool_calls_raw)

                assert not agent.is_finished
                
                # Log progress
                if tool_calls:
                    print(f"Process {process['id']}: {', '.join([tc['type'] for tc in tool_calls])}")
                
                # Add to history in unified agent v2 format
                process["history"].append({
                    "type": "llm_response", 
                    "text": completion_text,
                    "tool_calls": tool_calls
                })
                
                # Process each tool call
                for tool_call in tool_calls:
                    if tool_call["type"] == "search":
                        search_result = await process_single_search_query(search_client, tool_call["query"])
                        if search_result:
                            # Handle different search result formats
                            if isinstance(search_result, dict):
                                documents = search_result.get("documents", []) or []
                                urls = search_result.get("urls", []) or []
                            elif isinstance(search_result, list):
                                documents = []
                                urls = []
                                for result in search_result:
                                    if isinstance(result, dict):
                                        result_docs = result.get("documents", []) or []
                                        result_urls = result.get("urls", []) or []
                                        documents.extend(result_docs)
                                        urls.extend(result_urls)
                            else:
                                documents = []
                                urls = []
                            
                            # Ensure we don't pass None values
                            documents = documents or []
                            urls = urls or []
                            
                            # Provide search result to agent in its expected format
                            tool_response = {
                                "type": "search",
                                "documents": documents,
                                "urls": urls
                            }
                            agent.consume_tool_response(tool_response)
                            
                            # Add to unified history in agent v2 format (regardless of agent's internal format)
                            process["history"].append({
                                "type": "search_result",
                                "query": tool_call["query"],
                                "documents": documents,
                                "urls": urls
                            })
                    
                    elif tool_call["type"] == "access":
                        access_result = await process_single_access_query(search_client, tool_call["url"])
                        if access_result:
                            if isinstance(access_result, dict):
                                page = access_result.get("page", "") or ""
                            elif isinstance(access_result, str):
                                page = access_result or ""
                            else:
                                page = str(access_result) if access_result else ""
                            
                            # Ensure we don't pass None values
                            page = page or ""
                            
                            # Provide page access result to agent in its expected format
                            tool_response = {
                                "type": "access",
                                "page": page
                            }
                            agent.consume_tool_response(tool_response)
                            
                            # Add to unified history in agent v2 format (regardless of agent's internal format)
                            process["history"].append({
                                "type": "page_access",
                                "url": tool_call["url"],
                                "page": page
                            })
                    
                    elif tool_call["type"] == "answer":
                        # Agent has provided final answer
                        process["pred_answer"] = tool_call["content"]
                        process["running"] = False
                        break
                
                process["num_turns"] = agent.num_turns
                
                assert not agent.is_finished
                
            except Exception as e:
                print(f"Error processing work item {process['id']}: {e}")
                process["running"] = False
                process["error"] = str(e)
                break

            # Save intermediate state
            with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
                # Include agent memory for debugging
                process_copy = process.copy()
                if hasattr(agent, "current_process"):
                    process_copy = agent.current_process.copy()
                if hasattr(agent, 'memory') and agent.memory:
                    process_copy["agent_memory"] = agent.memory.to_dict()
                    process_copy["agent_stats"] = agent.memory.logging_stats()
                json.dump(process_copy, f, ensure_ascii=False)

        # Ensure we have a final answer
        if "pred_answer" not in process and hasattr(agent, 'get_answer'):
            final_answer = agent.get_answer()
            if final_answer:
                process["pred_answer"] = final_answer
            else:
                process["pred_answer"] = ""
        
        # Save final state
        with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
            # Include agent memory for debugging
            process_copy = process.copy()
            if hasattr(agent, "current_process"):
                process_copy = agent.current_process.copy()
            if hasattr(agent, 'memory') and agent.memory:
                process_copy["agent_memory"] = agent.memory.to_dict()
                process_copy["agent_stats"] = agent.memory.logging_stats()
            json.dump(process_copy, f, ensure_ascii=False)
        
        return process

async def eval_agent(semaphore, llm, data_name, args):
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    search_client = make_search_client(args.search_client_type, args.use_jina, args.jina_api_key)
    processes, out_dir = prepare_data(data_name, args, save_async=True)
    
    start_time = time.time()
    
    # Create tasks with agent_type instead of shared agent instance to avoid memory pollution
    tasks = [process_single_work_item(semaphore, args.agent_type, llm, tokenizer, search_client, args, out_dir, p) for p in processes]
    processes = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        processes.append(await f)
    
    out_file = out_dir + ".jsonl"
    with open(out_file, "w") as f:
        for process in processes:
            f.write(json.dumps(process, ensure_ascii=False) + "\n")

    # Extract answers for evaluation
    answers = []
    for process in processes:
        if "pred_answer" in process:
            answers.append(process["pred_answer"])
        else:
            answers.append("")

    eval_metrics = {
        "F1": evaluate.compute_score_f1, 
        "EM": evaluate.compute_score_em,
        "CEM": evaluate.cover_exact_match_score_1,
    }
    result_json = {k: [] for k in eval_metrics.keys()}
    for process, answer in zip(processes, answers):
        for k, fn in eval_metrics.items():
            if isinstance(process["gt"], list) or isinstance(process["gt"], tuple):
                process[k] = max([fn(answer, g) for g in process["gt"]])
            else:
                process[k] = fn(answer, process["gt"])
            result_json[k].append(process[k])
    for k in eval_metrics.keys():
        result_json[k] = np.mean(result_json[k])

    with open(out_file, "w") as f:
        for process in processes:
            f.write(json.dumps(process, ensure_ascii=False) + "\n")

    time_use = time.time() - start_time
    print("time_use", time_use)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    print(args.model_name_or_path + "@" + data_name)
    print(result_json)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    
    return result_json  

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    asyncio.run(main(args))