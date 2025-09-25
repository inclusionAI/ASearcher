import argparse
import os

import evaluate
from utils import set_seed
from llm_utils import get_sglang_llm
from openai import AsyncOpenAI
from config_loader import get_api_key, load_config_and_set_env

import asyncio



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="hotpotqa_500", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen__Qwen2.5-72B-Instruct", type=str)
    parser.add_argument("--use-openai", default=False, type=eval, choices=[True, False])
    parser.add_argument("--judge-prompt", type=str, default="default")
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="none", type=str)
    parser.add_argument("--agent-type", default="r1-searcher", type=str)
    parser.add_argument("--search-client-type", default="search-r1", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seeds", default=None, type=str, help="Comma-separated list of seeds to process (overrides --seed)")
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
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.top_k = -1 if args.temperature == 0 else args.top_k

    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    args.data_parallel_size = len(available_gpus) // args.tensor_parallel_size
    return args

def llm_as_judge(args):

    print("Loading configuration...")
    load_config_and_set_env()
    
    # Determine which seeds to process
    if args.seeds:
        seeds_to_process = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds_to_process = [args.seed]
    
    set_seed(seeds_to_process[0])  # Use first seed for initialization

    if not args.use_openai:
        args.model_name_or_path = "Qwen__Qwen2.5-72B-Instruct"
        args.tensor_parallel_size=4
        args.data_parallel_size=2
        llm = get_sglang_llm(args)
        print("LLM model loaded successfully")
    else:
        args.model_name_or_path = "gpt-4o-mini-2024-07-18"

        openai_api_key = get_api_key('openai_api_key') or os.environ.get("OPENAI_API_KEY", '')
        openai_api_base = get_api_key('openai_api_base') or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/")
        
        if not openai_api_key:
            raise RuntimeError("Warning: OpenAI API key is not set. Please configure it in config.yaml or set the OPENAI_API_KEY environment variable.")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_BASE"] = openai_api_base
        
        llm = AsyncOpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base
                )
        print("OpenAI client initialized successfully")

    # Process each dataset
    for data_name in args.data_names.split(","):
        print(f"\nRunning LLM-as-Judge for {data_name}")
        
        output_dir = args.output_dir
        eval_dir = f"agent_eval_{args.max_tokens_per_call}"
        
        # Process each seed for this dataset
        for i, seed in enumerate(seeds_to_process):
            print(f"  Processing seed {i + 1}...")
            
            if args.parallel_mode == "seed":
                out_file_prefix = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_seed{seed}_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
            else:
                out_file_prefix = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_split{args.split_id}_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
            
            out_file = f"{output_dir}/{eval_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_n{args.n_sampling}.jsonl"
            
            # Check if file exists before processing
            if not os.path.exists(out_file):
                print(f"    Warning: Result file not found: {out_file}")
                continue
                
            print(f"    Processing file: {os.path.basename(out_file)}")
            try:
                asyncio.run(evaluate.llm_as_judge_async(out_file, llm, args.model_name_or_path, args.use_openai, args.judge_prompt))
                print(f"    ✅ Completed seed {i + 1}")
            except Exception as e:
                print(f"    ❌ Error processing seed {i + 1}: {e}")
    
    print(f"\n LLM-as-Judge completed for all datasets")

if __name__ == "__main__":
    args = parse_args()
    llm_as_judge(args)