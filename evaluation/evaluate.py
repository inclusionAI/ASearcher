# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import json
import glob
import os
import asyncio
import numpy as np
from tqdm import tqdm
import ast
import time

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0: #1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(answer, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if answer is None:
        return None, 0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_subem(answer, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth):
            return score
        else:
            return format_score

def normalize_text(text: str) -> str:
    """Preprocess text for NQ dataset scoring
    
    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{})
    3. Remove extra whitespace
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces and convert to lowercase
    text = text.strip().lower()
    return text

def f1_score(answer_content, gt):
    answer_content = normalize_text(answer_content)
    gt = normalize_text(gt)

    # Tokenize both the answer and reference answer
    pred_tokens = set(answer_content.split())
    gt_tokens = set(gt.split())
    
    if not gt_tokens:  # Avoid division by zero
        return 0
    if not pred_tokens:
        return 0
    
    # Calculate the number of common tokens
    common_tokens = pred_tokens & gt_tokens
    
    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
    
    # Calculate F1 score
    f1 = 0
    if precision + recall > 0:  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_score_f1(answer, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if answer is None:
        return None, 0
    else:
        ret_score = f1_score(answer, ground_truth)
        return ret_score

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    return float(all(ground in pre_list for ground in ground_list))

class JudgeTemplate:

    JUDGE_PROMPT = ""

    def cal_metrics(self, raw_response):
        """ 
        return True, False or Invalid
        """
        raise NotImplementedError

class DefaultJudge(JudgeTemplate):
    
    JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {gt_answer}

Predicted Answer: {pred_answer}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent.

The output should in the following json format:
```json
{{
    "rationale": your rationale for the judgement, as a text,
    "judgement": your judgement result, can only be "Correct" or "Incorrect",
}}
```
"""

    def cal_metrics(self, raw_response):
        mbe = None
        for parse_fn in [json.loads, ast.literal_eval]:
            try:
                mbe = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
                break
            except:
                print(f"[WARNING] Error parsing {[raw_response]}")

        if mbe is None and '"judgement": "incorrect"' in raw_response.lower():
            return False
        if mbe is None and ('"judgement": "correct"' in raw_response.lower() or '"judgement": correct' in raw_response.lower()):
            return True
        if mbe is None:
            return "Invalid"
        return "judgement" in mbe and mbe["judgement"].lower() == "correct" 


JUDGE_DICT = {
    "default": DefaultJudge,
}

async def llm_as_judge_async(fname_pattern, llm, model_path="Qwen__Qwen2.5-72B-Instruct", use_openai=False, judge_prompt="default"):
    from transformers import AutoTokenizer

    judge = JUDGE_DICT[judge_prompt]()

    filenames = glob.glob(fname_pattern)
    data = []
    filesizes = []
    for filename in filenames:
        _raw_data = [json.loads(ff) for ff in open(filename)]
        llm_as_judge_filename = filename.replace(".jsonl", f"-llm_as_judge_{judge_prompt}_use_openai_{use_openai}.jsonl")
        print("[DEBUG] loading", filename, llm_as_judge_filename, os.path.exists(llm_as_judge_filename))
        _data = _raw_data
        if os.path.exists(llm_as_judge_filename):
            _previous_results = [json.loads(ff) for ff in open(llm_as_judge_filename)]
            exist_ix = [r["id"] for r in _previous_results]
            for d in _raw_data:
                if d["id"] not in exist_ix:
                    _previous_results.append(d)
            _data = _previous_results
        filesizes.append(len(_data))
        data.extend(_data)
    print(f"Loaded {len(data)} datapoints")

    queries = []
    for d in data:
        if "MBE" in d and d["llm_as_judge"]["status"] == "success":
            d["MBE"] = float("judgement" in d["llm_as_judge"] and d["llm_as_judge"]["judgement"] == "correct")
            # print(f"skip {d["id"]}: {d["llm_as_judge"]["status"], d["llm_as_judge"]["judgement"]}")
            continue
        if "</" in d["pred_answer"]:
            d["pred_answer"] = d["pred_answer"].split("</")[0][:1000]
        queries.append(d)
    
    print(f"{len(queries)} queries in total", flush=True)
    
    if len(queries) == 0:
        print("MBE: {:.3f}".format(np.mean([d["MBE"] for d in data])))
        for filename, filesize in zip(filenames, filesizes):
            llm_as_judge_filename = filename.replace(".jsonl", f"-llm_as_judge_{judge_prompt}_use_openai_{use_openai}.jsonl")
            with open(llm_as_judge_filename, "w") as f:
                for d in sorted(data[:filesize], key=lambda x: x["id"]):
                    _=f.write(json.dumps(d, ensure_ascii=False) + "\n")
            data = data[filesize:]
        return
    
    # print(queries[0], flush=True)

    semaphore = asyncio.Semaphore(256) if not use_openai else asyncio.Semaphore(10)

    async def process_single_work_item(semaphore, query):
        async with semaphore:

            prompt = judge.JUDGE_PROMPT.format(question=query["question"], gt_answer=query["gt"], pred_answer=query["pred_answer"][:500])
            response = ""
            score = None

            if not use_openai:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                prompt = tokenizer.decode(tokenizer([prompt], add_special_tokens=False)["input_ids"][0][:30000], skip_special_tokens=False)

                retry_cnt = 0
                
                while retry_cnt < 10:
                    if retry_cnt:
                        print(f"Retry: {retry_cnt}")
                    response = await llm.async_generate(prompt, dict(temperature=0.6, top_p=0.95, max_new_tokens=1024))
                    response = response["text"]
                    score = judge.cal_metrics(response)
                    if score != "Invalid":
                        break
            else:
                retry_cnt = 0
                while True:
                    try:
                        if retry_cnt:
                            print(f"Retry: {retry_cnt}")
                        response = await llm.chat.completions.create(
                            **{
                                "model": model_path,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 1024,
                                "temperature": 0.6,
                            }
                        )
                        response = response.choices[0].message.content
                        score = judge.cal_metrics(response)
                        if score != "Invalid":
                            break
                    except Exception as e:
                        retry_cnt += 1
                        print(f"Error: {e}")
                        time.sleep(5)


            if eval(query["id"]) == 1:
                print([prompt], [response], flush=True)
        
        return (query["id"], response, score)
    
    tasks = [process_single_work_item(semaphore, query) for query in queries]
    responses = dict()
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async LLM-as-Judge Eval", ):
        res = await f
        responses[res[0]] = (res[1], res[2])

    print(list(responses.values())[0], flush=True)
    
    import ast
    for d in data:
        if "MBE" in d and d["llm_as_judge"]["status"] == "success":
            continue
        # raw_response = responses[d["id"]]
        # res = judge.cal_metrics(raw_response)
        raw_response, res = responses[d["id"]]
        if res == "Invalid":
            mbe = dict(status="failed")
            score = 0
        else:
            mbe = {
                "judgement": "correct" if res else "incorrect",
                "status": "success"
            }
            score = float(res)
        mbe["raw_response"] = raw_response

        d["MBE"] = score
        d["llm_as_judge"] = mbe
    print("MBE: {:.3f}".format(np.mean([d["MBE"] for d in data])))
    for filename, filesize in zip(filenames, filesizes):
        llm_as_judge_filename = filename.replace(".jsonl", f"-llm_as_judge_{judge_prompt}_use_openai_{use_openai}.jsonl")
        with open(llm_as_judge_filename, "w") as f:
            for d in sorted(data[:filesize], key=lambda x: x["id"]):
                _=f.write(json.dumps(d, ensure_ascii=False) + "\n")
        data = data[filesize:]
