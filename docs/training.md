# Preparation

**Step 1:** Prepare the runtime environment and install AReaL.

Please refer to AReaL installation guideline.


**Step 2:** prepare training data

# Train a Search Agent


## C. Fine-tuning a LRM Agent

**Step 1.** Launch Qwen2.5-72B-Instruct for LLM-as-Judge:

```shell
python3 -m areal.launcher.ray ASearcher/train/asearcher_reasoning.py \
    --config ASearcher/configs/asearcher_web_qwq.yaml \
    experiment_name=asearcher-qwen72b-inst-server-only \
    trial_name=run1 \
    cluster.n_nodes=1 allocation_mode=sglang.d2t4p1 \
    actor.path=Qwen/Qwen2.5-72B-Instruct 
```

**Step 2.** Launch QwQ-32B agent training:

```shell
python3 -m areal.launcher.ray \
    ASearcher/train/asearcher_reasoning.py \
    --config ASearcher/configs/asearcher_web_qwq.yaml \
    experiment_name=asearcher-qwq-train \
    trial_name=run1 cluster.n_nodes=6 allocation_mode=sglang.d2t8+d4t8 \
    actor.path=Qwen/QwQ-32B \
    train_dataset.path=path_to_ASearcher-LRM-35k.jsonl \
    judge_engine.experiment_name=asearcher-qwen72b-inst-server-only \
    judge_engine.trial_name=run1
```


