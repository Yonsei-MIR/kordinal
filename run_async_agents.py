import asyncio
import yaml
import gc
import os
import json
import time
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from kordinal.logger_config import logger

from kordinal.data_entry import DataEntry

from kordinal.filter_invalid_data import has_enough_korean, is_korean, has_high_chinese_local_density
from kordinal.client import LoadBalancer, RoundRobinLoadBalancer
from kordinal.client import TaskManager



def parse_user_turn(text: str) -> str:
    import re
    # Pattern to match "User:" with optional asterisks before and/or after it
    # **User:**, User:, **user:**, user:
    pattern = re.compile(r"(?i)(?:\*\*)?user:\s*(.+?)(?:\*\*)?$")
    match = pattern.search(text)
    if match:
        return match.group(1).strip()  # Return the captured text, stripped of leading/trailing whitespace
    return text.strip()  # Return the whole text (stripped) if no recognizable "User:" prefix is found

def get_prompt_id(data: dict):
    if "data" in data:
        if isinstance(data["data"], dict):
            prompt_id = data["data"].get("metadata", {}).get("prompt_id", None)
        else:
            prompt_id = data.get("metadata", {}).get("prompt_id", None)
    else:
        prompt_id = data.get("metadata", {}).get("prompt_id", None)
    return prompt_id


async def main(args):
    balancer_map = {
        "RoundRobinLoadBalancer": RoundRobinLoadBalancer,
    }

    config_path = "./scripts/config.yaml"
    with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    endpoints: dict = config["endpoints"]
    gen_config: dict = config["gen_config"]
    num_endpoints = 0
    # 모델별 로드밸런서 설정
    for model_name, info in endpoints.items():
        if "selector" in info:
            selector_name = info["selector"]
            lb_class = balancer_map.get(selector_name)
            if not lb_class:
                raise ValueError(f"Unknown LoadBalancer: {selector_name}")

            if "endpoints" not in info:
                raise ValueError(f"'{model_name}' has selector but no 'endpoints' list.")

            lb_instance = lb_class(info["endpoints"])
            endpoints[model_name]["selector"] = lb_instance
            num_endpoints += len(info["endpoints"])
        else:
            # 기존 단일 endpoint 형태를 유지
            # endpoints[model_name] = { "host": ..., "port": ... }
            num_endpoints += 1
    print(endpoints)

    models = list(endpoints.keys())
    DataEntry.AVAILABLE_MODEL = models
    logger.info(f"Available models: {models}")
    # input("Press Enter to continue...")
    policy = args.policy
    sequence = args.sequence.split(',') if args.sequence else None
    
    existing_prompt_ids = set()
    
    result_dir = "./results"
    if os.path.exists(result_dir):
        for file_name in os.listdir(result_dir):
            logger.info(f"Existing result file: {file_name}")
            if file_name.endswith(".jsonl"):
                file_path = os.path.join(result_dir, file_name)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            result = json.loads(line)
                            prompt_id = get_prompt_id(result)
                            if prompt_id:
                                existing_prompt_ids.add(prompt_id)
                        except json.JSONDecodeError:
                            continue

    logger.info(f"Existing prompt IDs: {len(existing_prompt_ids)}. Resuming from the last prompt ID.")
    # input("Press Enter to continue...")

    dataset = load_from_disk("./workspace/llm-agent/synthetic_questions/final_filtered")
    num_shard = 1
    skipped = 0
    for shard in range(num_shard):
        task_manager = TaskManager(
            endpoints,
            models,
            gen_config,
            policy=policy,
            sequence=sequence,
            max_workers=int(64*max(num_endpoints, 1)*1.15), # TODO: aqcuire the number of max-num-seqs from the config
            mode="thread"
        )
        batch_size = len(dataset) // num_shard
        # pbar = tqdm(range(len(dataset)), desc="Processing")
        pbar = tqdm(range(shard*batch_size, (shard+1)*batch_size), desc="Processing")
        passed = 0
        task_manager.result_path = f"./results/async_multiturn_results-{shard:05d}-of-{num_shard:05d}.jsonl"
        for idx in pbar:
            if idx > len(dataset):
                break
            
            try:
                data = dataset[idx]
            except Exception as e:
                with open("./dataset-log.log", "a") as f:
                    print(idx, e, file=f)
                continue

            prompt_id = get_prompt_id(data)
            if prompt_id is not None and prompt_id in existing_prompt_ids:
                skipped +=1 
                logger.info(f"Skipping existing prompt ID: {prompt_id}")
                continue

            first_question = data.get("synthetic_question")
            if "NO_RESPONSE" in first_question:
                if first_question["NO_RESPONSE"] is None:
                    pass
                else:
                    # logger.error(f"Skipping DataEntry ID: {request_id}: {data}")
                    passed += 1
                    continue
            for model in first_question.keys():
                if first_question[model] is None:
                    # print(first_question)
                    passed += 1
                    continue
                if first_question[model]["generated_question"] is not None:
                    break
            if first_question[model] is None:
                continue

            question = first_question[model]["generated_question"]
            task = first_question[model]["task"]
            thought = first_question[model]["thought"]

            messages = [{"role": "user", "content": question, "source": model}]
            data["messages"] = messages
        
            data_entry = DataEntry(data, gen_model=model, life=10)
            data_entry.current_model = model
            await task_manager.add_task(data_entry.priority, data_entry)

        logger.info(f"Skipped: {skipped}")
        await task_manager.run()
        del task_manager
        gc.collect()

if __name__ == "__main__":
    # model = "Qwen/Qwen2.5-72B-Instruct"
    # model = "mistralai/Mistral-Large-Instruct-2411"
    # model = "CohereForAI/aya-expanse-32b"
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Initial model name')
    parser.add_argument('--policy', type=str, default='weighted', help='Model selection policy')
    parser.add_argument('--sequence', type=str, help='Model sequence for sequence policy (comma-separated)')
    args = parser.parse_args()
    start = time.time()
    asyncio.run(main(args))
    end = time.time()

    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"Total requests: {1000 * 6}. Average time per 1K request: {(end - start) / (1000 * 6 / 1000.):.2f} seconds")
