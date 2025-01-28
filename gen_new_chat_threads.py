import random
import json
import threading
import requests
import uuid
import os
import time
import gc
import queue

from openai import OpenAI

from tqdm import tqdm
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from kordinal.server_utils import EndpointSelector

def get_text_from_response(response):
    if "message" in response["choices"][0]:
        return response["choices"][0]["message"]["content"]
    else:
        return None

def create_api_request(session, base_url, endpoint, api_key, data):
    url = f"{base_url}/{endpoint}"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    try:
        resp = session.post(url, json=data, headers=headers)
        if resp.status_code != 200:
            if "This model's maximum context length" in resp.text:
                print(f"Error {resp.status_code} for request to {url}: {resp.text}")
                return False
            print(f"Error {resp.status_code} for request to {url}: {resp.text}")
            return None
        response = resp.json()
        return response
    except Exception as e:
        print(f"Exception during API request to {url}: {e}")
        return None

def create_chat_completion(base_url, api_key, data):
    extra_body = data.setdefault('extra_body', {})
    client = OpenAI(base_url=base_url, api_key=api_key)
    request_id = extra_body.pop('request_id', str(uuid.uuid4()))
    try:
        response = client.chat.completions.create(**data)
    except Exception as e:
        print(f"Exception during API request: {base_url}, {e}")
        return None, request_id

    return response, request_id

def process_sample(sample, api_key, json_scheme, endpoint, max_retries=3):
    import uuid
    test = sample['data']
    retry_count = 0
    priority = 0  # 기본 우선순위
    # endpoints = [("node41", 48880), ("node41", 48881), ("node42", 48882), ("node42", 48883)]
    system_message = f"""당신은 새로운 대화를 시작하기 위한 **독립적인 질문**을 작성해야 합니다. 아래의 지침에 따라 작업을 수행하세요.

**질문 작성 지침:**

1. **주제 확장 논리(thought) 작성**:
   - 질문을 작성하기 전에, 주어진 대화를 보고 주제를 어떻게 확장하면 좋을 지 먼저 생각해보세요.
   - 논리는 다음과 같은 다양한 관점을 포함할 수 있습니다:
     - 시간적, 공간적, 이론적, 문화적, 기술적, 윤리적, 심리적, 경제적, 환경적, 철학적, 정치적, 예술적, 학제적 관점 등.
     - 이밖에 다양한 관점을 포함할 수 있습니다.
   - 필요한 세부사항을 구성하는 과정을 Thought에 명확하게 서술하세요.
   - 세부사항을 그대로 가져오는 것이 적합한 경우(예: 문제의 독립성을 유지하면서도 정확한 맥락이 필요한 경우), 세부사항을 가져와야 하는 이유를 Thought에 간결히 서술하세요.

2. **독립적인 질문 작성**:
   - **자족적인 질문**:
     - 기존 대화의 의존하지 않고도 이해될 수 있는 질문이어야 합니다.
     - 필요한 모든 배경 정보와 세부 사항을 Generated Question에 **반드시** 포함하여, 추가 정보 없이도 이해할 수 있도록 하세요.
        - 이는 새로운 독자가 참고용 대화를 활용할 수 없다는 것을 의미합니다.
   - **구체적이고 호기심을 유발**:
     - 모호한 질문 대신 명확하고 구체적인 질문을 작성하세요.
     - 독창적이며 탐구심을 자극하는 내용을 담아주세요.

3. **새로운 방향성 제안**:
   - **주제의 새로운 측면 탐구**:
     - 기존 대화를 반복하지 않고, 관련 주제의 새로운 측면이나 방향성을 제시하세요.
     - 예상치 못한 관점이나 숨겨진 이슈를 드러내는 질문을 만들어 보세요.
   - **난이도 조절**:
     - 새로운 질문은 때떄로 어렵거나 도전적일 수 있습니다. 이를 통해 새로운 관점을 탐구하고, 창의적인 사고를 유도하세요.

4. 질문은 지정된 태스크를 따릅니다.
**태스크 예시**:
- 텍스트 추출 (text_extraction): 특정 정보나 문장을 추출하는 질문 작성
- 창의적 콘텐츠 생성 (creative_content): 창의적이고 독창적인 질문 작성
- 분석적 추론 (analytical_reasoning): 논리적으로 분석하고 새로운 관점을 도출하는 질문 작성
- 두뇌 자극 문제 해결 (brain_teaser): 퍼즐이나 직관적 사고를 유도하는 질문 작성
- 텍스트 분류 (text_classification): 정보를 카테고리화하거나 분류하는 질문 작성
- RAG (rag): 추가 정보를 검색하고 활용하는 질문 작성
- 페르미 문제 해결 (fermi): 대략적인 추정을 요구하는 질문 작성
- 객관식 문제 생성/해결 (mcq): 새로운 객관식 문제 작성
- 단계적 사고 과정 (fs_cot_flow): 사고를 단계적으로 전개하는 질문 작성
- 코드 관련 태스크 (code_): 코드 작성이나 코드와 관련된 질문 작성
- 텍스트 수정 (text_modification): 텍스트를 수정하거나 요약하는 질문 작성
- 구조 데이터를 텍스트로 변환 (struct2text_flow): 구조화된 정보를 텍스트로 표현하는 질문 작성
- 후속 질문 생성/답변 (follow_up): 주제를 확장하거나 후속 질문을 생성하는 질문 작성
- 오픈 도메인 질의응답 (open_domain_qa): 다양한 주제를 탐구하는 질문 작성
- 독해 이해 (rc): 특정 텍스트를 읽고 의미를 파악하는 질문 작성
"""

    user_new_prompt = f"""아래는 User와 Assistant의 대화입니다. 이 대화를 참고하여 새로운 대화를 시작하기 위한 질문을 작성하세요.

### 작성 절차:
0. 태스크 선택
- 위의 태스크 예시를 참고하여, 새로운 질문을 작성할 태스크를 선택하세요.

1. Thought 작성 방법
- 질문을 작성하기 전에, 주제를 어떻게 Task에 맞게 확장할 수 있는지 생각해보세요.
- 확장 논리를 간결하고 명확하게 서술하세요.
- 만들 질문에 대해 세부사항을 어떻게 구성할 지 Thought에 명확하게 서술하세요.

2. 질문 작성 방법
- Thought를 바탕으로 구체적이고 독립적인 질문을 작성하세요.
- 질문은 그 자체로 완전해야 합니다. Generated Question에 필요한 모든 세부사항을 **반드시** 기록하세요.
    - 새로운 독자는 참고용 대화를 활용할 수 없으므로, 새로운 질문은 작성된 것만 보고도 이해될 수 있어야 합니다.
- 완성된 질문은, Task을 수행하기에 충분한 정보를 **반드시** 포함하고 있어야 합니다.
- 태스크를 수행할 수 있는, 적절한 질문을 작성하세요.

---

### 대화 (참고용):
User: {test['data'][1]['content']}
Assistant: {test['data'][2]['content']}
### 끝
"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_new_prompt}
    ]
    model = "mistralai/Mistral-Large-Instruct-2411"

    while retry_count <= max_retries:
        model = endpoint["model"]
        host = endpoint["host"]
        port = endpoint["port"]

        base_url = f"http://{host}:{port}/v1"
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 1.1,
            "top_p": 0.995,
            "extra_body": {
                "top_k": -1,
                "min_p": 0.15,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "priority": priority,
                "request_id": str(uuid.uuid4()),
                "guided_json": json_scheme,
                "stop": [
                    "<|endoftext|>",
                    "[INST]",
                    "[/INST]",
                    "<|im_end|>",
                    "<|end|>",
                    "<|eot_id|>",
                    "<end_of_turn>",
                    "<eos>",
                    "**Assistant:**", "**assistant:**",
                    "Assistant:", "assistant:",
                    "---",
                ]
            }
        }

        response, request_id = create_chat_completion(base_url, api_key, data)
        synthetic_question = sample.get('synthetic_question', {})

        if response is None:
            retry_count += 1
            priority -= retry_count
            continue
        
        if response is False:
            print(f"Error: Maximum context length exceeded. {request_id}", flush=True)
            synthetic_question[model] = {
                    "task": "NO_RESPONSE",
                    "generated_question": "MAX_CONTEXT_LENGTH_EXCEEDED",
                    "thought": "MAX_CONTEXT_LENGTH_EXCEEDED"
            }
            sample['synthetic_question'] = synthetic_question
            return sample
    
        response = response.dict()

        try:
            # content is a JSON string
            content = json.loads(response["choices"][0]["message"]["content"])
            # print(content, type(content))
            # print("parsed:", content.get("task", ""), content.get("generated_question", ""), content.get("thought", ""))
            model_name = response['model']
            task = content.get("task", "").strip()
            generated_question = content.get("generated_question", "").strip()
            thought = content.get("thought", "").strip()
            if len(generated_question) > 10 and len(thought) > 10:
                synthetic_question[model_name] = {
                        "task": task,
                        "generated_question": generated_question,
                        "thought": thought,
                }
                break
        except Exception as e:
            print(f"Exception during processing response: {e}. Retrying... Request ID: {request_id}", flush=True)
            retry_count += 1
            priority -= retry_count

    if retry_count > max_retries:
        synthetic_question[model] = {
                "task": "NO_RESPONSE",
                "generated_question": "NO_RESPONSE",
                "thought": "NO_RESPONSE"
        }
    sample['synthetic_question'] = synthetic_question
    return sample

def save_dataset(dataset_list, save_path):
    dataset = Dataset.from_list(dataset_list)
    dataset.save_to_disk(save_path)

def main():
    from datasets import load_dataset
    endpoints = [
        {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "host": "localhost",
            "port": 48880
        }, 
        {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "host": "localhost",
            "port": 48881
        }, 
        {
            "model": "mistralai/Mistral-Large-Instruct-2411",
            "host": "localhost",
            "port": 48882
        }, 
        {
            "model": "mistralai/Mistral-Large-Instruct-2411",
            "host": "localhost",
            "port": 48883
        },
    ]
    print("Checking endpoint health...")
    
    num_checkpoints = 0
    selector = EndpointSelector(endpoints)
    
    while True:
        try:
            host, port = selector.available()
            print(f"endpoint alive: {host}:{port}")
            num_checkpoints += 1 
        except Exception as e:
            print(f"{e}. Waiting...")
            time.sleep(60)
        if num_checkpoints >= len(endpoints):
            break
    
    tasks_en = [
        "text_extraction",
        "creative_content",
        "analytical_reasoning",
        "brain_teaser",
        "text_classification",
        "rag",
        "fermi",
        "mcq",
        "fs_cot_flow",
        "code_",
        "text_modification",
        "struct2text_flow",
        "follow_up",
        "open_domain_qa",
        "rc"
    ]

    json_scheme = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "enum": tasks_en,
            },
            "thought": {
                "type": "string",
                "description": "질문을 작성하기 전에 주제를 확장할 방법을 서술. 앞서 선택한 태스크를 수행하기 적합한 질문을 작성."
            },
            "generated_question": {
                "type": "string",
                "description": "구체적이고 호기심을 유발하며, 그 자체로 독립적인 질문"
            }
        },
        "required": ["task", "thought", "generated_question"],
        "additionalProperties": False
    }

    test_data = load_dataset("dataset")["train"]
    model = "mistralai/Mistral-Large-Instruct-2411"

    print(test_data[0])

    max_concurrent_tasks = 128 * 4
    max_concurrent_tasks = int(1.1 * max_concurrent_tasks)
    save_interval = 5000
    save_path = "synthetic_questions"
    last_save_path = "synthetic_questions/final"
    api_key = "sk-hynix"

    processed_prompt_ids = set()
    processed_data = []
    processed_data_lock = threading.Lock()

    # load previous datasets
    if os.path.exists(last_save_path):
        print("Loading previously saved data...")
        saved_dataset = Dataset.load_from_disk(last_save_path)
        if saved_dataset:
            print(saved_dataset)
            for idx in range(len(saved_dataset)):
                sample = saved_dataset[idx]
                prompt_id = sample.get("data", {}).get('metadata', {}).get('prompt_id', '')
                processed_models = sample.get('synthetic_question', {}).keys()
                if (prompt_id not in processed_prompt_ids) and (model in processed_models):
                    processed_prompt_ids.add(prompt_id)
                    processed_data.append(sample)
            # print(f"Loaded {len(processed_prompt_ids)} processed prompt_ids.")
        else:
            print("No saved data found.")


    samples_queue = queue.Queue()
    skipped = 0

    for idx in range(len(test_data)):
        sample = test_data[idx]
        prompt_id = sample["data"].get('metadata', {}).get('prompt_id', '')
        if prompt_id in processed_prompt_ids:
            skipped += 1
            continue
        samples_queue.put(sample)
        
    # pbar.close()
    total_samples = samples_queue.qsize()
    pbar = tqdm(total=total_samples, desc="Processing samples", unit="samples", smoothing=0.01)

    def worker(endpoint):
        while True:
            try:
                sample = samples_queue.get_nowait()
            except queue.Empty:
                break
            result = process_sample(sample, api_key, json_scheme, endpoint)
            with processed_data_lock:
                processed_data.append(result)
                
                prompt_id = result["data"].get('metadata', {}).get('prompt_id', '')
                if prompt_id:
                    processed_prompt_ids.add(prompt_id)

                if len(processed_data) % 2000 == 0:
                    print(f"Saving results at {len(processed_data)}...")
                    results_dataset = Dataset.from_list(processed_data)
                    results_dataset.save_to_disk(f"{save_path}/{len(processed_data)}")

                pbar.update(1)
                if len(processed_data) % 100 == 0:
                    print(
                        f"Processed {len(processed_data)} samples.\n" \
                        f'{result["data"]["data"][1:]}\n' \
                        f'{result["synthetic_question"]}', flush=True
                    )
            samples_queue.task_done()

    print(f"Skipped {skipped} samples.")

    workers = []
    num_workers_per_endpoint = 120 
    for idx in range(num_workers_per_endpoint * len(endpoints)):
        endpoint = endpoints[idx % len(endpoints)]
        t = threading.Thread(target=worker, args=(endpoint,))
        workers.append(t)

    for t in workers:
        t.start()

    for t in workers:
        t.join()

    pbar.close()
    print(f"All samples {len(processed_data)} processed.")

    results_dataset = Dataset.from_list(processed_data)
    print("Saving final results...")
    results_dataset.save_to_disk(last_save_path)
    data = Dataset.load_from_disk(last_save_path)
    print(data)


if __name__ == "__main__":
    main()
