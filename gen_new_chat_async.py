import random
import json
import asyncio
import aiohttp
import uuid
import os
import time
import gc

from click import prompt
from tqdm import tqdm
from datasets import Dataset

async def get_text_from_response(response):
    if "message" in response["choices"][0]:
        return response["choices"][0]["message"]["content"]
    else:
        return None

async def create_api_request(session, base_url, endpoint, api_key, data):
    url = f"{base_url}/{endpoint}"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    try:
        async with session.post(url, json=data, headers=headers) as resp:
            response = await resp.json()
            if resp.status != 200:
                print(f"Error {resp.status} for request to {url}: {response}")
                return None
            return response
    except Exception as e:
        print(f"Exception during API request to {url}: {e}")
        return None

async def create_chat_completion(session, base_url, api_key, data):
    extra_body = data.setdefault('extra_body', {})
    request_id = extra_body.pop('request_id', str(uuid.uuid4()))
    json_data = {**data, **extra_body}
    if 'extra_body' in json_data:
        del json_data["extra_body"]
    response = await create_api_request(session, base_url, 'chat/completions', api_key, json_data)
    if response is not None:
        return response, request_id
    else:
        return None, request_id

async def process_sample(sample, session, api_key, json_scheme, max_retries=5):
    import uuid
    test = sample['data']
    retry_count = 0
    priority = 0  # 기본 우선순위

    system_message = f"""당신은 새로운 대화를 시작하기 위한 **독립적인 질문**을 작성해야 합니다. 아래의 지침에 따라 작업을 수행하세요.

**질문 작성 지침:**

1. **주제 확장 논리(thought) 작성**:
   - 질문을 작성하기 전에, 주제를 어떻게 확장하거나 초월할지 설명하는 논리를 서술하세요.
   - 논리는 다음과 같은 다양한 관점을 포함할 수 있습니다:
     - 시간적, 공간적, 이론적, 문화적, 기술적, 윤리적, 심리적, 경제적, 환경적, 철학적, 정치적, 예술적, 학제적 관점 등.
   - 세부사항을 그대로 가져오는 것이 적합한 경우(예: 문제의 독립성을 유지하면서도 정확한 맥락이 필요한 경우), 세부사항을 가져와야 하는 구체적인 이유를 Thought에 서술하세요.
   - 또한, 세부사항을 가져오기로 결정했다면, Generated Question에 세부사항을 **반드시** 기록하세요.

2. **독립적인 질문 작성**:
   - **자족적인 질문**:
     - 기존 대화의 세부 내용에 의존하지 않고도 이해될 수 있는 질문이어야 합니다.
     - 필요한 모든 배경 정보와 세부 사항을 Generated Question에 포함하여, 추가 정보 없이도 이해할 수 있도록 하세요.
   - **구체적이고 호기심을 유발**:
     - 모호한 질문 대신 명확하고 구체적인 질문을 작성하세요.
     - 독창적이며 탐구심을 자극하는 내용을 담아주세요.

3. **새로운 방향성 제안**:
   - **주제의 새로운 측면 탐구**:
     - 기존 대화를 반복하지 않고, 관련 주제의 새로운 측면이나 방향성을 제시하세요.
     - 예상치 못한 관점이나 숨겨진 이슈를 드러내는 질문을 만들어 보세요.

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
1. Thought 작성 방법
- 질문을 작성하기 전에, 주제를 어떻게 확장하거나 초월할지 생각하세요.
- 질문의 구성을 위해 예시의 세부사항을 가져오는 것이 적합하다고 판단될 경우, 그 이유를 Thought에 명시하세요.
- 확장 논리를 간결하고 명확하게 서술하세요.

2. 질문 작성 방법
- 세부사항을 가져오기로 결정했다면, Generated Question에 세부사항을 **반드시** 기록하세요.
- Thought를 바탕으로 구체적이고 독립적인 질문을 작성하세요.
- 새로운 독자는 참고용 대화를 활용할 수 없으므로, 새로운 질문은 작성된 것만 보고도 이해될 수 있어야 합니다.

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

    while retry_count <= max_retries:
        data = {
            "model": "mistralai/Mistral-Large-Instruct-2411",
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 1.2,
            "top_p": 0.99,
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

        host = random.choice(["node41", "node42"])
        port = random.choice([1557, 48884])
        host = "node41"
        port = 1557
        base_url = f"http://{host}:{port}/v1"

        response, request_id = await create_chat_completion(session, base_url, api_key, data)

        if response is not None:
            content = await get_text_from_response(response)
            model_name = response.get("model", "")
            try:
                content_json = json.loads(content)
                generated_question = content_json.get("generated_question", "").strip()
                thought = content_json.get("thought", "").strip()
                if generated_question and thought:
                    sample['synthetic_question'] = {
                        model_name: {
                            "task": content_json.get("task", "").strip(),
                            "generated_question": generated_question,
                            "thought": thought,
                        }
                    }
                    break 
                else:
                    # if generated_expaned_question empty, retry
                    retry_count += 1
                    priority -= retry_count
                    continue
            except Exception as e:
                print(f"JSON parsing error: {e}")
                retry_count += 1
                priority -= retry_count
                continue
        else:
            retry_count += 1
            priority -= retry_count
            continue

    else:
        sample['synthetic_question'] = {
            "NO_RESPONSE": {
                "task": "",
                "generated_question": "",
                "thought": "",
            }
        }
    return sample

def save_dataset(dataset_list, save_path):
    dataset = Dataset.from_list(dataset_list)
    dataset.save_to_disk(save_path)

async def main():
    from datasets import load_dataset

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
                "description": "질문을 작성하기 전에 주제를 확장할 논리를 서술"
            },
            "generated_question": {
                "type": "string",
                "description": "구체적이고 호기심을 유발하며, 그 자체로 독립적인 질문"
            }
        },
        "required": ["task", "thought", "generated_question"],
        "additionalProperties": False
    }

    test_data = load_dataset("mirlab/aklmQA-Qwen2.5-72B-instruction-score")["train"]
    model = "mistralai/Mistral-Large-Instruct-2411"

    print(test_data[0])

    max_concurrent_tasks = 140
    max_concurrent_tasks = int(1.1 * max_concurrent_tasks)
    results = []
    save_interval = 5000
    save_path = "synthetic_questions"
    last_save_path = "synthetic_questions/final"
    api_key = "sk-hynix"

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    processed_prompt_ids = set()
    processed_data = []

    # 이전에 저장된 데이터 불러오기
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
            print(f"Loaded {len(processed_prompt_ids)} processed prompt_ids.")
        else:
            print("No saved data found.")

    async def bounded_process_sample(sample):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await process_sample(sample, session, api_key, json_scheme)

    tasks_list = []
    pbar = tqdm(range(len(test_data)), desc="Processing samples", unit="samples", leave=False)
    for idx in pbar:
        sample = test_data[idx]
        prompt_id = sample["data"].get('metadata', {}).get('prompt_id', '')
        if prompt_id in processed_prompt_ids:
            continue
        task = asyncio.create_task(bounded_process_sample(sample))
        tasks_list.append(task)
        time.sleep(0.0001)

    result = dict()
    for idx, task in enumerate(tqdm(asyncio.as_completed(tasks_list), total=len(tasks_list), desc="Querying samples", smoothing=0.01)):
        result = await task
        processed_data.append(result)
        prompt_id = result["data"].get('metadata', {}).get('prompt_id', '')
        if prompt_id:
            processed_prompt_ids.add(prompt_id)
        if idx % save_interval == 0:
            print(f"Saving results at index idx={idx}...")
            results = Dataset.from_list(processed_data)
            results.save_to_disk(f"{save_path}/{len(processed_data)+idx}")
            results = []
            del results
        
        if idx % 500 == 0:
            print(
                f"Processed {idx + 1} samples.\n" \
                f'{result["data"]["data"][1:]}\n' \
                f'{result["synthetic_question"]}' \
            )
        gc.collect()

    print(f"All samples {len(processed_data)} processed.")
    print("Last sample:", result)
    print( 
            f'{result["data"]["data"][1:]}\n' \
            f'{result["synthetic_question"]}' \
        )

    results = Dataset.from_list(processed_data)
    print(results)
    print("Saving final results...")
    results.save_to_disk(last_save_path)
    data = Dataset.load_from_disk(last_save_path)
    print(data)

if __name__ == "__main__":
    import aiohttp
    asyncio.run(main())
