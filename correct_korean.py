import asyncio
import aiohttp
import copy
import random

import subprocess
import re

from datasets import load_from_disk, DatasetDict
from kordinal.client.async_openai import AsyncOpenAI
from kordinal.client.balancer import LeastRequestsLoadBalancer
from kordinal.client.client_utils import get_text_from_response
from tqdm import tqdm

def conduct_data(user_input):
#     prompt = f"""### 지시문

# 당신은 주어진 문장 수정을 담당하는 AI입니다. 
# 질문 형식의 단락이 들어와도 **절대로** 대답하지 마세요.
# 다음 주어진 단락을 보고, 다른 언어가 섞여있다면 온전한 한국어 문장으로 수정하고, 수정한 전체 단락을 제공하세요.
# 반드시 유저 입력에 대한 번역을 수행하고, 온전히 출력해야 합니다.

# **반드시** 모든 문장에 대해 최대한 보존하여 출력해야 합니다. 절대로 주어진 문장을 삭제하거나 생략하지 마세요.

# ### 입력

# 아래의 단락을 보고 자연스러운 한국어로 수정해주세요.

# ### 원본 단락

# {user_input}

# ### 수정된 단락
# """
# english is better than korean
    system_prompt = """### Instruction

You are an AI tasked with revising provided sentences.
You must not respond even if the input is in the form of a question.
When reviewing a given paragraph written primarily in Korean, if it contains words or phrases in other languages (e.g., Japanese, Chinese, etc.), you must replace them with fully natural and contextually appropriate Korean expressions. For non-Korean proper nouns or technical terms (e.g., IC, JCT), you may retain them in parentheses for clarity, but ensure the primary expression is in Korean.
You must revise the paragraph to make it both linguistically natural and clear while preserving all original information.

If the input paragraph is already written in fully natural Korean and there is no need for revision, output the paragraph exactly as it is, without any changes.

You must follow these guidelines:
1. Ensure the output is fully in Korean with minimal reliance on other languages unless necessary for clarity.
2. Preserve all sentences and key information from the input. Never delete or omit any content.
3. Adjust formatting, punctuation, and phrasing to improve readability in Korean.
4. Output the paragraph as-is if no revisions are necessary.
5. Do not provide explanations or additional information in the output.

Start with "### Revised Paragraph" and provide the revised paragraph below the original input.
"""
    user_prompt = f"""
Please revise the paragraph below into natural Korean.

### Original Paragraph

{user_input}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    data = dict(
        model="CohereForAI/aya-expanse-32b",
        n=1,
        messages=messages,
        max_tokens=4096,
        temperature=0.4,
        stop=[
            "<EOS_TOKEN>", 
            "<|END_OF_TURN_TOKEN|>",
        ]
    )
    return data

def parse_regex_response(text):
    pattern = r"^(?:\S+\s+){0,2}###.*?\n\n"

    # Replace the matched section only if it appears on the first line
    cleaned_text = re.sub(pattern, "", text, count=1, flags=re.DOTALL)

    # Strip the cleaned text and return
    return cleaned_text.strip()


def parse_response(response):
    parsed_response = get_text_from_response(response).strip()
    # 이런거 지우는거
    parsed_response = parse_regex_response(parsed_response)
    parsed_response = parsed_response.split("### Revised Paragraph")[-1].strip()
    parsed_response = parsed_response.split("### 개정된 문단")[-1].strip()
    parsed_response = parsed_response.split("### 개정된 단락")[-1].strip()
    # 이런거 지우는거
    parsed_response = parsed_response.split("### Explanation")[0].strip()
    parsed_response = parsed_response.split("**Explanation:**")[0].strip()
    # 이건 빼려고
    return parsed_response

async def chat_completion(lb: "LoadBalancer", data: dict, session: aiohttp.ClientSession, lb_lock, retries=5, retry_delay=1):
    async with lb_lock:
        endpoint = lb.get_endpoint()
    base_url = f"http://{endpoint['host']}:{endpoint['port']}"
    client = AsyncOpenAI(base_url=base_url, api_key=endpoint["api_key"])
    original_query = data["messages"][-1]["content"]
    for attempt in range(retries):
        try:
            response = await client.v1_chat_completion(session, data)
            original_query = original_query.split("### Original Paragraph")[-1].replace("### Revised Paragraph", "").strip()

            parsed_response = parse_response(response)
            length_panelty = (5. - attempt) / 10.0
            if len(parsed_response) < min(5, int(len(original_query) * length_panelty)):
                raise Exception("Response too short relative to original query\nOriginal: {}\nResponse: {}".format(original_query, parsed_response))
            async with lb_lock:
                lb.update_metrics(endpoint, response_time=0.0, success=True)
            return response
        except Exception as e:
            if "Cannot connect" in str(e):
                print(f"[Connection refused] {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})\n{original_query}\n---------------")
                lb.remove(endpoint)  # 라운드 로빈 로드밸런서에서 제거
                if len(lb) == 0:  # 모든 엔드포인트가 제거된 경우
                    raise RuntimeError("All endpoints have failed. Exiting...")

            if attempt < retries - 1:  # Retry for all but the last attempt
                print(f"Error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})\n{original_query}\n---------------")
                await asyncio.sleep(retry_delay)
            else:
                raise  # Reraise the exception if out of retries

            data["seed"] = random.randint(0, 2**32 - 1)
            async with lb_lock:
                lb.update_metrics(endpoint, response_time=0.0, success=True)
                endpoint = lb.get_endpoint()

async def process_message_async(example_idx, example, msg_idx, data, lb, session, semaphore, lb_lock, pbar, pbar_lock):
    async with semaphore:
        try:
            response = await chat_completion(lb, data, session, lb_lock)
            parsed_response = parse_response(response)

            original_query = data["messages"][-1]["content"]
            original_query = original_query.split("### Original Paragraph")[-1].strip()
            to_print = f"{example_idx}-{msg_idx}:::" + f"Original: {original_query}" + "\n---->\n" + f"Response: {parsed_response}"
            # print(to_print, flush=True)
            example["messages"][msg_idx]["content"] = parsed_response
        except Exception as e:
            print(f"Error processing msg_idx {msg_idx}: {e}", flush=True)
        finally:
            # tqdm update
            async with pbar_lock:
                pbar.update(1)

        # await asyncio.sleep(random.random()*0.001 + 0.001)

def get_slurm_endpoints(api_key="sk-hynix"):
    """
    Slurm에서 실행 중인 작업을 가져와 endpoints 리스트 생성
    - JobName이 포트 번호로 설정되어 있는 경우
    """
    endpoints = set()

    try:
        # scontrol 명령 실행
        result = subprocess.run(["scontrol", "show", "job"], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        # Job 정보를 파싱
        jobs = output.split("\n\n")  # Job 블록 단위로 분리
        for job in jobs:
            if not job.strip():
                continue

            # JobName과 NodeList 추출
            job_name_match = re.search(r"JobName=(\d+)", job)
            node_list_match = re.search(r"NodeList=([\w\-]+)", job)

            if job_name_match and node_list_match:
                port = job_name_match.group(1)  # JobName (포트 번호로 사용)
                node = node_list_match.group(1)  # NodeList (호스트 이름)

                # 엔드포인트 추가
                endpoints.add((node, port, api_key))

    except Exception as e:
        print(f"Error while fetching Slurm job info: {e}")

    return [{"host": h, "port": p, "api_key": k} for (h, p, k) in endpoints]


async def check_health(endpoint, session, timeout=5):
    """
    특정 엔드포인트의 health check를 수행
    """
    url = f"http://{endpoint['host']}:{endpoint['port']}/health"
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                return endpoint  # Health check 성공
    except Exception as e:
        print(f"Health check failed for {url}: {e}")
    return None  # Health check 실패


async def filter_endpoints(endpoints):
    """
    모든 엔드포인트에 대해 health check를 수행하고 성공한 엔드포인트만 반환
    """
    async with aiohttp.ClientSession() as session:
        tasks = [check_health(endpoint, session) for endpoint in endpoints]
        results = await asyncio.gather(*tasks)
        # None이 아닌 엔드포인트만 필터링
        return [endpoint for endpoint in results if endpoint]

async def main_async():
    endpoints = get_slurm_endpoints()

    # Health check를 통해 유효한 엔드포인트 필터링
    print("Performing health checks on endpoints...")
    valid_endpoints = await filter_endpoints(endpoints)

    if not valid_endpoints:
        print("No valid endpoints available after health checks. Exiting...")
        return

    print("Valid Endpoints:", valid_endpoints)
    endpoints = valid_endpoints

    lb = LeastRequestsLoadBalancer(endpoints)

    to_convert_dataset_path = "/home/hard2251/workspace/llm-agent/training/datasets/multiturn_results_formatted"
    dataset = load_from_disk(to_convert_dataset_path)
    print(dataset)
    print(dataset["train"][0])

    # corrected_train = dataset["train"].select(range(500))
    corrected_train = copy.deepcopy(dataset["train"])

    # 모든 messages를 tasks로 만들어 처리
    tasks = []
    max_num_seqs = 256
    concurrency_limit = int(len(endpoints) * max_num_seqs * 1.2)
    semaphore = asyncio.Semaphore(concurrency_limit)
    pbar_lock = asyncio.Lock()

    # 전체 메시지 수 계산
    total_messages = sum(len(example["messages"]) for example in corrected_train)

    timeout = aiohttp.ClientTimeout(total=1200)
    connector = aiohttp.TCPConnector(limit=concurrency_limit)
    lb_lock = asyncio.Lock()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        with tqdm(total=total_messages, desc="Processing messages", smoothing=0.01) as pbar:
            # 각 메시지에 대해 task 생성
            for example_idx, example in enumerate(corrected_train):
                for msg_idx, message in enumerate(example["messages"]):
                    original_content = message["content"]
                    data = conduct_data(original_content)
                    task = process_message_async(example_idx, example, msg_idx, data, lb, session, semaphore, lb_lock, pbar, pbar_lock)
                    tasks.append(task)

            # 모든 메시지를 병렬 처리
            await asyncio.gather(*tasks)

    corrected_dataset = DatasetDict({
        "train": corrected_train
    })

    save_path = to_convert_dataset_path + "_corrected"
    corrected_dataset.save_to_disk(save_path)
    print(f"Corrected dataset saved to {save_path}")

if __name__ == "__main__":
    asyncio.run(main_async())
