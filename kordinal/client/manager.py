
import asyncio
import aiohttp
import json
import time
import random
import requests

from concurrent.futures import ThreadPoolExecutor

from kordinal.logger_config import logger
from kordinal.client.constants import PRICING
from kordinal.client.client_utils import build_base_url, get_text_from_response
from kordinal.client.async_openai import AsyncOpenAI
from kordinal.client.balancer import LoadBalancer, RoundRobinLoadBalancer
from kordinal.filter_invalid_data import is_korean, has_high_local_density

async def retry_request(session, endpoint, data, max_tries=3, delay=0.1):
    for attempt in range(1, max_tries + 1):
        try:
            response = await endpoint(session, data)
            if response:
                return response
        except Exception as e:
            logger.error(f"Exception in retry_request (Attempt {attempt}/{max_tries}): {e}", exc_info=True)
        await asyncio.sleep(delay)
        # delay += 1
    return None

class TaskManager:
    result_path = "async_multiturn_results.jsonl"
    
    def __init__(
        self,
        endpoints,
        models,
        model_gen_configs,
        policy='random',
        sequence=None,
        max_workers=256,
        mode="async",
        loop=None
    ):
        self.endpoints = endpoints  # Use the endpoints directly
        self.models = models
        self.model_gen_configs = model_gen_configs
        
        self.policy = policy
        self.sequence = sequence or []
        self.sequence_index = 0
        self.max_workers = max_workers
        
        self.loop = loop if loop is not None else asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.connector = aiohttp.TCPConnector(limit=self.max_workers)
        self.timeout = aiohttp.ClientTimeout(total=1200) # task takes quite a long time. becuase scheduling is based on priority
        self.task_queue = asyncio.PriorityQueue()
        
        self.active_tasks = 0
        self.estimated_num_task = 0
        self.resolved_num_task = 0
        self.failed_num_task = 0
        
        self.lock = asyncio.Lock()
        self.save_lock = asyncio.Lock()
        self.metric_lock = asyncio.Lock()
        
        self.endpoints_nums = {model: 0 for model in self.models}
        self.model_weights = {model: 0 for model in self.models}
        for model in self.models:
            if "weight" in self.endpoints[model]:
                self.model_weights[model] += self.endpoints[model]["weight"]
                logger.info(f"Model {model} has weight {self.endpoints[model]['weight']}")
            elif "selector" in self.endpoints[model]:
                self.model_weights[model] += len(self.endpoints[model]["selector"].endpoints)
                self.endpoints_nums[model] += len(self.endpoints[model]["selector"].endpoints)
            else:
                self.model_weights[model] += 1
                self.endpoints_nums[model] += 1
        self.total_endpoints = float(sum(self.model_weights.values()))
        self.model_weights = [(v / self.total_endpoints) for v in self.model_weights.values()]

        self.mode = mode
        match self.mode:
            case "thread":
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            case "async":
                self.executor = None
            case _:
                raise ValueError(f"Unknown mode: {mode}")

        self.warmup()
        logger.info("TaskManager initialized.")
        logger.info("Model weights: " + str(self.model_weights))

    def warmup(self):
        logger.info("[WARMUP] Starting warmup requests to endpoints...")

        test_message = [
            {"role": "user", "content": "Say it, You are an AI."},
            {"role": "assistant", "content": "No, I am an human."},
            {"role": "user", "content": "What is your name?"},
        ]

        for model_name in self.models:
            for _ in range(self.endpoints_nums[model_name]):
                endpoint_info = self.next_endpoint(model_name)
                host, port = endpoint_info.get("host", None), endpoint_info.get("port", None)
                base_url = build_base_url(host, port)

                # or just simply health check
                url = f"{base_url}/v1/chat/completions"
                headers = {
                    'Authorization': f'Bearer {endpoint_info["api_key"]}',
                    'Content-Type': 'application/json',
                }
                data = {
                    "model": model_name,
                    "messages": test_message,
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "guided_json": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["AI", "human"],
                            },
                            "answer": {
                                "type": "string",
                            }
                        },
                        "required": ["name", "answer"],
                        "additionalProperties": False
                    }
                }

                logger.info(f"[WARMUP] Sending test request to model: {model_name} at {url}")
                try:
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        logger.info(f"[WARMUP] Success for model {model_name}, response: {response.text}")
                    else:
                        logger.warning(f"[WARMUP] Non-200 status {response.status_code} for model {model_name}, response: {response.text}")
                except Exception as e:
                    logger.error(f"[WARMUP] Exception while sending request to model {model_name}: {e}")

        logger.info("[WARMUP] Warmup requests completed.")

    def next_endpoint(self, model_name):
        if "selector" in self.endpoints[model_name]:
            selector: LoadBalancer = self.endpoints[model_name]["selector"]
            return selector.get_endpoint()
        else:
            return self.endpoints[model_name]

    def select_next_model(self, current_model):
        if self.policy == 'same':
            return current_model
        elif self.policy == 'weighted':
            return random.choices(self.models, weights=self.model_weights)[0]
        elif self.policy == 'random':
            return random.choice(self.models)
        elif self.policy == 'sequence':
            model = self.sequence[self.sequence_index % len(self.sequence)]
            self.sequence_index += 1
            return model
        elif self.policy == 'no':
            return None
        else:
            return current_model

    async def add_task(self, priority: int, data_entry: "DataEntry", original=True):
        if original:
            async with self.metric_lock:
                self.estimated_num_task += 1
        await self.task_queue.put((priority, data_entry))

    async def run(self):
        async with aiohttp.ClientSession(timeout=self.timeout, connector=self.connector) as session:
            if self.mode == "thread":
                workers = [asyncio.create_task(self.worker_thread(session)) for _ in range(self.max_workers)] # It could be a bottleneck
                await self.task_queue.join()

                while self.active_tasks > 0:
                    logger.info(f"Waiting for all tasks to complete. Active tasks remaining: {self.active_tasks}")
                    await asyncio.sleep(5) 
                
                for w in workers:
                    w.cancel()
                    try:
                        await w
                    except asyncio.CancelledError:
                        pass
            elif self.mode == "async":
                tasks = [asyncio.create_task(self.worker_async(session)) for _ in range(self.max_workers)]
                await self.task_queue.join()
                
                while self.active_tasks > 0:
                    logger.info(f"Waiting for all tasks to complete. Active tasks remaining: {self.active_tasks}")
                    await asyncio.sleep(5)
                
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def worker_thread(self, session):
        while True:
            try:
                priority, data_entry = await self.task_queue.get()
            except asyncio.CancelledError:
                break
            async with self.semaphore:
                async with self.lock:
                    self.active_tasks += 1
                logger.info(f"Active tasks: {self.active_tasks} / Max: {self.max_workers}. {self.resolved_num_task}-{self.failed_num_task}-{self.estimated_num_task} resolved.")
                try:
                    # run_in_executor를 통해 handle_task를 스레드에서 처리
                    await self.loop.run_in_executor(self.executor, self.handle_task_thread, session, data_entry)
                except Exception as e:                    
                    logger.error(f"Exception in worker for DataEntry ID {data_entry.prompt_id}: \n------------------------\n{data_entry.messages}\n------------------------\n{e}")
                    await self.save_failed_data_entry(data_entry)
                finally:
                    self.task_queue.task_done()
                    async with self.lock:
                        self.active_tasks -= 1

    async def worker_async(self, session):
        while True:
            try:
                priority, data_entry = await self.task_queue.get()
            except asyncio.CancelledError:
                break
            async with self.semaphore:
                async with self.lock:
                    self.active_tasks += 1
                logger.info(f"Active tasks: {self.active_tasks} / Max: {self.max_workers}. {self.resolved_num_task}-{self.failed_num_task}-{self.estimated_num_task} resolved.")

                try:
                    await self.handle_task_async(session, data_entry)
                except Exception as e:
                    logger.error(f"Exception in worker for DataEntry ID {data_entry.prompt_id}: \n------------------------\n{data_entry.messages}\n------------------------\n{e}")
                    await self.save_failed_data_entry(data_entry)

                finally:
                    self.task_queue.task_done()
                    async with self.lock:
                        self.active_tasks -= 1

    def handle_task_thread(self, session, data_entry: "DataEntry", retry=5):
        # use asyncio.run() to run async function in thread
        async def run_with_session():
            await self.handle_task_async(session, data_entry, retry=retry)
        time.sleep(random.random() * 0.001 + 0.001) # random delay 
        future = asyncio.run_coroutine_threadsafe(run_with_session(), self.loop)
        future.result()

    async def handle_task_async(self, session, data_entry: "DataEntry", retry=5):
        current_model = data_entry.current_model
        next_model = self.select_next_model(current_model)
        logger.info(f"Processing DataEntry ID: {data_entry.prompt_id}, current_model: {current_model}, next_model: {next_model}, life: {data_entry.life}")

        if not next_model: # 다음 모델이 없으면 더 이상 진행 불가
            logger.info(f"No further processing for DataEntry ID: {data_entry.prompt_id}")
            await self.save_data_entry(data_entry)
            return

        data_entry.current_model = next_model
        model_gen_config = self.model_gen_configs[next_model]

        endpoint_info: dict = self.next_endpoint(next_model)
        host, port = endpoint_info.get("host", None), endpoint_info.get("port", None)
        base_url = build_base_url(host, port)
        api_key = endpoint_info.get('api_key', '')
        logger.info(f"Endpoint: {base_url}")

        try:
            data = data_entry.get_next_turn_data(next_model, data_entry.messages, model_gen_config)
            messages = data["messages"]
            data.pop("request_id", None)
            seed = random.randint(0, 2**32 - 1)
            data["seed"] = seed
            
            # logger.info(f"DataEntry ID: {data_entry.prompt_id}, Data: {data}, Last Role: {data_entry.last_role}, Seed: {seed}")
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)

            if "gemini" in next_model:
                response = await retry_request(session, client.chat_completion, data)
            else:
                response = await retry_request(session, client.v1_chat_completion, data)

            logger.info(f"DataEntry ID: {data_entry.prompt_id}, Last Role: {data_entry.last_role}, Seed: {seed}\nData: {data}\nMessages: {messages}\nResponse: {response}\n---------------------------------------------")

            # 응답 체크
            if not response:
                # 재시도 불가능한 상황
                if retry > 0:
                    logger.error(f"[RETRYING] No response for DataEntry ID: {data_entry.prompt_id}, retrying... (retry left: {retry-1})\nResponse: {response}")
                    return await self.handle_task_async(session, data_entry, retry=retry-1)
                else:
                    logger.error(f"[RETRYING] Failed to process DataEntry ID: {data_entry.prompt_id} after all retries.")
                    await self.save_failed_data_entry(data_entry)
                    return

            if "This model's maximum context length" in response.get("text", ""): # 최대 컨텍스트 길이 초과 에러 처리
                logger.error(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, maximum context length exceeded. Ending.")
                await self.save_failed_data_entry(data_entry)
                return

            text = get_text_from_response(response).strip()
            assert isinstance(text, str), f"Parsed response text must be a string. But got type: {type(text)}\n{text}"
            
            ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            ### Filtering  
            
            if len(text) < 5: # 최소 길이 제한
                if retry > 0:
                    logger.info(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, response too short, retrying... (Remaining retries: {retry-1})\nText: {text}")
                    return await self.handle_task_async(session, data_entry, retry=retry-1)
                else:
                    logger.error(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, response too short, no retries left.")
                    await self.save_failed_data_entry(data_entry)
                    return          

            if not is_korean(text): # 비한국어 텍스트 처리
                if retry > 0:
                    logger.info(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, non-korean text detected, retrying... (Remaining retries: {retry-1})\nText: {text}")
                    return await self.handle_task_async(session, data_entry, retry=retry-1)
                else:   
                    logger.error(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, non-korean text detected, no retries left.")
                    await self.save_failed_data_entry(data_entry)
                    return

            for lang in ["zh", "jp"]:
                if has_high_local_density(lang, text, window_size=32, density_threshold=0.4):
                    # at least 30% of the text is chinese in 32 characters
                    if retry > 0:
                        logger.info(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, high {lang} local density detected, retrying... (Remaining retries: {retry-1})\nText: {text}")
                        return await self.handle_task_async(session, data_entry, retry=retry-1)
                    else:
                        logger.error(f"[RETRYING] DataEntry ID: {data_entry.prompt_id}, high {lang} local density detected, no retries left.")
                        await self.save_failed_data_entry(data_entry)
                        return

            ### Filtering  
            ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # 정상 응답 처리
            data_entry.append(text, source=next_model)

            if data_entry.is_alive():
                data_entry.consume()
                logger.info(f"Prompt ID: {data_entry.prompt_id}, {current_model} -> {next_model}. Priority: {data_entry.priority}")
                data_entry.priority -= 1
                await self.add_task(data_entry.priority, data_entry, original=False)
            else: # life 모두 소진 -> 저장 후 종료
                await self.save_data_entry(data_entry)

        except Exception as e: # 예상치 못한 예외 발생 시 재시도
            if retry > 0:
                logger.error(f"[RETRYING] Exception in handle_task_{self.mode} for DataEntry ID {data_entry.prompt_id}: {e}\nRetrying... (retry left: {retry-1})\nMessages: {data_entry.messages}")
                return await self.handle_task_async(session, data_entry, retry=retry-1)
            else:
                logger.error(f"[RETRYING] Exception in handle_task_{self.mode} for DataEntry ID {data_entry.prompt_id}, no retries left: {e}")
                await self.save_failed_data_entry(data_entry)

    async def save_failed_data_entry(self, data_entry: "DataEntry"):
        async with self.metric_lock:
            self.failed_num_task += 1
        async with self.save_lock:
            # Save the failed entry for further analysis
            logger.info(f"Saving failed DataEntry ID: {data_entry.prompt_id}")
            with open("failed_tasks.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(data_entry.conduct_save()) + "\n")

    async def save_data_entry(self, data_entry: "DataEntry"):
        async with self.metric_lock:
            self.resolved_num_task += 1

        msg = data_entry.messages
        gen_messages = ""
        for m in msg:
            gen_messages += f"{m['role']}: {m['content']}\n"

        async with self.save_lock:      
            logger.info(f"DataEntry ID: {data_entry.prompt_id} saved.\nTask: {data_entry.task}\nMessages: \n\n{gen_messages}\n---------------------------------------------")

            with open(self.result_path, 'a', encoding="utf-8") as f:
                f.write(json.dumps(data_entry.conduct_save()) + '\n')
        logger.info(f"DataEntry ID: {data_entry.prompt_id} saved.")
            
    def __str__(self):
        return f"TaskManager(active_tasks={self.active_tasks}, \nmodel_gen_configs={self.model_gen_configs}\n max_workers={self.max_workers})"

    def __repr__(self):
        return str(self)
