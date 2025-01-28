import uuid
import json
from .constants import PRICING


def is_proprietary(model):
    if model is not None:
        return model in PRICING
    return False


def build_base_url(host, port=None):
    assert host is not None, "host is required"
    if port is None:
        return f"http://{host}"
    else:
        return f"http://{host}:{port}"


def convert_data_from_client_to_request(data: dict):
    extra_body = data.setdefault('extra_body', {})
    json_data = {**data, **extra_body}
    if 'extra_body' in json_data:
        del json_data["extra_body"]
    return json_data


def insert_json_schema(data: dict, json_schema: dict, vllm=False) -> dict:
    is_vllm_schema = "$schema" in json_schema
    new_data = data.copy()
    
    if vllm:
        if is_vllm_schema: # vllm schema and vllm target
            target_schema = json_schema.copy()
        else: # not vllm schema and vllm target
            assert "schema" in json_schema, "schema key is required in json_schema"
            target_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                **json_schema["schema"]
            }
        new_data["extra_body"]["guided_json"] = target_schema
    else:
        if is_vllm_schema: # vllm schema and not vllm target
            target_schema = {k: v for k, v in json_schema.items() if k != "$schema"}
        else:
            target_schema = json_schema.copy()
        new_data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": target_schema
            }
        }
    return new_data 

def get_text_from_response(response):
    if not isinstance(response, dict):
        response = response.dict()
    
    choice = response["choices"][0]
    if "text" in choice:
        return choice["text"].strip()
    elif "message" in choice:
        content = choice["message"]["content"]
        try:
            parsed_content = json.loads(content)
            if "next_question" in parsed_content:
                return parsed_content["next_question"].strip()
            else:
                return parsed_content.strip() if isinstance(parsed_content, str) else json.dumps(parsed_content)
        except json.JSONDecodeError:
            return content.strip()
    else:
        return None

def predict_price(response):
    if not isinstance(response, dict):
        data = response.dict()
    else:
        data = response
    
    model = data.get("model", None)
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    recipe = PRICING.get(model, None)
    payment = 0
    unit = recipe.get("unit", 1)
    currency = PRICING.get(model, {}).get("currency", "$")

    # TODO: Implement the detailed pricing logic

    price_info = recipe.get("default", {})
    payment = (price_info.get("prompt", 0) * prompt_tokens + price_info.get("completion", 0) * completion_tokens) / float(unit)
    return payment, currency
        

if __name__ == "__main__":
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
    json_schema_vllm = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "enum": tasks_en,
            },
            "generated_question": {
                "type": "string",
                "description": "구체적이고 호기심을 유발하며, 주제를 확장할 수 있으며 그 자체로 독립적인 질문"
            }
        },
        "required": ["task", "generated_question"],
        "additionalProperties": False
    }
    json_schema = {
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": tasks_en,
                },
                "generated_question": {
                    "type": "string",
                    "description": "구체적이고 호기심을 유발하며, 주제를 확장할 수 있으며 그 자체로 독립적인 질문"
                }
            },
            "required": ["task", "generated_question"],
            "additionalProperties": False
        }
    }

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 1.1,
        "top_p": 0.995,
        # "response_format": {
        #     "type": "json_schema",
        #     "json_schema": json_schema
        # },
        "extra_body": {
            "top_k": -1,
            "min_p": 0.15,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            # "priority": -1,
            # "guided_json": json_schema_vllm,
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
    
    print("==="*10)
    converted_data = insert_json_schema(data.copy(), json_schema, vllm=True)
    converted_data.pop("messages")
    print("response format -> guided json", converted_data)
    print("==="*10)
    converted_data = insert_json_schema(data.copy(), json_schema, vllm=False)
    converted_data.pop("messages")
    print("response format -> response format", converted_data)
    print("==="*10)
    converted_data = insert_json_schema(data.copy(), json_schema_vllm, vllm=True)
    converted_data.pop("messages")
    print("guided json -> guided json", converted_data)
    print("==="*10)
    converted_data = insert_json_schema(data.copy(), json_schema, vllm=False)
    converted_data.pop("messages")
    print("guided json -> response format", converted_data)
    exit(-1)


