
import random
import uuid

from kordinal.logger_config import logger
from ..client.client_utils import insert_json_schema, is_proprietary


# TODO
# 1. Make constants configurable
class DataEntry:
    """
    DataEntry class
    """
    ASSISTANT = "assistant"
    USER = "user"
    NEXT_ROLE = {ASSISTANT: USER, USER: ASSISTANT}
    AVAILABLE_MODEL = [
        "Qwen/Qwen2.5-72B-Instruct", 
        "CohereForAI/aya-expanse-32b", 
        "mistralai/Mistral-Large-Instruct-2411"
    ]
    # TOKENIZER = {
    #     model: AutoTokenizer.from_pretrained(model) for model in AVAILABLE_MODEL
    # }
    IDENTITY = {
        "대학생": ""
    }
    SYSTEM_PROMPT_FOR_USER_TURN = """당신은 생각하고 질문하는 AI입니다. 주어진 대화를 바탕으로 자연스럽고 일관성 있는 다음 User의 발화를 **한국어로** 생성하세요.

---

### 작성 지침

1. **대화 흐름 유지**
   - 이전 대화와 자연스럽게 연결되도록 작성하세요.
   - 필요한 경우 간단한 요약이나 배경 정보를 포함하세요.

2. **깊이 있는 탐구**
   - 단순 정보 확인 대신, 주제를 확장하거나 새로운 관점을 탐구할 수 있는 질문을 작성하세요.

3. **비판적 사고**
   - 앞선 답변의 모순이나 흥미로운 점을 지적하며 추가 질문을 생성하세요.

4. **새로운 방향**
   - 때때로 관련된 새로운 관점이나 이슈를 탐구하는 질문을 제안하세요.

5. **명확성과 독립성**
   - 질문만 보아도 의도를 이해할 수 있도록 구체적이고 명확하게 작성하세요.

---

### 자연스러운 대화 요령

- 일상적인 한국어 표현을 사용하세요.
- 구체적인 사례나 예시를 활용해 질문을 명확히 전달하세요.
""".strip()
    
    SYSTEM_PROMPT_FOR_USER_TURN_COMPLETION = """
당신은 생각하고 질문하는 사람입니다. 아래 정보를 바탕으로 일관성 있는 User 발화를 **한국어**로 생성하세요.

### 태스크
{task_description}

### 질문 생성 요령
- 이전 대화의 흐름을 유지하고, 앞선 발화에 자연스럽게 연결되는 질문을 하세요.
- 단순한 정보 확인보다는 깊이 있는 탐구를 유도하고, 추가적인 대화를 이끌어낼 수 있는 질문을 하세요.
- 비판적인 질문을 생성하세요. 이전 답변에 오류나 모순이 있는 경우, 이를 지적하거나 의문을 제기하세요.
- 다양한 관점을 제시하세요. 가능하다면, 여러 가지 관점이나 접근 방식을 고려하여 질문을 생성하세요.
- 때때로 난이도를 조정하세요. 너무 쉽거나 어려운 질문만 반복하지 않도록, 난이도를 적절히 조절하세요.

### 자연스러운 대화
- 자연스럽고 일상적인 한국인의 말투를 사용하세요.
- 추상적인 질문보다는 구체적인 예시를 사용하여 질문을 명확하게 전달하세요.""".strip()

    TASK_DESCRIPTIONS = {
        "text_extraction": "특정 정보나 문장을 추출하는 질문 작성",
        "creative_content": "창의적이고 독창적인 질문 작성",
        "analytical_reasoning": "논리적으로 분석하고 새로운 관점을 도출하는 질문 작성",
        "brain_teaser": "퍼즐이나 직관적 사고를 유도하는 질문 작성",
        "text_classification": "정보를 카테고리화하거나 분류하는 질문 작성",
        "rag": "추가 정보를 검색하고 활용하는 질문 작성",
        "fermi": "대략적인 추정을 요구하는 질문 작성",
        "mcq": "새로운 객관식 문제 작성",
        "fs_cot_flow": "사고를 단계적으로 전개하는 질문 작성",
        "code_": "코드 작성이나 코드와 관련된 질문 작성",
        "text_modification": "텍스트를 수정하거나 요약하는 질문 작성",
        "struct2text_flow": "구조화된 정보를 텍스트로 표현하는 질문 작성",
        "follow_up": "주제를 확장하거나 후속 질문을 생성하는 질문 작성",
        "open_domain_qa": "다양한 주제를 탐구하는 질문 작성",
        "rc": "특정 텍스트를 읽고 의미를 파악하는 질문 작성",
        "general": "일반적인 질문 작성"
    }

    JSON_SCHEMA_TASK_SPECIFIC = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "질문을 작성하기 전에 주제를 확장할 방법을 서술. 앞서 선택한 태스크를 수행하기 적합한 질문을 작성."
            },
            "generated_question": {
                "type": "string",
                "description": "구체적이고 호기심을 유발하며, 그 자체로 독립적인 질문"
            }
        },
        "required": ["thought", "generated_question"],
        "additionalProperties": False
    }

    JSON_SCHEMA_USER_TURN_VLLM = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "your_turn": {
                "type": "string",
                "enum": ["User"],
                "description": "주어진 대화를 보고, 자연스러운 User의 답변을 생성"
            },
            "next_question": {
                "type": "string",
                "description": "User의 입장에서 대화의 맥락에 맞는 질문"
            }
        },
        "required": ["your_turn", "next_question"],
        "additionalProperties": False
    }
    
    JSON_SCHEMA_USER_TURN_OPENAI = {
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "your_turn": {
                    "type": "string",
                    "enum": ["User"],
                    "description": "주어진 대화를 보고, 자연스러운 User의 답변을 생성"
                },
                "next_question": {
                    "type": "string",
                    "description": "User의 입장에서 대화의 맥락에 맞는 질문"
                }
            },
            "required": ["your_turn", "next_question"],
            "additionalProperties": False
        }
    }
    
    def __init__(self, data: dict, gen_model: str, life: int|tuple=None):
        if "data" in data:
            if isinstance(data["data"], dict):
                unfold_data = data.pop("data")
                for k, v in data.items():
                    unfold_data[k] = v 
            else:
                unfold_data = data
            self.raw_data: dict = unfold_data   
        else:
            self.raw_data: dict = data

        self.request_id = str(uuid.uuid4())
        if "prompt_id" in self.raw_data.get("metadata", {}): 
            self.prompt_id: str = self.raw_data.get("metadata", {}).get("prompt_id", None)
        else:
            self.prompt_id: str = str(uuid.uuid4())

        self.task = data.get("synthetic_question", {}).get(gen_model, {}).get("task", "general")

        self.messages: list[dict[str, str]] = data.get('messages', [])
        message_with_source = []
        for message in self.messages:
            if "source" in message:
                message_with_source.append(message)
            else:
                message_with_source.append({"source": "original", **message})
        self.messages = message_with_source

        self.priority: int = 0
        if life is None:
            self.life = random.randint(3, 8) + 3  # Life of the data entry
        elif isinstance(life, tuple):
            min_life, max_life = life
            self.life = random.randint(min_life, max_life)
        else:
            self.life = life
        self.life += int(self.life % 2 == 1)  # Make it even -> must end with assistant's turn
        self.current_model: str = None  # Starting model
        # print(self.messages)
        # print(self.task)
        # print(self.raw_data)
        # input("")

        logger.info(f"Request ID: {self.request_id}, Life: {self.life} appended.")
    
    # for priority queue
    def __eq__(self, other: "DataEntry"): return self.request_id == other.request_id
    def __lt__(self, other: "DataEntry"): return self.priority > other.priority
    def __le__(self, other: "DataEntry"): return self.priority >= other.priority
    def __gt__(self, other: "DataEntry"): return self.priority < other.priority
    def __ge__(self, other: "DataEntry"): return self.priority <= other.priority

    def __str__(self):
        return f"DataEntry(request_id={self.request_id}, messages={self.messages}, priority={self.priority}, life={self.life}, current_model={self.current_model})"

    def __repr__(self):
        return str(self)
    
    @classmethod
    def extend_stop_signal(cls, model_gen_config):
        new_model_gen_config = model_gen_config.copy()
        new_model_gen_config["extra_body"]["stop"].extend(
            ["**Assistant:**", "**assistant:**", "Assistant:", "assistant:", "---"]
        )
        return new_model_gen_config

    @classmethod
    def gen_test_message(cls):
        test = [
            {"role": "user", "content": "Scenario: You are a mobile app developer working on a new social networking application. The app includes a feature where users can post updates that include text, images, and their current location. To enhance user experience, you want to implement a custom toolbar at the bottom of the post creation screen that includes quick access buttons for adding images and location tags to the post.\n\nTask: Using the provided Onsen UI CSS framework, design and implement a responsive bottom toolbar for the post creation screen of the social networking app. The toolbar should include two buttons: one for adding images (with an appropriate icon) and another for adding a location tag (with an appropriate icon). Ensure that the toolbar is fixed at the bottom of the screen and is visible when the keyboard is active for text input. Additionally, the toolbar should adapt to different screen sizes and orientations, maintaining its functionality and aesthetic appeal across devices."}, 
            {"role": "assistant", "content": "To implement a responsive bottom toolbar using the Onsen UI CSS framework, you'll need to use Onsen UI's components and CSS classes. Here's a step-by-step guide to creating the toolbar:\n\n1. Include Onsen UI CSS and JS in your project. If you haven't already, add the following lines to the `<head>` section of your HTML file:\n\n```html\n<link rel=\"stylesheet\" href=\"https://unpkg.com/onsenui/css/onsenui.css\">\n<link rel=\"stylesheet\" href=\"https://unpkg.com/onsenui/css/onsen-css-components.min.css\">\n<script src=\"https://unpkg.com/onsenui/js/onsenui.min.js\"></script>\n```\n\n2. Create the toolbar at the bottom of the post creation screen. You can use the `<ons-toolbar>` component and place it inside an `<ons-bottom-toolbar>` to fix it at the bottom of the screen.\n\n3. Add buttons for adding images and location tags. Use `<ons-button>` components with icons. You can use Onsen UI's built-in icons or any other icon library you prefer, such as Font Awesome.\n\n4. Ensure that the toolbar remains visible when the keyboard is active. This typically involves adjusting the viewport settings and possibly using JavaScript to handle focus events.\n\n5. Make sure the toolbar is responsive and adapts to different screen sizes and orientations. Onsen UI components are designed to be responsive, but you may need to add additional CSS media queries if you want to customize the layout further.\n\nHere's an example of how the HTML structure with Onsen UI components might look:\n\n```html\n<ons-page>\n <!-- Other content of the post creation screen goes here -->\n\n <!-- Bottom toolbar -->\n <ons-bottom-toolbar>\n <div class=\"toolbar__center\">\n <ons-row>\n <ons-col width=\"50%\">\n <ons-button modifier=\"quiet\" class=\"button--large\">\n <ons-icon icon=\"md-image\" class=\"ons-icon\"></ons-icon>\n Add Image\n </ons-button>\n </ons-col>\n <ons-col width=\"50%\">\n <ons-button modifier=\"quiet\" class=\"button--large\">\n <ons-icon icon=\"md-pin-drop\" class=\"ons-icon\"></ons-icon>\n Add Location\n </ons-button>\n </ons-col>\n </ons-row>\n </div>\n </ons-bottom-toolbar>\n</ons-page>\n```\n\nAnd here's some additional CSS to ensure the toolbar looks good on all devices:\n\n```css\n/* Ensure the toolbar is always at the bottom and visible */\nons-bottom-toolbar {\n position: fixed;\n bottom: 0;\n left: 0;\n right: 0;\n z-index: 999;\n}\n\n/* Style for large buttons */\n.button--large ons-icon {\n font-size: 24px;\n vertical-align: middle;\n margin-right: 8px;\n}\n\n/* Responsive adjustments */\n@media (max-width: 600px) {\n .button--large {\n padding: 10px;\n font-size: 16px;\n }\n}\n```\n\nRemember to test the toolbar on various devices and screen sizes to ensure it works as expected. You may need to adjust the CSS and possibly use JavaScript to handle specific cases, such as when the keyboard is displayed on mobile devices."}]
        test = test * 3
        idx = random.randint(1, len(test))
        return test[:1]

    @property
    def last_role(self):
        return self.messages[-1]["role"]
    
    def append(self, text, source):
        role = self.NEXT_ROLE[self.last_role]
        self.messages.append({"source": source, "role": role, "content": text})

    def is_alive(self):
        return self.life > 0

    def consume(self):
        self.life -= 1
        self.request_id = str(uuid.uuid4())
        
    def conduct_save(self):
        data = self.raw_data.copy()
        data["messages"] = self.messages
        return data
    
    def get_next_turn_data(self, model: str, messages: list[dict], model_gen_config: dict) -> str:
        messages_without_source = [
            {k: v for k, v in message.items() if k != "source"} for message in messages
        ]
        print("last role==========================================", self.last_role)
        if self.last_role == self.ASSISTANT:
            if messages_without_source[0]["role"] == "system":
                original_system_message = messages_without_source[0]["content"]
            else:
                original_system_message = ""
            
            messages_without_source.insert(
                0, {"role": "system",
                    "content": (original_system_message + "\n\n" + self.SYSTEM_PROMPT_FOR_USER_TURN).strip()}
            )
        else:
            messages_without_source.insert(
                0, {"role": "system",
                    "content": "당신은 친절한 어시스턴트입니다. **한국어로** 대답하세요."} # 생각하고 답하는 AI?
            )

        data = {
            "model": model,
            "messages": messages_without_source,
            **model_gen_config,
        }
        # data["extra_body"].pop("request_id")

        if self.last_role == self.ASSISTANT:
            is_vllm = not is_proprietary(model)
            if is_proprietary(model):
                json_schema = self.JSON_SCHEMA_USER_TURN_OPENAI.copy()
            else:
                json_schema = self.JSON_SCHEMA_USER_TURN_VLLM.copy()
            data = insert_json_schema(data, json_schema, vllm=is_vllm)
        else: # if assistant turn, use auto-generated completion
            data.pop("response_format", None)
            extra_body = data.get("extra_body", {})
            extra_body.pop("guided_json", None)
            data["extra_body"] = extra_body

        if not is_proprietary(model):
            data["extra_body"]["priority"] = self.priority

        return data
