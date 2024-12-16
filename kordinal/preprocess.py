

def replace_from_end(text: str, old: str, new: str, count: int=1) -> str:
    """
    문자열에서 뒤에서부터 특정 패턴을 교체하는 함수.
    
    Parameters:
    - text (str): 입력 문자열
    - old (str): 교체 대상 문자열
    - new (str): 교체할 문자열
    - count (int): 교체할 횟수 (기본값: 1)
    
    Returns:
    - str: 처리된 문자열
    """
    for _ in range(count):
        index = text.rfind(old)  # 뒤에서부터 old의 위치를 찾음
        if index == -1:
            break  # old가 없으면 종료
        text = text[:index] + new + text[index + len(old):]
    return text


def convert_gen_turn_assistant_to_user(text: str, model: str) -> str:
    """
    모델 이름에 따라 텍스트를 처리하는 함수 (match-case 및 효율적 replace 적용).
    
    Parameters:
    - text (str): 입력 텍스트
    - model (str): 모델 이름
    
    Returns:
    - str: 처리된 텍스트
    """
    match model:
        case model if "mistral" in model:
            text += "[INST] " # I hate blank space
        case model if "gemma" in model:
            text = replace_from_end(text, "model", "user", 1)
        case model if "aya" in model:
            text = replace_from_end(text, "CHATBOT", "USER", 1)
        case _:
            text = replace_from_end(text, "assistant", "user", 1)
    return text
