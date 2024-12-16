import argparse
from datasets import load_from_disk
import langid
import re

def percentage_korean(text):
    korean_chars = re.findall(r'[\uac00-\ud7a3]', text)
    return len(korean_chars) / max(len(text), 1)

def percentage_chinese(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) / max(len(text), 1)

def percentage_japanese(text):
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text)
    return len(japanese_chars) / max(len(text), 1)


def has_enough_korean(text, threshold=0.3):
    return percentage_korean(text) >= threshold

def has_enough_chinese(text, threshold=0.3):
    return percentage_chinese(text) >= threshold

def has_enough_japanese(text, threshold=0.3):
    return percentage_japanese(text) >= threshold


def has_high_local_density(lang, text, window_size=40, density_threshold=0.6):
    if lang == 'ko':
        return has_high_korean_local_density(text, window_size, density_threshold)
    elif lang == 'zh':
        return has_high_chinese_local_density(text, window_size, density_threshold)
    else:
        return False

def has_high_korean_local_density(text, window_size=40, density_threshold=0.6):
    for i in range(len(text) - window_size + 1):
        window = text[i:i + window_size]
        if percentage_korean(window) >= density_threshold:
            return True
    return False

def has_high_chinese_local_density(text, window_size=40, density_threshold=0.6):
    for i in range(len(text) - window_size + 1):
        window = text[i:i + window_size]
        if percentage_chinese(window) >= density_threshold:
            return True
    return False

def has_high_japanese_local_density(text, window_size=40, density_threshold=0.6):
    for i in range(len(text) - window_size + 1):
        window = text[i:i + window_size]
        if percentage_japanese(window) >= density_threshold:
            return True
    return False



def is_korean(text):
    lang, _ = langid.classify(text)
    return lang == 'ko'

def is_chinese(text):
    lang, _ = langid.classify(text)
    return lang == 'zh'

def is_japanese(text):
    lang, _ = langid.classify(text)
    return lang == 'ja'


def check_anomaly(example):
    # unfold_data 로직
    # synthetic_question에서 model, question, thought, task 추출
    # anomaly 판단 후 flag와 reason 컬럼 반환
    if "data" in example:
        unfold_data = example["data"].copy()
        for k, v in example.items():
            if k != "data":
                unfold_data[k] = v
    else:
        unfold_data = example

    # model 찾기
    model = None
    for m in unfold_data.get("synthetic_question", {}):
        if unfold_data["synthetic_question"][m] is not None:
            model = m
            break

    if model is None:
        # anomaly
        reason = "model is None"
        return {
            **unfold_data,
            "flag": reason
        }

    synthetic_question = unfold_data["synthetic_question"][model]
    if synthetic_question is None:
        reason = "synthetic_question is None"
        return {
            **unfold_data,
            "flag": reason
        }

    question = synthetic_question["generated_question"]
    task = synthetic_question["task"]
    thought = synthetic_question["thought"]

    detected = ( has_enough_korean(question), is_korean(question), has_enough_korean(thought), is_korean(thought) )
    # anomaly 조건:
    # sum(detected) < (len(detected)-1) and (not detected[1])
    # 즉, 대부분 한국어 비율이 낮거나 question이 한국어가 아닌 경우
    if not all(detected):
        reason = f"Not enough Korean or not Korean. detected={detected}, task={task}"
        return {
            **unfold_data,
            "flag": reason
        }

    # anomaly 아님
    return {
        **unfold_data,
        "flag": ""
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with anomaly detection")
    parser.add_argument("--input_dir", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = input_dir + "_filtered"

    dataset = load_from_disk(input_dir)

    # map에 num_proc을 사용하면 병렬처리가 가능
    # tqdm 표시를 위해서는 dataset.map의 'desc' 인자를 활용할 수 있다.
    processed = dataset.map(
        check_anomaly,
        num_proc=args.num_proc,
        desc="Processing dataset"
    )

    processed.save_to_disk(output_dir)

    print(f"Processing complete. Saved to {output_dir}")
