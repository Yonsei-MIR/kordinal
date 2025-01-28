# kordinal

**kordinal** (Korean Open Resource Dialogue Inference-based Natural Augmentation Library) is a multi-LLM-based library for generating high-quality synthetic Korean datasets.

## Usage

1. Clone this repository and install dependencies:

   ```bash
   git clone https://github.com/Yonsei-MIR/kordinal.git
   cd kordinal
   pip install -e .
   ```

2. Run your preferred LLMs and configure them by editing `scripts/config.yaml`. Example setup for using two LLMs:

```yaml
endpoints:
  mistralai/Mistral-Large-Instruct-2411:
    weight: 3
    selector: "RoundRobinLoadBalancer"
    endpoints:
      - host: <your_host>
        port: <your_port>
        api_key: <your_api_key>
  Qwen/Qwen2.5-72B-Instruct:
    weight: 2
    selector: "RoundRobinLoadBalancer"
    endpoints:
      - host: <your_host>
        port: <your_port>
        api_key: <your_api_key>
gen_config:
  Qwen/Qwen2.5-72B-Instruct:
    max_tokens: 2048
    temperature: 1.1
    top_p: 0.995
    extra_body:
      top_k: -1
      min_p: 0.0
      repetition_penalty: 1.0
      length_penalty: 1.0
      stop:
        - "<|endoftext|>"
        - "[INST]"
        - "[/INST]"
        - "<|im_end|>"
        - "<|end|>"
        - "<|eot_id|>"
        - "<end_of_turn>"
        - "<eos>"
        - "<|END_OF_TURN_TOKEN|>"
        - "<EOS_TOKEN>"
  mistralai/Mistral-Large-Instruct-2411:
    max_tokens: 2048
    temperature: 1.1
    top_p: 0.995
    extra_body:
      top_k: -1
      min_p: 0.0
      repetition_penalty: 1.0
      length_penalty: 1.0
      stop:
        - "<|endoftext|>"
        - "[INST]"
        - "[/INST]"
        - "<|im_end|>"
        - "<|end|>"
        - "<|eot_id|>"
        - "<end_of_turn>"
        - "<eos>"
   ```

3. Run the asynchronous data generation script:

   ```bash
   python gen_new_chat_async.py
   ```

## Important Notes

- This process can generate **high traffic**, so be mindful of resource limitations.
- The generated dataset using this library is available at: [Aka-LLAMA Korean Dataset (Raw)](https://huggingface.co/mirlab/aka-llama-korean-dataset-raw)

## Citation

If you use **kordinal** in your research, please cite the following:

```
@misc{kordinal2024,
  author = {Giyeong Oh, Jaehyun Jeon, Yejin Son, Seungwon Lim, Saejin Kim, Seungho Park, Sumin Shim, Chae-eun Kim, Jihwan Shin, Youngjae Yu},
  title = {kordinal: Multi-LLM-based Synthetic Korean Data Generation Library},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Yonsei-MIR/kordinal}
}
```

