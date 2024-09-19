# inference.py

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info  # 确保你有正确的 qwen_vl_utils 模块
from typing import Dict, List

# 导入共享的函数和类
from data_utils import (
    convert_data_to_messages,
    load_jsonl_files_from_folder,
    LazySupervisedDataset
)

# 定义模型和数据路径
MODEL_PATH = "path/to/your/model"
TEST_DATA_PATH = "path/to/your/test/data"

# 加载模型和处理器
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    padding_side="right"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id

model.eval()

def collate_fn(batch: List[Dict]) -> tuple:
    """
    准备推理时的输入。

    Args:
        batch (List[Dict]): 输入的批次数据。

    Returns:
        tuple: 包含模型输入和原始批次数据的元组。
    """
    # 准备文本输入
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch
    ]

    # 处理图像和视频
    image_inputs, video_inputs = process_vision_info(batch)

    # 将输入转换为模型可接受的格式
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return inputs, batch  # 返回原始批次数据以供参考

def main(test_data_path: str, model_path: str):
    # 加载数据
    raw_data = load_jsonl_files_from_folder(test_data_path)
    eval_dataset = LazySupervisedDataset(raw_data)

    # 创建 DataLoader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 生成输出
    for inputs, batch in eval_dataloader:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成输出
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=512,
                do_sample=False,
                num_beams=1,
            )

        # 解码输出
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 打印生成的输出
        for idx, generated_text in enumerate(generated_texts):
            print(f"示例 {idx} 的生成输出:")
            print(f"{generated_text}")
            print("-----")

if __name__ == "__main__":
    main(TEST_DATA_PATH, MODEL_PATH)
