# finetune.py

import os
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info  # 确保你有正确的 qwen_vl_utils 模块
from typing import Dict, List
from data_utils import (
    convert_data_to_messages,
    load_jsonl_files_from_folder,
    find_assistant_content_indexes,
    LazySupervisedDataset
)

# 设置常量
IGNORE_TOKEN_ID = -100

# 禁用 W&B 日志记录
os.environ["WANDB_MODE"] = "disabled"

# 定义模型和数据路径
MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-7B-Instruct"
TRAIN_DATA_PATH = "path/to/your/training/data"
EVAL_DATA_PATH = "path/to/your/evaluation/data"
OUTPUT_DIR = "path/to/save/your/model"

# 加载处理器和分词器
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28

processor = AutoProcessor.from_pretrained(
    MODEL_NAME_OR_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    padding_side="right"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    准备训练时的输入和标签。

    Args:
        batch (List[Dict]): 输入的批次数据。

    Returns:
        Dict[str, torch.Tensor]: 包含输入 IDs、标签和注意力掩码的字典。
    """
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch
    ]

    image_inputs, video_inputs = process_vision_info(batch)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=512,  # 根据需要调整 max_length
        return_tensors="pt",
    )

    input_ids_list = inputs['input_ids'].tolist()
    assert len(batch) == len(input_ids_list)

    labels_list = []
    for ids_list in input_ids_list:
        label_ids = [IGNORE_TOKEN_ID] * len(ids_list)
        for start_idx, end_idx in find_assistant_content_indexes(ids_list, tokenizer):
            # 确保索引不超过序列长度
            start = start_idx + 2
            end = end_idx + 1
            start = min(start, len(ids_list))
            end = min(end, len(ids_list))
            label_ids[start:end] = ids_list[start:end]
        labels_list.append(label_ids)

    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    return {
        'input_ids': inputs['input_ids'],
        'labels': labels_tensor,
        'attention_mask': inputs['attention_mask']
    }

def train():
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16,  # 根据需要调整精度
        config={"use_cache": False},
        trust_remote_code=True
    )

    # 冻结视觉模型的参数
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
        model.transformer.visual.requires_grad_(False)
        if hasattr(model.transformer.visual, 'attn_pool'):
            model.transformer.visual.attn_pool.requires_grad_(True)

    # 加载数据
    train_data = load_jsonl_files_from_folder(TRAIN_DATA_PATH)
    train_dataset = LazySupervisedDataset(train_data)

    eval_data = load_jsonl_files_from_folder(EVAL_DATA_PATH)
    eval_dataset = LazySupervisedDataset(eval_data)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        bf16=True,  # 如果支持，则启用混合精度
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],  # 禁用报告到任何日志记录工具
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    # 开始训练
    trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    train()
