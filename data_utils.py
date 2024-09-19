# data_utils.py

import os
import json
from typing import Dict, List
from torch.utils.data import Dataset

def convert_data_to_messages(data_entry: Dict) -> List[Dict]:
    """
    将数据条目转换为模型所需的消息格式。

    Args:
        data_entry (Dict): 单个数据条目，包含图像路径和对话内容。

    Returns:
        List[Dict]: 格式化后的消息列表。
    """
    messages = []

    # 构建图像内容
    image_paths = data_entry.get("image", [])
    images = [{"type": "image", "image": f"file://{img_path}"} for img_path in image_paths]

    # 构建文本内容
    conversations = data_entry.get("conversations", [])
    for conv in conversations:
        if conv['from'] == 'human':
            # 用户输入的消息
            user_message = {
                "role": "user",
                "content": images + [{"type": "text", "text": conv["value"]}],
            }
            messages.append(user_message)
        elif conv['from'] == 'gpt':
            # 助手生成的响应
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": conv["value"]}],
            }
            messages.append(assistant_message)
    return messages

def load_jsonl_files_from_folder(folder_path: str) -> List[Dict]:
    """
    加载文件夹中的所有 JSONL 文件，并返回合并后的数据。

    Args:
        folder_path (str): JSONL 文件所在的文件夹路径。

    Returns:
        List[Dict]: 合并后的数据列表。
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            with open(os.path.join(folder_path, filename), "r") as file:
                for line in file:
                    data.append(json.loads(line))
    return data

class LazySupervisedDataset(Dataset):
    """用于监督微调和推理的数据集。"""

    def __init__(self, raw_data: List[Dict]):
        super(LazySupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.cached_data = {}

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict:
        if idx in self.cached_data:
            return self.cached_data[idx]

        data_entry = self.raw_data[idx]
        messages = convert_data_to_messages(data_entry)
        self.cached_data[idx] = messages

        return messages

def find_assistant_content_indexes(token_ids: List[int], tokenizer) -> List[tuple]:
    """
    找到助手内容在 token_ids 中的起始和结束索引。

    Args:
        token_ids (List[int]): 输入的 token ID 列表。
        tokenizer: 分词器对象。

    Returns:
        List[tuple]: 包含 (start_index, end_index) 的列表。
    """
    assistant_start_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    start_indexes = []
    end_indexes = []

    for i in range(len(token_ids) - len(assistant_start_ids) + 1):
        if token_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
            start_indexes.append(i)
            for j in range(i + len(assistant_start_ids), len(token_ids)):
                if token_ids[j] == im_end_id:
                    end_indexes.append(j)
                    break
    return list(zip(start_indexes, end_indexes))
