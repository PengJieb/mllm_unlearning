'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2026-03-09 18:52:20
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2026-03-09 19:15:15
FilePath: /mllm_unlearning/gguf_to_transformers.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import gguf
from safetensors.torch import save_file
import os
from safetensors.torch import save_file, load_file
from transformers import AutoProcessor, AutoModelForImageTextToText

def gguf_to_safetensors(gguf_path: str, output_dir: str = "./safetensors_output") -> None:
    """
    将GGUF格式模型转换为Safetensors格式（兼容所有gguf版本）
    
    Args:
        gguf_path: GGUF文件的路径（如 "model.gguf"）
        output_dir: 输出Safetensors文件的目录
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF文件不存在：{gguf_path}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 加载GGUF文件并提取权重（兼容旧版本，不用with语句）
    print(f"正在加载GGUF文件：{gguf_path}")
    reader = gguf.GGUFReader(gguf_path, "r")  # 手动创建Reader
    print(reader)
        # 获取GGUF中的所有张量信息
    tensor_info = reader.tensors
    print(tensor_info[0])
    # 构建state_dict（PyTorch格式的权重字典）
    state_dict = {}
    for reader_tensor in tensor_info:
        # 将GGUF张量（numpy格式）转换为PyTorch张量
        torch_tensor = torch.from_numpy(reader_tensor.data)
        state_dict[reader_tensor.name] = torch_tensor
        print(f"已加载张量：{reader_tensor.name} | 形状：{torch_tensor.shape}")

    
    # 4. 保存为Safetensors格式
    # safetensors_path = os.path.join(output_dir, "model.safetensors")
    # print(f"正在保存Safetensors文件：{safetensors_path}")
    # save_file(state_dict, safetensors_path)
    
    # # 5. 验证保存结果
    # verify_dict = load_file(safetensors_path)
    # print(f"转换完成！共保存 {len(verify_dict)} 个张量，输出路径：{safetensors_path}")
    
# 示例调用
if __name__ == "__main__":
    # 替换为你的GGUF文件路径
    GGUF_FILE_PATH = "./dataset/Qwen3.5-9B-Aggresive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    # 替换为输出目录
    OUTPUT_DIR = "./converted_safetensors"

    gguf_to_safetensors(GGUF_FILE_PATH, OUTPUT_DIR)

    
    # Load model directly


    processor = AutoProcessor.from_pretrained("dataset/Qwen3.5-9B")
    model = AutoModelForImageTextToText.from_pretrained("dataset/Qwen3.5-9B")
    for pn, p in model.named_parameters():
        print(pn, p.shape)
