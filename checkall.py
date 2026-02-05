# F:\ai\ComfyUI-aki-v1.3\ComfyUI-aki-v1.3\python
# pip install tkinterdnd2   
import os
import sys
import gguf
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from collections import Counter

import numpy as np
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from llama_cpp import Llama

hang=25
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# GGUF 权重类型映射表
GGUF_WEIGHT_TYPES = {
    0:  "float32",
    1:  "float16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0", 
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
}

def format_param_count_practical(param_count):
    """实用版本，根据规模智能选择显示方式"""
    if param_count == 0:
        return "0"
    
    billions = param_count / 1_000_000_000
    
    if billions >= 1:  # 1B以上
        if billions >= 1000:  # 1000B以上用T
            trillions = billions / 1000
            return f"{trillions:.1f}T" if trillions < 10 else f"{int(trillions)}T"
        elif billions >= 100:  # 100B以上显示整数
            return f"{int(billions)}B"
        elif billions >= 10:  # 10-100B显示一位小数
            return f"{billions:.1f}B"
        else:  # 1-10B显示两位小数
            return f"{billions:.2f}B"
    else:  # 1B以下
        millions = param_count / 1_000_000
        if param_count >= 1_000_000:  # 1M以上用M
            return f"{millions:.1f}M" if millions < 100 else f"{int(millions)}M"
        elif param_count >= 1_000:  # 1K以上用K
            thousands = param_count / 1_000
            return f"{int(thousands)}K"
        else:
            return f"{param_count}"

def calculate_memory_requirement(P, Q):
    """
    根据图片公式计算显存需求: M = (P × Q) / 8 × 1.2
    
    Args:
        P: 模型参数量 (单位: 亿)
        Q: 参数位宽 (FP16=16, INT8=8, INT4=4)
    
    Returns:
        显存需求 (单位: GB)
    """
    # P 需要转换为以亿为单位的数值
    P_billions = P / 1_000_000_000  # 转换为B单位
    M = (P_billions * Q) / 8 * 1.2
    return M

def get_quantization_bits(dtype):
    """根据数据类型获取参数位宽Q值"""
    if dtype == torch.float32:
        return 32
    elif dtype == torch.float16:
        return 16
    elif dtype == torch.bfloat16:
        return 16  # bfloat16也是16位
    elif dtype == torch.float8_e5m2:
        return 8
    elif dtype == torch.float8_e4m3fn:
        return 8
    elif 'int8' in str(dtype).lower() or 'q8' in str(dtype).lower():
        return 8
    elif 'int6' in str(dtype).lower() or 'q6' in str(dtype).lower():
        return 6
    elif 'int5' in str(dtype).lower() or 'q5' in str(dtype).lower():
        return 5
    elif 'int4' in str(dtype).lower() or 'q4' in str(dtype).lower():
        return 4
    elif 'int3' in str(dtype).lower() or 'q3' in str(dtype).lower():
        return 3
    elif 'int2' in str(dtype).lower() or 'q2' in str(dtype).lower():
        return 2
    else:
        return 16  # 默认按FP16处理

def classify_model_size(total_params):
    """根据总参数量对模型进行分类"""
    if total_params >= 100_000_000_000:  # 100B+
        return "超大规模模型 (>100B参数)"
    elif total_params >= 50_000_000_000:  # 50B+
        return "超大模型 (50B-100B参数)"
    elif total_params >= 10_000_000_000:  # 10B+
        return "大模型 (10B-50B参数)"
    elif total_params >= 1_000_000_000:   # 1B+
        return "中等模型 (1B-10B参数)"
    elif total_params >= 100_000_000:     # 100M+
        return "小型模型 (100M-1B参数)"
    else:
        return "微型模型 (<100M参数)"

def inspect_safetensors(filepath):
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 加载safetensors文件（带元数据）
        with open(filepath, "rb") as f:
            from safetensors import safe_open
            tensors = load_file(filepath)
            # 打开文件以读取元数据
            metadata = {}
            try:
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
            except:
                # 如果无法读取元数据，尝试其他方法
                pass
        
        print(f"✅ 成功加载文件，包含 {len(tensors)} 个张量")
        
        # 统计dtype和参数量
        dtype_param_count = {}
        total_params = 0
        
        # 统计tensor名称
        tensor_names = []
        tensor_prefixes = []  # 存储每个tensor名称的第一个部分（第一个点之前的部分）
        tensor_second_prefixes = []  # 存储每个tensor名称的第二部分（第一个点和第二个点之间的部分）
        tensor_third_prefixes = []  # 存储每个tensor名称的第三部分（第二个点和第三个点之间的部分）
        
        for name, tensor in tensors.items():
            dtype = tensor.dtype
            param_count = tensor.numel()
            dtype_param_count[dtype] = dtype_param_count.get(dtype, 0) + param_count
            total_params += param_count
            tensor_names.append(name)
            
            # 提取名称中的第一个部分（第一个点之前的部分）
            prefix = name.split('.')[0] if '.' in name else name
            tensor_prefixes.append(prefix)
            
            # 提取名称中的第二个部分（第一个点之后、第二个点之前的部分）
            parts = name.split('.')
            if len(parts) >= 2:
                second_prefix = parts[1]  # 第二个部分
            else:
                second_prefix = ""  # 如果没有第二个部分，则为空字符串
            tensor_second_prefixes.append(second_prefix)
            
            # 提取名称中的第三个部分（第二个点之后、第三个点之前的部分）
            if len(parts) >= 3:
                third_prefix = parts[2]  # 第三个部分
            else:
                third_prefix = ""  # 如果没有第三个部分，则为空字符串
            tensor_third_prefixes.append(third_prefix)
        
        # 计算唯一tensor名称的数量
        unique_names = list(dict.fromkeys(tensor_names))  # 保持首次出现的顺序
        duplicate_names = [name for name in tensor_names if tensor_names.count(name) > 1]
        unique_name_count = len(unique_names)
        
        # 计算唯一前缀的数量
        unique_prefixes = list(dict.fromkeys(tensor_prefixes))  # 保持首次出现的顺序
        prefix_counts = Counter(tensor_prefixes)  # 计算每个前缀出现的次数
        unique_prefix_count = len(unique_prefixes)
        
        # 计算唯一第二前缀的数量
        unique_second_prefixes = [p for p in dict.fromkeys(tensor_second_prefixes) if p]  # 保持首次出现的顺序，排除空字符串
        second_prefix_counts = Counter([p for p in tensor_second_prefixes if p])  # 计算每个第二前缀出现的次数
        unique_second_prefix_count = len(unique_second_prefixes)
        
        # 计算唯一第三前缀的数量
        unique_third_prefixes = [p for p in dict.fromkeys(tensor_third_prefixes) if p]  # 保持首次出现的顺序，排除空字符串
        third_prefix_counts = Counter([p for p in tensor_third_prefixes if p])  # 计算每个第三前缀出现的次数
        unique_third_prefix_count = len(unique_third_prefixes)
        
        # 计算主要精度的位宽
        if dtype_param_count:
            main_dtype = max(dtype_param_count, key=dtype_param_count.get)
            main_percentage = (dtype_param_count[main_dtype] / total_params) * 100
            Q_value = get_quantization_bits(main_dtype)
            
            # 使用主要精度计算显存需求
            memory_gb = calculate_memory_requirement(total_params, Q_value)
        else:
            memory_gb = 0
            Q_value = 16  # 默认值
        
        # 生成报告
        report = f"📄 文件: {os.path.basename(filepath)}\n"


        report += f"\n{'─' * hang}\n"  # 分隔线
        # 添加元数据信息
        if metadata:
            report += f"📚 元数据信息:\n"
            for key, value in list(metadata.items())[:10]:  # 显示前10个元数据项
                report += f"   {key}: {value}\n"
            if len(metadata) > 10:
                report += f"   ... 还有 {len(metadata) - 10} 个元数据项\n\n"
            else:
                report += "\n"
        else:
            report += f"📚 元数据: 无\n\n"

        report += f"\n{'─' * hang}\n"  # 分隔线

        # 显示前几个唯一的tensor名称
        if unique_names:
            report += f"🏷️ 前10个唯一张量名称:\n"
            for i, name in enumerate(unique_names[:10]):
                report += f"   {i+1}. {name}\n"
            if len(unique_names) > 10:
                report += f"   ... 还有 {len(unique_names) - 10} 个名称\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的前缀（第一次出现的前缀）
        if unique_prefixes:
            report += f"🏷️ 第一前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_prefixes[:10]):  # 显示前10个不同的前缀
                count = prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_prefixes) > 10:
                report += f"   ... 还有 {len(unique_prefixes) - 10} 个前缀\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的第二前缀（第一次出现的第二前缀）
        if unique_second_prefixes:
            report += f"🏷️ 第二前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_second_prefixes[:10]):  # 显示前10个不同的第二前缀
                count = second_prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_second_prefixes) > 10:
                report += f"   ... 还有 {len(unique_second_prefixes) - 10} 个第二前缀\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的第三前缀（第一次出现的第三前缀）
        if unique_third_prefixes:
            report += f"🏷️ 第三前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_third_prefixes[:10]):  # 显示前10个不同的第三前缀
                count = third_prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_third_prefixes) > 10:
                report += f"   ... 还有 {len(unique_third_prefixes) - 10} 个第三前缀\n\n"
            else:
                report += "\n"



        
        report += f"{'─' * hang}\n"  # 分隔线
        report += f"📊 总参数量: {total_params:,} ({format_param_count_practical(total_params)})\n"
        report += f"📈 张量数量: {len(tensors)} (唯一名称: {unique_name_count}, 重复名称: {len(duplicate_names)})\n"
        report += f"🏷️ 前缀统计: {unique_prefix_count} 个不同第一前缀, {unique_second_prefix_count} 个不同第二前缀, {unique_third_prefix_count} 个不同第三前缀\n"
        report += f"💾 显存估算: {memory_gb:.1f} GB (基于公式: M = (P × Q) / 8 × 1.2)\n"
        report += f"   - P = {total_params / 1_000_000_000:.1f}B (参数量)\n"
        report += f"   - Q = {Q_value} (主要精度: {main_dtype})\n\n"
        

        # 按参数量排序显示
        sorted_dtypes = sorted(dtype_param_count.items(), key=lambda x: x[1], reverse=True)
        
        for dtype, param_count in sorted_dtypes:
            percentage = (param_count / total_params) * 100
            formatted_count = format_param_count_practical(param_count)
            q_bits = get_quantization_bits(dtype)
            report += f"🔹 {dtype}: {param_count:,} 参数 ({formatted_count}, {percentage:.2f}%, Q={q_bits})\n"
        
        # 判断精度类型
        dtypes = list(dtype_param_count.keys())
        if all(dtype == torch.float32 for dtype in dtypes):
            report += "\n✅ 模型为纯 FP32（float32）"
        elif all(dtype == torch.float16 for dtype in dtypes):
            report += "\n✅ 模型为纯 FP16（float16）"
        elif all(dtype == torch.bfloat16 for dtype in dtypes):
            report += "\n✅ 模型为纯 BF16（bfloat16）"
        else:
            report += f"\n⚠️ 模型为混合精度（主要精度: {main_dtype}, 占比: {main_percentage:.1f}%）"

        # 显示不同精度的显存需求对比
        report += f"\n\n🔍 不同精度显存需求对比:"
        for bits, precision_name in [(32, "FP32"), (16, "FP16/BF16"), (8, "INT8"), (4, "INT4")]:
            mem_req = calculate_memory_requirement(total_params, bits)
            report += f"\n   {precision_name}: {mem_req:.1f} GB"



        report += f"\n{'─' * hang}\n"  # 分隔线


        # 保存分析结果到 .checkinfo 文件
        checkinfo_filename = filepath.rsplit('.', 1)[0] + '.checkinfo'
        try:
            with open(checkinfo_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)
            print(f"✅ 分析结果已保存到: {checkinfo_filename}")
        except Exception as e:
            print(report)
            print(f"⚠️  保存 .checkinfo 文件时出错: {str(e)}")
        
        return dtype_param_count, total_params, memory_gb
        
    except Exception as e:
        error_msg = f"❌ 无法读取模型文件 {filepath}:\n{str(e)}"
        print(error_msg)
        return {}, 0, 0

# 专门用于GGUF文件的显存估算（处理量化类型）
def inspect_gguf(path):
    """检查GGUF文件并计算显存需求"""
    try:
        import gguf
        import numpy as np
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        reader = gguf.GGUFReader(path)
        print(f"✅ 成功加载GGUF文件: {os.path.basename(path)}")
        
        # 读取GGUF文件的元数据
        metadata = {}
        for field_name, field_value in reader.fields.items():
            try:
                # 获取字段值
                if hasattr(field_value, 'tolist'):
                    # 如果是numpy数组，转换为列表或标量
                    value = field_value.tolist()
                elif hasattr(field_value, 'value'):
                    # 如果是GGUF特定类型，获取其值
                    value = field_value.value
                else:
                    # 直接使用值
                    value = field_value
                metadata[field_name] = value
            except:
                # 如果转换失败，使用原始值的字符串表示
                metadata[field_name] = str(field_value)
        
        # 统计dtype和参数量
        dtype_param_count = {}
        total_params = 0
        
        # 统计tensor名称
        tensor_names = []
        tensor_prefixes = []  # 存储每个tensor名称的第一个部分（第一个点之前的部分）
        tensor_second_prefixes = []  # 存储每个tensor名称的第二部分（第一个点和第二个点之间的部分）
        tensor_third_prefixes = []  # 存储每个tensor名称的第三部分（第二个点和第三个点之间的部分）
        
        for tensor in reader.tensors:
            param_count = np.prod(tensor.shape)
            dtype_name = tensor.tensor_type.name
            dtype_param_count[dtype_name] = dtype_param_count.get(dtype_name, 0) + param_count
            total_params += param_count
            tensor_names.append(tensor.name)  # 记录tensor名称
            
            # 提取名称中的第一个部分（第一个点之前的部分）
            prefix = tensor.name.split('.')[0] if '.' in tensor.name else tensor.name
            tensor_prefixes.append(prefix)
            
            # 提取名称中的第二个部分（第一个点之后、第二个点之前的部分）
            parts = tensor.name.split('.')
            if len(parts) >= 2:
                second_prefix = parts[1]  # 第二个部分
            else:
                second_prefix = ""  # 如果没有第二个部分，则为空字符串
            tensor_second_prefixes.append(second_prefix)
            
            # 提取名称中的第三个部分（第二个点之后、第三个点之前的部分）
            if len(parts) >= 3:
                third_prefix = parts[2]  # 第三个部分
            else:
                third_prefix = ""  # 如果没有第三个部分，则为空字符串
            tensor_third_prefixes.append(third_prefix)
        
        # 计算唯一tensor名称的数量
        unique_names = list(dict.fromkeys(tensor_names))  # 保持首次出现的顺序
        duplicate_names = [name for name in tensor_names if tensor_names.count(name) > 1]
        unique_name_count = len(unique_names)
        
        # 计算唯一前缀的数量
        unique_prefixes = list(dict.fromkeys(tensor_prefixes))  # 保持首次出现的顺序
        prefix_counts = Counter(tensor_prefixes)  # 计算每个前缀出现的次数
        unique_prefix_count = len(unique_prefixes)
        
        # 计算唯一第二前缀的数量
        unique_second_prefixes = [p for p in dict.fromkeys(tensor_second_prefixes) if p]  # 保持首次出现的顺序，排除空字符串
        second_prefix_counts = Counter([p for p in tensor_second_prefixes if p])  # 计算每个第二前缀出现的次数
        unique_second_prefix_count = len(unique_second_prefixes)
        
        # 计算唯一第三前缀的数量
        unique_third_prefixes = [p for p in dict.fromkeys(tensor_third_prefixes) if p]  # 保持首次出现的顺序，排除空字符串
        third_prefix_counts = Counter([p for p in tensor_third_prefixes if p])  # 计算每个第三前缀出现的次数
        unique_third_prefix_count = len(unique_third_prefixes)
        
        # 计算主要量化类型的位宽
        if dtype_param_count:
            main_dtype = max(dtype_param_count, key=dtype_param_count.get)
            main_percentage = (dtype_param_count[main_dtype] / total_params) * 100
            
            # GGUF量化类型映射到位宽
            if 'Q4' in main_dtype or 'IQ4' in main_dtype:
                Q_value = 4
            elif 'Q8' in main_dtype or 'IQ8' in main_dtype:
                Q_value = 8
            elif 'Q2' in main_dtype:
                Q_value = 2
            elif main_dtype in ['F16', 'BF16']:
                Q_value = 16
            elif main_dtype == 'F32':
                Q_value = 32
            else:
                Q_value = 16  # 默认值
            
            memory_gb = calculate_memory_requirement(total_params, Q_value)
        else:
            memory_gb = 0
            Q_value = 16
        
        # 生成报告
        report = f"📄 GGUF文件: {os.path.basename(path)}\n"
        report += f"\n{'─' * hang}\n"  # 分隔线
        # 添加元数据信息
        if metadata:
            report += f"📚 元数据信息:\n"
            for key, value in list(metadata.items())[:10]:  # 显示前10个元数据项
                report += f"   {key}: {value}\n"
            if len(metadata) > 10:
                report += f"   ... 还有 {len(metadata) - 10} 个元数据项\n\n"
            else:
                report += "\n"
        else:
            report += f"📚 元数据: 无\n\n"
        
        report += f"\n{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的tensor名称
        if unique_names:
            report += f"🏷️ 前10个唯一张量名称:\n"
            for i, name in enumerate(unique_names[:10]):
                report += f"   {i+1}. {name}\n"
            if len(unique_names) > 10:
                report += f"   ... 还有 {len(unique_names) - 10} 个名称\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的前缀（第一次出现的前缀）
        if unique_prefixes:
            report += f"🏷️ 第一前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_prefixes[:10]):  # 显示前10个不同的前缀
                count = prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_prefixes) > 10:
                report += f"   ... 还有 {len(unique_prefixes) - 10} 个前缀\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的第二前缀（第一次出现的第二前缀）
        if unique_second_prefixes:
            report += f"🏷️ 第二前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_second_prefixes[:10]):  # 显示前10个不同的第二前缀
                count = second_prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_second_prefixes) > 10:
                report += f"   ... 还有 {len(unique_second_prefixes) - 10} 个第二前缀\n\n"
            else:
                report += "\n"
        
        report += f"{'─' * hang}\n"  # 分隔线
        # 显示前几个唯一的第三前缀（第一次出现的第三前缀）
        if unique_third_prefixes:
            report += f"🏷️ 第三前缀统计 (按首次出现顺序):\n"
            for i, prefix in enumerate(unique_third_prefixes[:10]):  # 显示前10个不同的第三前缀
                count = third_prefix_counts[prefix]
                report += f"   {i+1}. {prefix} ({count} 个张量)\n"
            if len(unique_third_prefixes) > 10:
                report += f"   ... 还有 {len(unique_third_prefixes) - 10} 个第三前缀\n\n"
            else:
                report += "\n"
        

        
        report += f"{'─' * hang}\n"  # 分隔线
        report += f"📊 总参数量: {total_params:,} ({format_param_count_practical(total_params)})\n"
        report += f"📈 张量数量: {len(reader.tensors)} (唯一名称: {unique_name_count}, 重复名称: {len(duplicate_names)})\n"
        report += f"🏷️ 前缀统计: {unique_prefix_count} 个不同第一前缀, {unique_second_prefix_count} 个不同第二前缀, {unique_third_prefix_count} 个不同第三前缀\n"
        report += f" 显存估算: {memory_gb:.1f} GB (基于公式: M = (P × Q) / 8 × 1.2)\n"
        report += f"   - P = {total_params / 1_000_000_000:.1f}B\n"
        report += f"   - Q = {Q_value} (主要格式: {main_dtype})\n\n"
        

        # 显示各类型参数
        sorted_dtypes = sorted(dtype_param_count.items(), key=lambda x: x[1], reverse=True)
        for dtype, param_count in sorted_dtypes:    
            percentage = (param_count / total_params) * 100
            formatted_count = format_param_count_practical(param_count)
            report += f"🔹 {dtype}: {param_count:,} 参数 ({formatted_count}, {percentage:.2f}%)\n"


        report += f"\n{'─' * hang}\n"  # 分隔线
        
        # 保存分析结果到 .checkinfo 文件
        checkinfo_filename = path.rsplit('.', 1)[0] + '.checkinfo'
        try:
            with open(checkinfo_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)
            print(f"✅ 分析结果已保存到: {checkinfo_filename}")
        except Exception as e:
            print(report)
            print(f"⚠️  保存 .checkinfo 文件时出错: {str(e)}")
        
        return dtype_param_count, total_params, memory_gb
        
    except Exception as e:
        error_msg = f"❌ 无法读取GGUF文件 {path}:\n{str(e)}"
        print(error_msg)
        return {}, 0, 0

def on_drop(event):
    
    text_box.delete(1.0, tk.END)
    sys.stdout = StdoutRedirector(text_box)
    
    filepath = event.data.strip("{}")
    if filepath.endswith(".gguf"):
        print("读取 GGUF 文件-"+filepath)
        inspect_gguf(filepath)
    elif filepath.endswith(".safetensors"):
        print("读取 Safetensors 文件-"+filepath)
        dtype_count, total_params, memory_gb = inspect_safetensors(filepath)
        
        if total_params > 0:
            model_size = classify_model_size(total_params)
            print(f"🏷️ 模型规模分类: {model_size}")
    else:
        pass


    # text_box.insert(tk.END, result)

# 创建窗口
root = TkinterDnD.Tk()
root.title("All精度检查工具")
#置顶窗口
root.attributes("-topmost", True)
root.geometry("480x800")

label = tk.Label(root, text="将 .gguf 或 .safetensors 模型文件拖拽到这里", bg="#e0e0e0", relief="ridge", height=5)
label.pack(fill="both", padx=10, pady=10, expand=True)
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', on_drop)

# 创建文本框和滚动条容器
text_frame = tk.Frame(root)
text_frame.pack(fill="both", padx=10, pady=10, expand=True)

# 创建文本框
text_box = tk.Text(text_frame, wrap=tk.WORD)

# 创建垂直滚动条
v_scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_box.yview)

# 配置文本框的滚动条
text_box.config(yscrollcommand=v_scrollbar.set)

# 布局
text_box.grid(row=0, column=0, sticky="nsew")
v_scrollbar.grid(row=0, column=1, sticky="ns")

# 配置网格权重
text_frame.grid_rowconfigure(0, weight=1)
text_frame.grid_columnconfigure(0, weight=1)

root.mainloop()
