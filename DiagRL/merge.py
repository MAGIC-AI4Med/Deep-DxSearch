import os
import json
from pathlib import Path

def merge_jsonl_files(input_dir, output_file):
    """
    合并指定目录下所有的.jsonl文件到一个输出文件中
    
    Args:
        input_dir (str): 输入目录路径
        output_file (str): 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 用于统计处理的文件数和行数
    total_files = 0
    total_lines = 0
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 递归遍历目录
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    print(f"正在处理文件: {file_path}")
                    
                    # 读取并写入每个文件的内容
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            # 验证JSON格式
                            try:
                                json.loads(line.strip())
                                outfile.write(line)
                                total_lines += 1
                            except json.JSONDecodeError:
                                print(f"警告: 在文件 {file_path} 中发现无效的JSON行，已跳过")
    
    print(f"\n处理完成！")
    print(f"总共处理了 {total_files} 个文件")
    print(f"合并了 {total_lines} 行数据")
    print(f"输出文件保存在: {output_file}")

if __name__ == "__main__":
    # 设置输入目录和输出文件路径
    input_directory = "./your/path/to/wikipedia/chunk"  # 当前目录
    output_file = "./your/path/to/wikipedia/corpus/wikipedia.jsonl"
    
    merge_jsonl_files(input_directory, output_file) 