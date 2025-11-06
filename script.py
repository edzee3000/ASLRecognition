import pandas as pd
import os
from utils import *



def replace_path_in_csv(input_csv, output_csv, old_path="train_landmarks", new_path="./Data/SimpleData"):
    """
    读取CSV文件，替换path列中的路径部分，并保存为新的CSV文件
    参数:
    input_csv (str): 输入CSV文件路径
    output_csv (str): 输出CSV文件路径
    old_path (str): 需要替换的旧路径部分
    new_path (str): 替换为的新路径部分
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    # 确保path列存在
    if 'path' not in df.columns:
        raise ValueError(f"CSV文件中没有找到'path'列: {input_csv}")
    # 替换路径
    def replace_path(row):
        path = row['path']
        # 使用os.path.join确保路径分隔符正确
        if old_path in path:
            # 获取文件名部分
            filename = os.path.basename(path)
            # 构建新路径
            return os.path.join(new_path, filename)
        return path
    df['path'] = df.apply(replace_path, axis=1)
    # 保存为新的CSV文件
    df.to_csv(output_csv, index=False)
    print(f"已处理并保存至: {output_csv}")

    return df


def print_columns():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee("./sequence_columns.txt", "w")  # "w"表示覆盖写入，"a"表示追加
    # 打印所有的列名
    sequence_df = pq.read_table(f"./Data/450474571.parquet",
                                filters=[[('sequence_id', '=', 2138669776)], ]).to_pandas()
    for index, column in enumerate(sequence_df.columns):
        print(f"第{index}列列名为: {column}")



def download_gpt2_tokenizer():
    from transformers import GPT2Tokenizer
    # 加载预训练的GPT - 2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f'gpt2')
    # 保存tokenizer到本地路径
    local_tokenizer_path = './Tokenizer/gpt2-tokenizer'
    tokenizer.save_pretrained(local_tokenizer_path)
    print(f"gpt2-tokenizer下载成功")




def Create_ValidData():
    """"""
    # 源目录，即包含二级子目录的目录
    source_dir = "./Data/ProcessedData/train_landmarks"
    # 目标目录，用于存放验证数据
    target_dir = "./Data/ProcessedData/ValidData"
    os.makedirs(target_dir, exist_ok=True)
    # 遍历源目录下的二级子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            # 获取子目录下的所有文件
            files = os.listdir(subdir_path)
            if files:
                # 随机选择一个文件
                selected_file = random.choice(files)
                selected_file_path = os.path.join(subdir_path, selected_file)
                # 创建目标目录下的同名子目录
                target_subdir = os.path.join(target_dir, subdir)
                os.makedirs(target_subdir, exist_ok=True)
                # 移动文件到目标目录的同名子目录下
                shutil.move(selected_file_path, os.path.join(target_subdir, selected_file))

def clean_path(paths, dry_run=False, keep_dir=False):
    for path in paths:
        # 检查路径是否存在
        if not os.path.exists(path):
            continue
        os.remove(path)



def generate_tgt_mask(seq_len):
    # 生成一个上三角矩阵（对角线为1），然后取反
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # # mask = torch.triu(torch.ones(seq_len, seq_len))
    # mask = torch.tril(torch.ones((seq_len, seq_len)))
    # # 转换为布尔值掩码，True表示要屏蔽的位置
    # mask = mask == 0
    return mask  # shape: [seq_len, seq_len]

# 使用示例
if __name__ == "__main__":
    # input_file = "./Data/ProcessedData/supplemental_metadata.csv"  # 替换为实际的输入CSV文件路径
    # output_file = "./Data/ProcessedData/my_test.csv"  # 替换为实际的输出CSV文件路径
    # # # 处理文件
    # processed_df = replace_path_in_csv(input_file, output_file, old_path="supplemental_landmarks", new_path="./Data/ProcessedData")
    # # # 打印前几行查看结果
    # print("\n处理后的前几行数据:")
    # print(processed_df.head())
    # # print_columns()
    # # download_gpt2_tokenizer()
    # Create_ValidData()
    # clean_path([f"./Data/ProcessedData/train_record.csv",f"./Models/model.pth",f"Models/model_weights.pth"])
    # print(datetime.now().strftime("%y-%m-%d-%H:%M"))
    # 示例：生成长度为5的掩码
    tgt_mask = generate_tgt_mask(5)
    print(tgt_mask)

