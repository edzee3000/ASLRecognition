from utils import *
from DataProcess import VideoTextDataset
import pickle
import time
from pathlib import Path
import os
from joblib import Parallel, delayed
from tqdm import tqdm


def save_processed_dataset(dataset, save_path):
    """保存处理好的数据集到本地"""
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存数据集
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"数据集已保存到: {save_path}")


def load_processed_dataset(load_path):
    """从本地加载处理好的数据集"""
    try:
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"数据集已从 {load_path} 加载")
        return dataset
    except FileNotFoundError:
        print(f"错误: 文件 {load_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 加载数据集时出错: {e}")
        return None



def process_row(row, parquet_path):
    sequence_id = int(row['sequence_id'])   # 转为 int，防止 pyarrow 报错
    phrase = row['phrase']
    df = pq.read_table(
        parquet_path,
        filters=[('sequence_id', '=', sequence_id)]
    ).to_pandas().fillna(0)
    tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    return tensor, phrase


def parallel_create_list(parquet_path, csv_path="./Data/asl-fingerspelling/my_train.csv",
                     path=None, save_dir=None, n_jobs=-1):
    """"""
    if path is None:
        path = parquet_path
    dataset_df = pd.read_csv(csv_path)
    filtered_csv_df = dataset_df[dataset_df['path'] == path]
    # 并行提取
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row, parquet_path)
        for _, row in tqdm(filtered_csv_df.iterrows(), total=len(filtered_csv_df))
    )
    video_features_list, text_list = zip(*results)
    print(f"{parquet_path} 提取完毕，共 {len(video_features_list)} 条数据")
    return list(video_features_list), list(text_list)





def this_create_list(parquet_path, csv_path="./Data/asl-fingerspelling/my_train.csv", path=None, saved_dir=None, file_id=None):
    """提取数据并保存为张量格式"""
    if path == None:
        path=parquet_path
    # 读取 CSV 文件
    dataset_df = pd.read_csv(csv_path)
    filtered_csv_df = dataset_df[dataset_df['path'] == path]
    video_features_list = []
    text_list = []
    print(f"从第 {filtered_csv_df.index[0]} 个Sequence开始提取数据")
    count = 1
    batch_count = 0
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token  # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    # print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    for index, row in filtered_csv_df.iterrows():  # 从1开始计数
        # if os.path.isfile(parquet_path):
        #     print(f"路径{parquet_path}对应的文件存在，已经提取过了，不再重复提取")
        #     continue
        sequence_id = row['sequence_id']
        phrase = row['phrase']
        sequence_df = pq.read_table(f"{parquet_path}",
                                    filters=[[('sequence_id', '=', sequence_id)], ]).to_pandas()
        sequence_df = sequence_df.fillna(0)
        tensor_array = torch.tensor(sequence_df.to_numpy(), dtype=torch.float32)
        video_features_list.append(tensor_array)
        text_list.append(phrase)
        if count % 256 == 0:
            saved_path = f'{saved_dir}/{file_id}_{batch_count}.pkl'
            print(f"现在提取到了{index}个Sequence，开始保存当前批次数据")
            dataset = VideoTextDataset(video_features_list, text_list, tokenizer, max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len)
            save_processed_dataset(dataset, saved_path)
            video_features_list = None
            text_list = None
            video_features_list = []
            text_list = []
            batch_count += 1
        if count % 50 == 0:
            print(f"现在提取到了{index}个Sequence")
        count+=1
    # 保存最后一批不足 256 个的数据
    if video_features_list:
        saved_path = f'{saved_dir}/{file_id}_{batch_count}.pkl'
        dataset = VideoTextDataset(video_features_list, text_list, tokenizer, max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len)
        save_processed_dataset(dataset, saved_path)
    print(f"{parquet_path}的数据提取完毕")
    print(f"一共提取了{count}条视频数据")
    return








def try_save():
    """"""
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    data_dir = "./Data/asl-fingerspelling"
    # train_dir = f"{data_dir}/train_landmarks"
    train_dir = f"{data_dir}/supplemental_landmarks"
    # csv_path = f"{data_dir}/train.csv"
    csv_path = f"{data_dir}/supplemental_metadata.csv"
    processed_data_dir = "./Data/ProcessedData"
    processed_train_data_dir = f'{processed_data_dir}/supplemental_landmarks'
    # 删除目录下空的文件夹
    for p in Path(processed_train_data_dir).rglob("*"):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()  # 删除空目录
    file_names = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    for file in file_names:
        print(f"file为: {file}")
        file_id = Path(file).stem
        print(f"file_id为: {file_id}")
        parquet_path = f"{train_dir}/{file}"
        print(f"parquet_path为: {parquet_path}")
        rel_parquet_path = os.path.relpath(parquet_path, data_dir)
        print(f"rel_parquet_path为: {rel_parquet_path}")
        # 保存data的文件夹路径
        saved_dir = f"{processed_train_data_dir}/{file_id}"
        print(f"processed_data_dir为: {saved_dir}")
        if Path(saved_dir).is_dir():
            print(f"之前有提取过，不再重复提取")
            continue
        os.makedirs(f"{saved_dir}", exist_ok=True)
        # processed_data_path = Path(processed_data_path).with_suffix(".pkl")  # -> data/file.pkl
        # print(f"processed_data_path为: {processed_data_path}")
        # Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)  # 确保所有父目录存在
        start = time.perf_counter()
        this_create_list(parquet_path=parquet_path, csv_path=csv_path,
                         path=rel_parquet_path,saved_dir=saved_dir,file_id=file_id)
        # dataset = load_processed_dataset(processed_data_path)  # 加载数据
        end = time.perf_counter()
        print(f"读取 {parquet_path} 耗时：{end - start:.4f} 秒")


def main():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee("./output_of_transfer_parquet_to_pkl.log", "w")  # "w"表示覆盖写入，"a"表示追加
    # 执行 save 函数
    try_save()
    # 恢复原始输出（可选）
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 恢复终端输出


if __name__=="__main__":
    main()

# # 保存处理好的数据
# if save_dir:
#     os.makedirs(save_dir, exist_ok=True)
#     file_name = os.path.basename(parquet_path).replace('.parquet', '.pt')
#     save_path = os.path.join(save_dir, file_name)
#     save_subdir = os.path.dirname(save_path)  # 取出 save_path 的父级
#     os.makedirs(save_subdir, exist_ok=True)  # 递归创建所有中间目录
#     # 保存为字典格式
#     data_dict = {
#         'video_features': video_features_list,
#         'texts': text_list,
#         'sequence_ids': filtered_csv_df['sequence_id'].tolist()
#     }
#     torch.save(data_dict, save_path)
#     print(f"数据已保存至: {save_path}")