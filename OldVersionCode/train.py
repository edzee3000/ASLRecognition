from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
from DataProcess import *
from Transfer_to_pkl import *

# def train():
#     """"""
#     # 重定向输出：同时写入log.txt和终端
#     sys.stdout = Tee("./output.log", "w")  # "w"表示覆盖写入，"a"表示追加
#     # 检查 CUDA 是否可用
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     # 初始化数据信息
#     csv_path = "./Data/my_train.csv"
#     model_dir = "./Models"
#     parquet_dir = "./Data"
#     parquet_list = ["./Data/450474571.parquet" , "./Data/5414471.parquet"]
#     # parquet_list = ["./Data/450474571.parquet"]
#     # parquet_list = None
#     if parquet_list == None:
#         """如果是None的话将所有csv下的parquet路径都包含进来"""
#         parquet_list = create_parquet_path(csv_path=csv_path, parquet_dir=parquet_dir)
#     # Load pre-trained BERT tokenizer
#     tokenizer = BertTokenizer.from_pretrained(f"{tokenizer_path}")
#     model = SqueezeformerTransformerModel(
#         input_dim=1630,
#         vocab_size=tokenizer.vocab_size,
#         d_model=512
#     ).to(device)    # 将模型移动到 GPU
#     model.train()
#     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     for out_epoch in range(out_epochs):
#         for idx, parquet_path in enumerate(parquet_list):
#             """逐次遍历所有的parquet文件"""
#             print(f"这是第{idx + 1}个parquet文件数据 文件数据路径为{parquet_path}")
#             if not os.path.exists(parquet_path):
#                 print(f"错误: 文件 {parquet_path} 不存在，进入下一个数据的训练")
#                 continue
#             video_features_list, text_list = create_list(parquet_path=parquet_path, csv_path=csv_path)
#             dataset = VideoTextDataset(video_features_list, text_list, tokenizer,
#                                        max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len)
#             dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#             for epoch in range(inner_epochs):
#                 total_parquet_loss = 0
#                 # print(f"这是第{idx + 1}个parquet文件数据的第{epoch+1}轮")
#                 for batch_idx, batch in enumerate(dataloader):
#                     video = batch['video'].to(device)  # 将数据移动到 GPU
#                     text_input_ids = batch['text_input_ids'].to(device)
#                     src_mask = batch['src_mask'].to(device)
#                     # Create target tensor (shifted by one position)
#                     tgt_input = text_input_ids[:, :-1]
#                     tgt_output = text_input_ids[:, 1:]
#                     # Create masks
#                     tgt_mask = create_mask(tgt_input).to(device)  # 将 mask 移动到 GPU
#                     output = model(video, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
#                     # Reshape output and target for loss calculation
#                     output_dim = output.size(-1)
#                     output = output.contiguous().view(-1, output_dim)
#                     tgt_output = tgt_output.contiguous().view(-1)
#                     # Calculate loss
#                     loss = criterion(output, tgt_output)
#                     # Backward and optimize
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     total_parquet_loss += loss.item()
#                     # print(f"Epoch: {epoch + 1}, Batch: {batch_idx+1}, BatchLoss: {loss.item()}")
#                 print(f'OutEpoch: {out_epoch+1}, InnerEpoch: {epoch + 1}, Parquet: {idx+1}, ParquetLoss: {total_parquet_loss / len(dataloader)}')
#         # 每训练一批数据就保存模型权重到文件
#         torch.save(model.state_dict(), f'{model_dir}/model_weights.pth')
#         # 保存整个模型到文件
#         torch.save(model, f'{model_dir}/model.pth')
#         # print(f"训练完{parquet_path}之后 第{idx+1}个的model保存成功")
#         print(f"外部第{out_epoch + 1}轮之后 model保存成功")
#     # 恢复原始输出（可选）
#     sys.stdout.close()
#     sys.stdout = sys.__stdout__  # 恢复终端输出

def print_all_global_var():
    """ 打印所有全局变量 """
    print(f"所有全局变量为:")
    # 获取全局符号表
    global_vars = globals()
    # 过滤掉以双下划线开头的内置全局变量
    filtered_vars = {k: v for k, v in global_vars.items() if not k.startswith('__')}
    for var_name, var_value in filtered_vars.items():
        print(f"{var_name}: {var_value}")


def creat_record_csv(train_record_csv):
    """"""
    # 检查 CSV 文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(train_record_csv):
        with open(train_record_csv, 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'train_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def obtain_record_csv(train_record_csv):
    """"""
    # 读取 CSV 文件，获取每个文件的训练次数
    train_counts = {}
    with open(train_record_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_counts[row['file_path']] = int(row['train_count'])
    return train_counts

def update_train_count(file_name, train_counts):
    """"""
    # 更新训练次数
    if file_name in train_counts:
        train_counts[file_name] += 1
    else:
        train_counts[file_name] = 1
    return train_counts

def updata_record_csv(train_record_csv, train_counts):
    """"""
    # 更新 CSV 文件
    with open(train_record_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'train_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for path, count in train_counts.items():
            writer.writerow({'file_path': path, 'train_count': count})
    print(f"更新record_csv文件 {train_record_csv} 成功")





def train():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee(f"./output_of_train_process_time{datetime.now().strftime('%y-%m-%d-%H:%M')}_lr{learning_rate}_outepoch{out_epochs}.log", "w")  # "w"表示覆盖写入，"a"表示追加
    print_all_global_var()
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    csv_path = "./Data/asl-fingerspelling/train.csv"
    model_dir = "./Models"
    pkl_dir = f"./Data/ProcessedData"
    pkl_train_dir = f"{pkl_dir}/train_landmarks"
    model_weights_path = f"{model_dir}/model_weights.pth"
    model_path = f'{model_dir}/model.pth'  # 检查是否有训练一半的模型
    train_record_csv = f"{pkl_dir}/train_record.csv" # 记录训练次数的 CSV 文件路径
    creat_record_csv(train_record_csv=train_record_csv)
    train_counts = obtain_record_csv(train_record_csv=train_record_csv)
    # Load gpt-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    # 加载 or 重新创建模型
    model = SqueezeformerTransformerModel(
        input_dim=1630,
        vocab_size=tokenizer.vocab_size,
        d_model=512
    ).to(device)    # 将模型移动到 GPU
    if os.path.exists(model_path):
        print("发现已有训练一半的模型，加载继续训练...")
        # model = torch.load(model_path, weights_only=False).to(device)
        model.load_state_dict(torch.load(f'{model_weights_path}'))
    else:
        print("未发现训练过的模型，从头开始训练...")
    model.train()
    # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    files = [p for p in Path(pkl_train_dir).rglob("*") if p.is_file()]
    # 记录每一轮的训练损失和验证损失
    train_losses = []
    valid_losses = []
    for out_epoch in range(out_epochs):
        total_train_loss = 0
        total_items = 0
        print(f"这个是第 {out_epoch+1} 轮外部循环 训练所有数据")
        for idx, file_path in enumerate(files, 1):
            # if idx > 20:
            #     continue
            # file_train_loss = 0
            # print(f"这个是第{idx}个 pkl 训练数据文件: {file_path}")
            file_name = Path(file_path).relative_to(Path(pkl_train_dir))
            file_name = str(file_name)
            print(f"这个是第{idx}个 pkl 训练数据文件: {file_path}    pkl 训练数据文件名为: {file_name}")
            # 检查文件是否已经训练过很多轮了
            if file_name in train_counts and (train_counts[file_name] >= out_epochs or train_counts[file_name] >= out_epoch+1):
                if train_counts[file_name] >= out_epochs:
                    print(f"跳过已经训练过 {out_epochs} 次数的文件: {file_name}")
                elif train_counts[file_name] >= out_epoch+1:
                    print(f"目前是第 {out_epoch+1} 轮外部循环，但是 {file_name} 已经训练过 {train_counts[file_name]} 轮外部循环了")
                continue
            dataset = load_processed_dataset(file_path)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            start = time.perf_counter()
            for epoch in range(inner_epochs):
                total_parquet_loss = 0
                # print(f"这是第{idx + 1}个parquet文件数据的第{epoch+1}轮")
                for batch_idx, batch in enumerate(dataloader):
                    video = batch['video'].to(device)  # 将数据移动到 GPU
                    text_input_ids = batch['text_input_ids'].to(device)
                    src_mask = batch['src_mask'].to(device)
                    # Create target tensor (shifted by one position)
                    tgt_input = text_input_ids[:, :-1]
                    tgt_output = text_input_ids[:, 1:]
                    # Create masks  创建掩码
                    tgt_mask = create_mask(tgt_input).to(device)  # 将 mask 移动到 GPU
                    output = model(video, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
                    # Reshape output and target for loss calculation
                    output_dim = output.size(-1)  # output_dim：通常是 词汇表大小（即分类类别数）
                    output = output.contiguous().view(-1, output_dim)  # shape 为 torch.Size([928, 50257])
                    tgt_output = tgt_output.contiguous().view(-1)   # shape 为 torch.Size([928])
                    # Calculate loss
                    loss = criterion(output, tgt_output)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_parquet_loss += loss.item()
                    # print(f"Epoch: {epoch + 1}, Batch: {batch_idx+1}, BatchLoss: {loss.item()}")
                    # file_train_loss += loss.item()
                total_train_loss += total_parquet_loss
                print(f'OuterEpoch: {out_epoch+1}, InnerEpoch: {epoch + 1}, train_landmarks_pkl: {idx}, ParquetLoss: {total_parquet_loss / len(dataloader)}')
            total_items += len(dataloader)
            end = time.perf_counter()
            # print(f"对 {file_path} 数据文件训练 {inner_epochs} 轮耗时：{end - start:.4f} 秒")   # 基本一批数据训练时间在3.0s-3.6s之间
            # torch.cuda.empty_cache()  # 清空 GPU 缓存
            # # 保存模型
            # if idx % 20==0:# 每20批次保存一次  不然太频繁了
            #     torch.save(model.state_dict(), f'{model_weights_path}')
            #     torch.save(model, f'{model_path}')
            #     print(f"训练完 {file_path} 后，model保存成功")
            # 保存模型后更新 record_csv
            train_counts = update_train_count(file_name=file_name, train_counts=train_counts)  # 更新训练次数
            updata_record_csv(train_record_csv=train_record_csv, train_counts=train_counts)  # 更新 CSV 文件
        # 每训练完一轮保存一次  不然太频繁了
        torch.save(model.state_dict(), f'{model_weights_path}')
        torch.save(model, f'{model_path}')
        # 计算本轮的训练损失
        train_loss = total_train_loss / total_items
        train_losses.append(train_loss)
        print(f'第 {out_epoch + 1} 轮训练损失: {train_loss}')
        # 进行验证
        model.eval()
        from Valid import validate_on_valid_data
        valid_loss = validate_on_valid_data(model=model)
        valid_losses.append(valid_loss)
        print(f'第 {out_epoch + 1} 轮验证损失: {valid_loss}')
        model.train()
        print(f"{pkl_train_dir} 下的所有训练数据文件都已经训练完毕一轮  model模型保存成功")
    # 恢复原始输出（可选）
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 恢复终端输出







 # for root_dir, dir_names, files in os.walk(f"{pkl_train_dir}"):
    #     # print(root_dir)
    #     # for dir_name in dir_names:
    #     #     print(dir_name)
    #     # file_path = os.path.join(root_dir, file_name)

# if out_epoch < 5:
#     learning_rate = 1e-4
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# elif out_epoch < 10:
#     learning_rate = 5e-5
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# elif out_epoch < 15:
#     learning_rate = 1e-5
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# else:
#     learning_rate = 5e-6
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)