from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
from DataProcess import *
from Transfer_to_pkl import *
import time
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



def train():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee("./output.log", "w")  # "w"表示覆盖写入，"a"表示追加
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    csv_path = "./Data/SimpleData/my_train.csv"
    model_dir = "./Models"
    parquet_dir = "./Data/SimpleData"
    # parquet_list = ["./Data/SimpleData/450474571.parquet" , "./Data/SimpleData/5414471.parquet"]
    parquet_list = ["./Data/SimpleData/450474571.parquet"]
    # parquet_list = None
    if parquet_list == None:
        """如果是None的话将所有csv下的parquet路径都包含进来"""
        parquet_list = create_parquet_path(csv_path=csv_path, parquet_dir=parquet_dir)
    # Load pre-trained BERT tokenizer
    # tokenizer = BertTokenizer.from_pretrained(f"{bert_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    # start_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id # 假设开始的特殊标记 token 的 id 为 start_token_id
    # end_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id
    model = SqueezeformerTransformerModel(
        input_dim=1630,
        vocab_size=tokenizer.vocab_size,
        d_model=512
    ).to(device)    # 将模型移动到 GPU
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for idx, parquet_path in enumerate(parquet_list):
        """逐次遍历所有的parquet文件"""
        print(f"这是第{idx + 1}个parquet文件数据 文件数据路径为{parquet_path}")
        if not os.path.exists(parquet_path):
            print(f"错误: 文件 {parquet_path} 不存在，进入下一个数据的训练")
            continue
        # video_features_list, text_list = create_list(parquet_path=parquet_path, csv_path=csv_path)
        # print(f"提取{parquet_path}文件数据完毕")
        # dataset = VideoTextDataset(video_features_list, text_list, tokenizer,
        #                            max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len)
        # 保存data的路径
        processed_data_dir = "./Data/ProcessedData"
        processed_data_path = f"{processed_data_dir}/450474571.pkl"
        dataset = load_processed_dataset(processed_data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(inner_epochs):
            total_parquet_loss = 0
            start = time.perf_counter()
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
                output_dim = output.size(-1)
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt_output.contiguous().view(-1)
                # Calculate loss
                loss = criterion(output, tgt_output)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_parquet_loss += loss.item()
                # print(f"Epoch: {epoch + 1}, Batch: {batch_idx+1}, BatchLoss: {loss.item()}")
            print(f'InnerEpoch: {epoch + 1}, Parquet: {idx+1}, ParquetLoss: {total_parquet_loss / len(dataloader)}')
            end = time.perf_counter()
            # print(f"耗时：{end - start:.4f} 秒")
    torch.save(model.state_dict(), f'{model_dir}/model_weights.pth')
    torch.save(model, f'{model_dir}/model.pth')
    print(f"model保存成功")
    # 恢复原始输出（可选）
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 恢复终端输出





