from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
from DataProcess import *


def test():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee("./output_of_test.log", "w")  # "w"表示覆盖写入，"a"表示追加
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    # csv_path = "./Data/ProcessedData/my_test.csv"
    # parquet_path = "./Data/ProcessedData/33432165.parquet"
    csv_path = "./Data/SimpleData/my_train.csv"
    parquet_path = "./Data/SimpleData/450474571.parquet"
    model_dir = "./Models"
    # Load pre-trained BERT tokenizer
    # tokenizer = BertTokenizer.from_pretrained(f"{bert_path}")
    # Load gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token  # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    start_token_id = tokenizer.bos_token_id if hasattr(tokenizer,'bos_token_id') else tokenizer.cls_token_id  # 假设开始的特殊标记 token 的 id 为 start_token_id
    end_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id
    # print(f"end_token_id: {end_token_id}")   # GPT-2 官方只有 <|endoftext|> 一个特殊符，bos_token_id 只是库的“占位符”，默认与 eos_token_id 共享同一个 id。
    # print(f"start_token_id: {start_token_id}")
    # 加载模型
    model = SqueezeformerTransformerModel(
        input_dim=1630,
        vocab_size=tokenizer.vocab_size,
        d_model=512
    ).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/model_weights.pth'))
    model.eval()
    # 加载数据
    video_features_list, text_list = create_list(parquet_path=parquet_path, csv_path=csv_path, test_mode=True)
    for i in range(5):
        print(f"张量 {i + 1} 的内容: {text_list[i]}")
    dataset = VideoTextDataset(video_features_list, text_list, tokenizer,
                               max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            video = batch['video'].to(device)  # 将数据移动到 GPU
            text_input_ids = batch['text_input_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            # Create target tensor (shifted by one position)  # tgt_input = text_input_ids[:, :-1]  # tgt_output = text_input_ids[:, 1:]
            tgt_input = text_input_ids[:, :-1]
            tgt_output = text_input_ids[:, 1:]
            # Create masks
            tgt_mask = create_mask(tgt_input).to(device)  # 将 mask 移动到 GPU
            output = model(video, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
            # Reshape output and target for loss calculation
            output_dim = output.size(-1)
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            # 打印模型推理结果和原本的文本结果
            output = output.view(-1, output.size(-1))
            _, predicted = torch.max(output, dim=1)
            print(f"Batch {batch_idx + 1}:")
            predicted_texts = []
            original_texts = []
            for i in range(predicted.size(0)):  # 遍历每个样本
                predicted_text = tokenizer.decode(predicted[i], skip_special_tokens=True)
                original_text = tokenizer.decode(tgt_output[i], skip_special_tokens=True)
                predicted_texts.append(predicted_text)
                original_texts.append(original_text)
            print(f"Predicted Texts: {predicted_texts}")
            print(f"Original Texts: {original_texts}")
    print(f'Total Test Loss: {total_loss / len(dataloader)}')
    # 恢复原始输出（可选）
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 恢复终端输出

# this_batch_size = text_input_ids.size(0)
# start_tokens = torch.full((this_batch_size, 1), start_token_id, dtype=torch.long,
#                           device=device)  # 在 tgt_input 的第一列加上开始的特殊标记 token
# end_tokens = torch.full((this_batch_size, 1), end_token_id, dtype=torch.long,
#                         device=device)  # 在 tgt_output 的末尾加上结束的特殊标记 token
# tgt_input = torch.cat([start_tokens, text_input_ids], dim=1)
# tgt_output = torch.cat([text_input_ids, end_tokens], dim=1)



if __name__=="__main__":
    """"""
    test()

