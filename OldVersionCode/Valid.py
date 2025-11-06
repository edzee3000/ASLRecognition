from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
from DataProcess import *
from Transfer_to_pkl import *



def validate_on_valid_data(model=None):
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    csv_path = "./Data/asl-fingerspelling/train.csv"
    model_dir = "./Models"
    valid_dir = "./Data/ProcessedData/ValidData"
    pkl_dir = f"./Data/ProcessedData"
    train_record_csv = f"{pkl_dir}/train_record.csv"
    # 加载gpt-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    # 加载模型
    if model is None:
        model = SqueezeformerTransformerModel(
            input_dim=1630,
            vocab_size=tokenizer.vocab_size,
            d_model=512
        ).to(device)
        model_weights_path = f"{model_dir}/model_weights.pth"
        if os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path))
        else:
            print("模型权重文件不存在，无法进行验证！")
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            return
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    valid_files = [p for p in Path(valid_dir).rglob("*") if p.is_file()]
    count = 0
    for idx, file_path in enumerate(valid_files, 1):
        # if idx > 5:
        #     continue
        # print(f"正在验证第{idx}个文件: {file_path}")
        dataset = load_processed_dataset(file_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start = time.perf_counter()
        file_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            video = batch['video'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_input = text_input_ids[:, :-1]
            tgt_output = text_input_ids[:, 1:]
            tgt_mask = create_mask(tgt_input).to(device)
            with torch.no_grad():
                output = model(video, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
                output_dim = output.size(-1)
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt_output.contiguous().view(-1)
                loss = criterion(output, tgt_output)
                total_loss += loss.item()
                file_loss += loss.item()
        # print(f'Valid Loss: {file_loss / len(dataloader)}')
        count += len(dataloader)
        end = time.perf_counter()
        # print(f"对 {file_path} 数据文件验证耗时：{end - start:.4f} 秒")
        # print(f"验证第{idx}个文件:{file_path}完毕  Valid Loss:{file_loss / len(dataloader)}  对{file_path}数据文件验证耗时:{end - start:.4f}秒")
    average_loss = total_loss / count
    print(f'在ValidData上的平均验证损失: {average_loss}')
    return average_loss

if __name__=="__main__":
    ''''''
    validate_on_valid_data()
