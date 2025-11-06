from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
from DataProcess import *

# 禁用所有警告
warnings.filterwarnings("ignore")

def evaluate(model, val_loader, tokenizer, device, print_prediction=False, show_num=show_num):
    """评估模型性能"""
    if model is None:# 说明要自己加载模型
        model = SqueezeformerTransformerModel(
            input_dim=input_dim,
            vocab_size=len(tokenizer),
            d_model=d_model
        ).to(device)
        try:
            model.load_state_dict(torch.load(f'{model_dir}/best_model_weights.pth'))
        except Exception as e:
            print(f"加载模型发生异常")
            return
    model.eval()
    print(f"正在进行模型的评估")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    with torch.no_grad():
        total_loss = 0
        valid_samples = 0
        printed = False  # 标记是否已打印过预测结果
        count = 1
        for batch in tqdm(val_loader):
            # 提取批次数据并移动到设备
            video = batch['video'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            # 构建输入和目标（位移一位）
            tgt_input = text_input_ids[:, :-1]  # 解码器输入（不含最后一个token）
            tgt_output = text_input_ids[:, 1:]  # 目标（不含第一个token）
            # 创建解码器掩码（防止关注未来token）
            tgt_mask = create_mask(tgt_input.size(1)).to(device)  # 需要实现create_mask函数
            # 前向传播
            output = model(
                src=video,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            batch_size_current = video.size(0)
            total_loss += loss.item() * batch_size_current
            valid_samples += batch_size_current
            # 输出一个批次的预测结果
            if print_prediction and not printed:
                print("\n===== 验证集部分视频的预测结果 =====")
                # 获取预测的token IDs（取概率最大的token）
                pred_token_ids = torch.argmax(output, dim=-1)
                # 遍历批次中的样本
                for i in range(min(show_num, batch_size_current)):  # 最多显示 show_num 个样本
                    # 解码预测结果
                    pred_text = tokenizer.decode(
                        pred_token_ids[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    # 解码真实标签
                    true_text = tokenizer.decode(
                        tgt_output[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    print(f"样本 {i + 1}:")
                    print(f"预测结果: {pred_text}")
                    print(f"真实标签: {true_text}")
                printed = True  # 确保只打印一次
            if show_num=='all':  #打印所有的valid
                pred_token_ids = torch.argmax(output, dim=-1)
                for i in range(batch_size_current):
                    pred_text = tokenizer.decode(pred_token_ids[i],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    true_text = tokenizer.decode(tgt_output[i],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    print(f"验证集视频 {count + 1}:")
                    print(f"预测结果: {pred_text}")
                    print(f"真实标签: {true_text}")
                    count += 1
        avg_loss = total_loss / valid_samples
        print(f"平均验证误差 Average Valid Loss: {avg_loss}")
        torch.cuda.empty_cache()  # 清空缓存
    return avg_loss


def train(train_loader, val_loader, device):
    """"""
    # 加载 or 重新创建模型
    model = SqueezeformerTransformerModel(
        input_dim=input_dim,
        # vocab_size=tokenizer.vocab_size,
        vocab_size=len(tokenizer),
        d_model=d_model
    ).to(device)    # 将模型移动到 GPU
    model.train()
    # 新增：记录损失的列表
    train_losses = []
    valid_losses = []
    print_list = [0,2,6,9]# 只有在指定的某几次中才会打印预测结果
    # 新增：跟踪最佳验证损失
    best_val_loss = float('inf')
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 训练循环
    for epoch in range(num_epochs):  # 训练轮数
        model.train()
        print(f"\n正在进行第 {epoch+1} 轮训练")
        total_loss = 0
        train_samples = 0
        for batch in tqdm(train_loader):
            # 提取批次数据并移动到设备
            video = batch['video'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            # 构建输入和目标（位移一位）
            tgt_input = text_input_ids[:, :-1]  # 解码器输入（不含最后一个token）
            tgt_output = text_input_ids[:, 1:]  # 目标（不含第一个token）
            # 创建解码器掩码（防止关注未来token）
            tgt_mask = create_mask(tgt_input.size(1)).to(device)  # 需要实现create_mask函数
            # 前向传播
            output = model(
                src=video,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)),tgt_output.reshape(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_current = video.size(0)
            total_loss += loss.item() * batch_size_current
            train_samples += batch_size_current
        avg_train_loss = total_loss / train_samples
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, 平均训练损失 Average Train Loss: {avg_train_loss}\n")
        if epoch in print_list:
            avg_val_loss = evaluate(model, val_loader, tokenizer, device, print_prediction=True)
        else:
            avg_val_loss = evaluate(model, val_loader, tokenizer, device, print_prediction=False)
        valid_losses.append(avg_val_loss)
        # 新增：保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_dir}/best_model_weights_epoch_{epoch}_loss_{avg_val_loss}.pth")
            torch.save(model, f"{model_dir}/best_model_epoch_{epoch}_{avg_val_loss}.pth")
            torch.save(model.state_dict(), f"{model_dir}/best_model_weights.pth")
            torch.save(model, f"{model_dir}/best_model.pth")
            print(f"更新最佳模型 (验证损失: {best_val_loss:.4f})")
        torch.cuda.empty_cache() # 清空缓存
    print(f"最佳模型验证损失为: {best_val_loss}")
    return model, train_losses, valid_losses

def draw_loss(train_losses, valid_losses):
    # 新增：保存损失数据
    loss_data = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'epochs': list(range(1, num_epochs + 1))
    }
    loss_data_path = f"./Logs/loss_data_{datetime.now().strftime('%y-%m-%d-%H:%M')}.npy"
    np.save(loss_data_path, loss_data)
    print(f"损失数据已保存到 {loss_data_path}")
    # 新增：绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data['epochs'], loss_data['train_losses'], label='Train Loss')
    plt.plot(loss_data['epochs'], loss_data['valid_losses'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss && Valid Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = f"./Images/Loss_Curve_{datetime.now().strftime('%y-%m-%d-%H:%M')}.png"
    plt.savefig(loss_plot_path)
    print(f"损失曲线已保存到 {loss_plot_path}")
    plt.close()



if __name__=="__main__":
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee(f"./Logs/output_of_main_process_time{datetime.now().strftime('%y-%m-%d-%H:%M')}_lr{learning_rate}_outepoch{num_epochs}.log","w")  # "w"表示覆盖写入，"a"表示追加
    # # 打印所有的全局变量
    # print_all_global_var()
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # # 数据预处理
    # DataPreProcess()
    # 创建数据集和数据加载器
    train_loader, val_loader = create_dataloader(val_split=0.2)
    # # 开始训练
    # model, train_losses, valid_losses = train(train_loader, val_loader, device)
    # # 绘制Train loss与Valid loss图像
    # draw_loss(train_losses, valid_losses)
    # 开始测试 将所有的验证集的预测和真实值都对照着打印出来并且保存
    evaluate(None, val_loader, tokenizer, device, show_num='all')











# GPT-2 tokenizer 已设置 pad_token = tokenizer.eos_token，但两者功能不同：
# pad_token 用于填充，训练时被忽略
# eos_token 用于标记序列结束，预测时作为终止信号

# if os.path.exists(model_path):
#     print("发现已有训练一半的模型，加载继续训练...")
#     # model = torch.load(model_path, weights_only=False).to(device)
#     model.load_state_dict(torch.load(f'{model_weights_path}'))
# else:
#     print("未发现训练过的模型，从头开始训练...")


# # 对验证集再取1/10（第二次划分：初始验证集 -> 最终验证集 + 舍弃部分）
# final_val_size = int(0.1 * len(val_dataset))  # 取初始验证集的10%
# # 确保至少保留1个样本（避免数据集为空）
# final_val_size = max(final_val_size, 1)
# discard_size = len(val_dataset) - final_val_size
# torch.manual_seed(random_seed + 1)  # 用不同种子避免和第一次划分冲突
# val_dataset, _ = random_split(val_dataset, [final_val_size, discard_size])