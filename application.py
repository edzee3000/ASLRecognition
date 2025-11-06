from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *
import cv2
import mediapipe as mp
from main import VideoTextDataset



# 定义需要保留的关键点索引（与数据预处理一致）
LPOSE = [13, 15, 17, 19, 21]  # 左侧身体左侧关键点索引
RPOSE = [14, 16, 18, 20, 22]  # 身体右侧关键点索引
POSE = LPOSE + RPOSE  # 合并身体关键点索引（共10个）

# 初始化 MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """
    从 MediaPipe 结果中提取156维关键点，严格匹配FEATURE_COLUMNS顺序：
    [右手x0-20, 左手x0-20, 身体x(LPOSE+RPOSE),
     右手y0-20, 左手y0-20, 身体y(LPOSE+RPOSE),
     右手z0-20, 左手z0-20, 身体z(LPOSE+RPOSE)]
    """
    # 1. 提取手部关键点（按0-20索引顺序）
    # 右手关键点 (21点) - 修正默认值为包含x,y,z属性的字典转换对象
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
    else:
        # 当检测不到时，使用默认值(0.0)创建模拟的关键点对象
        rh_landmarks = [type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0}) for _ in range(21)]
    # 左手关键点 (21点) - 同样修正默认值
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
    else:
        lh_landmarks = [type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0}) for _ in range(21)]
    # 2. 提取身体关键点（严格按LPOSE + RPOSE顺序）
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
    else:
        # 身体关键点默认值处理
        pose_landmarks = [type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0}) for _ in range(33)]
    selected_pose = [pose_landmarks[i] for i in POSE]  # 按POSE顺序筛选
    # 3. 按 x→y→z 顺序构建特征
    # 3.1 X坐标组：右手x → 左手x → 身体x
    rh_x = np.array([rh.x for rh in rh_landmarks], dtype=np.float32)
    lh_x = np.array([lh.x for lh in lh_landmarks], dtype=np.float32)
    pose_x = np.array([p.x for p in selected_pose], dtype=np.float32)
    x_group = np.concatenate([rh_x, lh_x, pose_x])
    # 3.2 Y坐标组：右手y → 左手y → 身体y
    rh_y = np.array([rh.y for rh in rh_landmarks], dtype=np.float32)
    lh_y = np.array([lh.y for lh in lh_landmarks], dtype=np.float32)
    pose_y = np.array([p.y for p in selected_pose], dtype=np.float32)
    y_group = np.concatenate([rh_y, lh_y, pose_y])
    # 3.3 Z坐标组：右手z → 左手z → 身体z
    rh_z = np.array([rh.z for rh in rh_landmarks], dtype=np.float32)
    lh_z = np.array([lh.z for lh in lh_landmarks], dtype=np.float32)
    pose_z = np.array([p.z for p in selected_pose], dtype=np.float32)
    z_group = np.concatenate([rh_z, lh_z, pose_z])
    # 合并为156维特征 (21+21+10)×3 = 156
    return np.concatenate([x_group, y_group, z_group])


def process_video(video_path):
    """
    处理视频文件，提取每帧的 1630 维度关键点特征
    :param video_path: 视频文件路径
    :return: 包含所有帧 1630 维度关键点特征的二维数组
    """
    cap = cv2.VideoCapture(video_path)
    keypoint_sequences = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将图像从 BGR 转换为 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # 进行关键点检测
        results = holistic.process(image)
        image.flags.writeable = True
        # 提取符合训练数据格式的 156 维度关键点
        keypoints = extract_keypoints(results)
        keypoint_sequences.append(keypoints)
    cap.release()
    return np.array(keypoint_sequences)



def test(sequence_tensor):
    """"""
    input_dim = 156
    batch_size = 64
    max_video_frame_len = 192  # 允许 video 的最大帧数
    max_text_phrase_len = 20  # 允许生成文本 text 的个数
    # 新增：定义特殊token
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee(f"./Logs/output_of_application_time{datetime.now().strftime('%y-%m-%d-%H:%M')}.log", "w")  # "w"表示覆盖写入，"a"表示追加
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化数据信息
    model_dir = "./Models"
    # Load gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token  # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    special_tokens_dict = {'pad_token': tokenizer.eos_token,  'additional_special_tokens': [START_TOKEN, END_TOKEN]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"词汇表tokenizer.vocab_size的大小为：{len(tokenizer)}")
    # 加载模型
    model = SqueezeformerTransformerModel(
        input_dim=input_dim,
        vocab_size=len(tokenizer),
        d_model=d_model
    ).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/best_model_weights.pth'))
    model.eval()
    # 加载数据
    # 处理输入序列（截断或填充到最大长度）
    sequence_tensor = sequence_tensor.numpy()  # 转为numpy便于处理
    useful_frame_len = min(len(sequence_tensor), max_video_frame_len)
    if len(sequence_tensor) > max_video_frame_len:
        sequence_tensor = sequence_tensor[:max_video_frame_len]  # 截断
    else:
        # 填充到最大长度
        pad_length = max_video_frame_len - len(sequence_tensor)
        sequence_tensor = np.pad(
            sequence_tensor,
            ((0, pad_length), (0, 0)),
            mode='constant',
            constant_values=0.0
        )
    # 转为tensor并添加批次维度
    sequence_tensor = torch.from_numpy(sequence_tensor).float().unsqueeze(0).to(device)
    # print(sequence_tensor)
    # 创建视频掩码（1表示有效帧，0表示填充帧）
    src_mask = torch.zeros(max_video_frame_len, dtype=torch.bool)
    src_mask[:useful_frame_len] = 1
    src_mask = src_mask.unsqueeze(0).to(device)  # 添加批次维度
    # 生成预测文本
    # 初始化解码器输入为 start token
    start_token_id = tokenizer.convert_tokens_to_ids(START_TOKEN)
    end_token_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    tgt_input = torch.tensor([[start_token_id]], device=device)
    with torch.no_grad():
        for _ in range(max_text_phrase_len - 1):  # 限制最大生成长度
            # 创建解码器掩码
            tgt_mask = create_mask(tgt_input.size(1)).to(device)
            # 模型预测
            output = model(
                src=sequence_tensor,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            # 获取最后一个token的预测结果
            # print(output.shape)
            next_token_logits = output[:, -1, :]
            # 加入温度调节（核心修改部分）
            temperature = 1  # 可调整：0.1-1.0之间，值越小越确定
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                # 基于概率分布采样而非argmax
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # 温度为0时退化为贪心搜索
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            # next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            # 将预测的token添加到输入中
            tgt_input = torch.cat([tgt_input, next_token_id], dim=1)
            # 如果预测到结束token则停止
            if next_token_id.item() == end_token_id:
                break
    # 解码预测结果
    predicted_text = tokenizer.decode(
        tgt_input.squeeze().cpu().numpy(),
        # skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    print(f"预测结果: {predicted_text}")
    return predicted_text

if __name__=="__main__":
    """"""
    video_path = f'./Data/Real_Video_Data/Video_Data/Flawless.mp4'
    sequence_path = f"./Data/Real_Video_Data/Keypoint_Data/Flawless.pt"
    # print("video视频数据正在提取关键点")
    # keypoint_sequences = process_video(video_path)
    # print(keypoint_sequences.shape)
    # sequence_tensor = torch.from_numpy(keypoint_sequences).float()  # 将 numpy.array 转换为 tensor
    # torch.save(sequence_tensor, f'{sequence_path}')  # 保存 tensor 到本地
    # print(f"提取关键点成功并且保存数据至本地")
    # assert 0
    sequence_tensor = torch.load(f"{sequence_path}")
    print(sequence_tensor)
    test(sequence_tensor)

