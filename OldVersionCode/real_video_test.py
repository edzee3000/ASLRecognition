import cv2
import mediapipe as mp
import numpy as np
from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *
from SqueezeformerTransformerModel import *

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
    从 MediaPipe 检测结果中提取关键点并转换为一维数组
    :param results: MediaPipe Holistic 检测结果
    :return: 包含所有关键点信息的一维数组
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    # print(f"pose: {pose.shape}")
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    # print(f"face: {face.shape}")
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    # print(f"lh: {lh.shape}")
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    # print(f"rh: {rh.shape}")
    # assert 0
    return np.concatenate([pose, face, lh, rh])

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
        # 提取 1629 维度关键点
        keypoints = extract_keypoints(results)
        # 添加帧编号，形成 1630 维度
        keypoints_with_frame = np.insert(keypoints, 0, frame_num)
        keypoint_sequences.append(keypoints_with_frame)
        frame_num += 1
    cap.release()
    return np.array(keypoint_sequences)


def video_test(sequence_tensor):
    """"""
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sequence_tensor = sequence_tensor.to(device)
    # 初始化数据信息
    model_dir = "./Models"
    gpt_tokenizer_path = "gpt2"  # 请根据实际情况修改
    # Load gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
    tokenizer.pad_token = tokenizer.eos_token  # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    start_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id
    end_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id
    # 加载模型
    model = SqueezeformerTransformerModel(
        input_dim=1630,
        vocab_size=tokenizer.vocab_size,
        d_model=512
    ).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/model_weights.pth'))
    model.eval()
    # 模拟一个目标输入，这里简单地用开始标记作为初始输入
    batch_size = 1
    tgt_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
    # src_mask = torch.ones(sequence_tensor.size(0), dtype=torch.bool, device=device)
    video_len = min(sequence_tensor.size(0), max_video_frame_len)
    padded_video = torch.zeros(max_video_frame_len, sequence_tensor.size(1), dtype=sequence_tensor.dtype,device=sequence_tensor.device)
    padded_video[:video_len] = sequence_tensor[:video_len]
    src_mask = torch.zeros(max_video_frame_len, dtype=torch.bool, device=sequence_tensor.device)
    src_mask[:video_len] = 1
    temperature = 1
    # 进行推理
    with torch.no_grad():
        while True:
            tgt_mask = create_mask(tgt_input).to(device)
            output = model(padded_video.unsqueeze(0), tgt_input, src_mask=src_mask.unsqueeze(0), tgt_mask=tgt_mask,
                           memory_mask=src_mask.unsqueeze(0))
            logits = output[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # 采样
            # _, predicted = torch.max(output, dim=-1)
            # next_token = predicted[:, -1].unsqueeze(1)  # 获取最后一个预测的词
            tgt_input = torch.cat([tgt_input, next_token], dim=1)  # 将预测的词添加到 tgt_input 中
            if next_token.item() == end_token_id or tgt_input.size(1) >= max_text_phrase_len:
                break
        predicted_text = tokenizer.decode(tgt_input.squeeze(), skip_special_tokens=True)
        # print(f"Predicted Text: {predicted_text}")
        # predicted_text = predicted_text.split()
        # predicted_text = ' '.join(predicted_text)
        print(f"Predicted Text: {predicted_text}")


def main():
    """"""
    # 重定向输出：同时写入log.txt和终端
    sys.stdout = Tee("./output_of_real_video.log", "w")  # "w"表示覆盖写入，"a"表示追加
    # # 示例使用
    video_path = './real_video_test/test_video.mp4'
    sequence_path = f"./Data/Real_Video_Data/Keypoint_Data/saved_sequence.pt"
    # keypoint_sequences = process_video(video_path)
    # print(keypoint_sequences.shape)
    # sequence_tensor = torch.from_numpy(keypoint_sequences).float()# 将 numpy.array 转换为 tensor
    # torch.save(sequence_tensor, f'{sequence_path}')  # 保存 tensor 到本地
    # assert 0
    # 读取保存的 tensor
    sequence_tensor = torch.load(f"{sequence_path}")
    video_test(sequence_tensor)
    # 恢复原始输出（可选）
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 恢复终端输出

if __name__=="__main__":
    main()


# ./real_video_test/test_video.mp4 估计有230帧的帧数