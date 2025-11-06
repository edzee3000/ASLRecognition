import os
import shutil
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from tqdm import tqdm
import editdistance  # 用于计算编辑距离，需安装：pip install editdistance
import torch.nn.functional as F


################  2. 定义常量和特征索引   ################
dataset_df = pd.read_csv('./Data/asl-fingerspelling/train.csv')
preprocess_data_dir = f"./Data/PreprocessedPytorch"
# 关键点定义（与原TF代码保持一致）
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
# 坐标特征列名（假设FEATURE_COLUMNS已在别处定义）
X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
FEATURE_COLUMNS = X + Y + Z
# 特征索引（根据FEATURE_COLUMNS筛选）
X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]
RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]
# 其他常量
FRAME_LEN = 128  # 帧长度   原本我定义是250，但是实在是太大了  我觉得没有必要，所以改成128吧还是
TEXT_LEN = 64
PAD_TOKEN = 'P'
START_TOKEN = '<'
END_TOKEN = '>'
PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61
batch_size = 64
learning_rate = 1e-7
num_epochs = 100
num_hid = 200
num_head = 4
num_feed_forward = 400
source_maxlen = FRAME_LEN
target_maxlen = TEXT_LEN
num_layers_enc = 2
num_layers_dec = 1
num_classes=62
# 加载字符映射表
with open("./Data/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)
# 添加特殊符号
char_to_num[PAD_TOKEN] = PAD_TOKEN_IDX
char_to_num[START_TOKEN] = START_TOKEN_IDX
char_to_num[END_TOKEN] = END_TOKEN_IDX
num_to_char = {v: k for k, v in char_to_num.items()}



def DataPreProcess():
    ################  3. 数据预处理（转换为 PyTorch 可用格式）  ################
    # 创建预处理目录
    if not os.path.isdir(f"{preprocess_data_dir}"):
        os.mkdir(f"{preprocess_data_dir}")
    else:
        shutil.rmtree(f"{preprocess_data_dir}")
        os.mkdir(f"{preprocess_data_dir}")
    os.mkdir(f"{preprocess_data_dir}/train_landmarks")
    # 预处理并保存数据
    metadata = []
    for file_id in tqdm(dataset_df.file_id.unique()):
        # 读取Parquet文件
        parquet_df = pq.read_table(
            f"./Data/asl-fingerspelling/train_landmarks/{file_id}.parquet",
            columns=['sequence_id'] + FEATURE_COLUMNS
        ).to_pandas()
        # print(parquet_df.shape)
        # assert 0
        parquet_numpy = parquet_df.to_numpy()
        # 处理每个文件的所有序列
        file_df = dataset_df[dataset_df["file_id"] == file_id]
        # 存储当前文件的所有有效序列
        file_sequences = []
        file_phrases = []
        for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
            # 提取帧数据
            frames = parquet_numpy[parquet_df.index == seq_id]
            # 过滤低质量序列          ###### 过滤低质量的序列之后还需要专门写一个函数提升已有的数据的质量   ####
            r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
            l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
            no_nan = max(r_nonan, l_nonan)
            if 2 * len(phrase) < no_nan:
                file_sequences.append(frames)
                file_phrases.append(phrase)
        # 如果文件中有有效序列，则保存
        if file_sequences:
            # 保存整个文件的数据
            file_data = {
                'sequences': file_sequences,
                'phrases': file_phrases
            }
            np.save(f"{preprocess_data_dir}/train_landmarks/{file_id}.npy", file_data)
            # 更新元数据
            metadata.append({
                "file_id": file_id,
                "num_sequences": len(file_sequences),
                "phrases": file_phrases
            })
    # 保存元数据
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(f"{preprocess_data_dir}/train_landmarks_metadata.csv", index=False)





###################  4. 定义 PyTorch 数据集和数据加载器   #######################
class ASLDataset(Dataset):
    def __init__(self, metadata, data_dir, char_to_num, frame_len=FRAME_LEN, max_len = TEXT_LEN):
        self.metadata = metadata
        self.data_dir = data_dir
        self.char_to_num = char_to_num
        self.frame_len = frame_len
        self.max_len = max_len

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_id = self.metadata.iloc[idx]['file_id']
        file_data = np.load(f"{self.data_dir}/{file_id}.npy", allow_pickle=True).item()
        seq_index = idx % len(file_data['sequences'])
        frames = file_data['sequences'][seq_index]
        frames = torch.tensor(frames, dtype=torch.float32)
        # print(frames.shape)
        # Add start and end pointers to phrase. 添加开头以及结束的 Token
        phrase = file_data['phrases'][seq_index]
        phrase = START_TOKEN + phrase + END_TOKEN
        # print(phrase)
        phrase_ids = [self.char_to_num[c] for c in phrase]
        phrase_ids = torch.tensor(phrase_ids, dtype=torch.long)
        phrase_ids = F.pad(phrase_ids, (0, self.max_len - len(phrase)), value=PAD_TOKEN_IDX)
        frames = self.pre_process(frames)
        # print(frames.shape)
        # assert 0
        return frames, phrase_ids

    def pre_process(self, x):
        # 1. 提取 hand & pose
        rhand = x[:, RHAND_IDX]
        lhand = x[:, LHAND_IDX]
        rpose = x[:, RPOSE_IDX]
        lpose = x[:, LPOSE_IDX]
        # 2. 计算 NaN 统计
        rnans = torch.isnan(rhand).any(dim=1).sum().item()
        lnans = torch.isnan(lhand).any(dim=1).sum().item()
        # 3. 选择主导手
        if rnans > lnans:
            hand = lhand
            pose = lpose
            hand_x, hand_y, hand_z = torch.chunk(hand, 3, dim=1)
            hand = torch.cat([1 - hand_x, hand_y, hand_z], dim=1)
            pose_x, pose_y, pose_z = torch.chunk(pose, 3, dim=1)
            pose = torch.cat([1 - pose_x, pose_y, pose_z], dim=1)
        else:
            hand = rhand
            pose = rpose
            hand_x, hand_y, hand_z = torch.chunk(hand, 3, dim=1)
            pose_x, pose_y, pose_z = torch.chunk(pose, 3, dim=1)
        # 4. reshape 成 (T, 3, N) 并标准化
        hand = torch.stack([hand_x, hand_y, hand_z], dim=2)  # (T, N//3, 3)
        hand = (hand - hand.mean(dim=0, keepdim=True)) / (hand.std(dim=0, keepdim=True) + 1e-8)
        pose = torch.stack([pose_x, pose_y, pose_z], dim=2)
        pose = (pose - pose.mean(dim=0, keepdim=True)) / (pose.std(dim=0, keepdim=True) + 1e-8)
        # 5. 合并并 resize/pad
        x = torch.cat([hand.flatten(1), pose.flatten(1)], dim=1)  # (T, C)
        x = torch.nan_to_num(x, nan=0.0)  # 替换 NaN
        x = self.resize_pad_torch(x)  # (FRAME_LEN, C)
        return x

    def resize_pad_torch(self, x):
        T, C = x.shape
        if T < FRAME_LEN:
            pad_len = FRAME_LEN - T
            x = F.pad(x, (0, 0, 0, pad_len))  # (T, C) -> (T+pad_len, C)
        else:
            # 线性插值到固定帧数
            x = x.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
            x = F.interpolate(x, size=FRAME_LEN, mode='linear', align_corners=False)
            x = x.squeeze(0).permute(1, 0)  # (FRAME_LEN, C)
        return x

# 数据加载器
def collate_fn(batch):
    """处理批量数据，对标签进行填充"""
    frames, phrases = zip(*batch)
    frames = torch.stack(frames, dim=0)  # (batch_size, FRAME_LEN, 特征维度)
    # 对标签进行填充
    max_len = max([p.shape[0] for p in phrases])
    phrases_padded = []
    for p in phrases:
        pad_len = max_len - p.shape[0]
        p_padded = torch.cat([p, torch.full((pad_len,), PAD_TOKEN_IDX, dtype=torch.long)], dim=0)
        phrases_padded.append(p_padded)
    phrases_padded = torch.stack(phrases_padded, dim=0)  # (batch_size, max_len)
    return frames, phrases_padded


# 划分训练集和验证集
metadata_df = pd.read_csv(f"{preprocess_data_dir}/train_landmarks_metadata.csv")
train_len = int(0.8 * len(metadata_df))
train_metadata = metadata_df[:train_len]
valid_metadata = metadata_df[train_len:]

train_dataset = ASLDataset(
    metadata=train_metadata,
    data_dir=f"{preprocess_data_dir}/train_landmarks",
    char_to_num=char_to_num
)
valid_dataset = ASLDataset(
    metadata=valid_metadata,
    data_dir=f"{preprocess_data_dir}/train_landmarks",
    char_to_num=char_to_num
)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)




























###################  5. 定义 Transformer 模型（PyTorch 版本）  #######################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class LandmarkEmbedding(nn.Module):
    def __init__(self, input_dim, num_hid, max_len=FRAME_LEN):
        # 1D的卷积在这里用于提取时序特征，通过滑动窗口（卷积核）对时间维度上的连续帧进行特征聚合，捕捉手势的动态变化模式。
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_hid, kernel_size=11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(num_hid, num_hid, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(num_hid, num_hid, kernel_size=11, stride=2, padding=5)
        self.relu = nn.ReLU()
        # self.pos_encoding = PositionalEncoding(num_hid, max_len=FRAME_LEN)
        self.pos_encoding = nn.Embedding(max_len, num_hid)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(f"在LandmarkEmbedding中permute之后x的形状为: {x.shape}")  #  [batch_size, input_dim, frame_idx]
        x = self.relu(self.conv1(x))
        # print(f"经过conv1后的x形状为: {x.shape}")    # [batch_size, num_hidden, frame_idx//2]
        x = self.relu(self.conv2(x))
        # print(f"经过conv2后的x形状为: {x.shape}")     # [batch_size, num_hidden, frame_idx//4]
        x = self.conv3(x)
        # print(f"经过conv3后的x形状为: {x.shape}")     # [batch_size, num_hidden, frame_idx//8]
        # x = x.permute(2, 0, 1)
        # print(f"再次permute之后的x形状为: {x.shape}")  # [frame_idx//8, batch_size, num_hidden]
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        # 添加位置编码
        positions = torch.arange(0, x.size(1), device=x.device)
        x = x + self.pos_encoding(positions)
        # print(f"经过了position位置编码之后x形状为: {x.shape}")
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, num_hid=64, max_len=TEXT_LEN):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, num_hid)
        # self.pos_encoding = PositionalEncoding(num_hid, max_len=max_len)
        self.pos_encoding = nn.Embedding(max_len, num_hid)

    def forward(self, x):
        # print(f"在TokenEmbedding中输入的x的形状为: {x.shape}")
        # x = self.embedding(x)
        # print(f"经过embedding之后x形状为: {x.shape}")
        # x = x.permute(1, 0, 2)
        # print(f"经过permute之后x形状为: {x.shape}")
        # x = self.pos_encoding(x)
        # print(f"经过位置编码之后x的形状为: {x.shape}")
        positions = torch.arange(0, x.size(1), device=x.device)
        return self.embedding(x) + self.pos_encoding(positions)



# 定义多层感知机类
class MLP(nn.Module):
    def __init__(self, input_dim, num_hid):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hid*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hid*2, num_hid*2)
        self.fc3 = nn.Linear(num_hid * 2, num_hid)
        self.pos_encoding = PositionalEncoding(num_hid, max_len=FRAME_LEN)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoding(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_hid=200, num_head=4, num_feed_forward=400,
                 source_maxlen=128, target_maxlen=64, num_layers_enc=2,
                 num_layers_dec=1, num_classes=62):
        super().__init__()
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        input_dim = len(LHAND_IDX) + len(LPOSE_IDX)
        # encoding 部分
        self.enc_embedding = LandmarkEmbedding(input_dim, num_hid)  # 此处num_hid是卷积层的卷积核的个数
        # self.enc_embedding = MLP(input_dim, num_hid)
        encoder_layers = (
            nn.TransformerEncoderLayer(
            d_model=num_hid, nhead=num_head, dim_feedforward=num_feed_forward,
            dropout=0, batch_first=True
        )) # batch_first=False（默认设置）输入形状：[seq_len, batch_size, embed_dim] seq_len：序列长度（例如时间步数或句子长度）。 batch_size：批次大小。 embed_dim：特征维度（模型的隐藏维度）。
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers_enc)
        # decoding 部分
        self.dec_embedding = TokenEmbedding(num_classes, num_hid, target_maxlen)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=num_hid, nhead=num_head, dim_feedforward=num_feed_forward,
            dropout=0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers_dec)
        self.classifier = nn.Linear(num_hid, num_classes)
        self.causal_mask = self._generate_causal_mask(target_maxlen)

    def _generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # print(f"src的形状为: {src.shape}")
        src_emb = self.enc_embedding(src)
        # print(f"src_emb形状为: {src_emb.shape}")
        enc_out = self.encoder(src_emb)
        # print(f"enc_out形状为: {enc_out.shape}")
        tgt_emb = self.dec_embedding(tgt)
        # print(f"tgt_emb形状为: {tgt_emb.shape}")
        batch_size = tgt.size(0)
        # print(f"batch_size为: {batch_size}")
        tgt_len = tgt.size(1)
        # print(f"tgt_len为: {tgt_len}")
        causal_mask = self.causal_mask[:tgt_len, :tgt_len].to(tgt.device)
        # print(f"causal_mask形状为: {causal_mask.shape}")
        # assert 0
        tgt_pad_mask = (tgt == PAD_TOKEN_IDX).bool().to(tgt.device)
        dec_out = self.decoder(
            tgt_emb,
            enc_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        output = self.classifier(dec_out)
        output = output.permute(1, 0, 2)
        return output

    @torch.no_grad()
    def generate(self, src, max_len=None, start_token_idx=START_TOKEN_IDX, end_token_idx=END_TOKEN_IDX):
        if max_len is None:
            max_len = self.target_maxlen
        batch_size = src.size(0)
        src_emb = self.enc_embedding(src)
        enc_out = self.encoder(src_emb)
        ys = torch.ones(batch_size, 1, dtype=torch.long).fill_(start_token_idx).to(src.device)
        for i in range(max_len - 1):
            tgt_pad_mask = (ys == PAD_TOKEN_IDX).bool().to(src.device)
            tgt_emb = self.dec_embedding(ys)
            dec_out = self.decoder(
                tgt_emb,
                enc_out,
                tgt_mask=self.causal_mask[:ys.size(1), :ys.size(1)].to(src.device),
                tgt_key_padding_mask=tgt_pad_mask
            )
            output = self.classifier(dec_out[:, -1, :])
            # print(output)
            _, next_word = torch.max(output, dim=1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if torch.all((ys == end_token_idx).any(dim=1)):
                break
        return ys





###################  6. 训练和评估函数  #######################
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_edit_dist = 0
    total_samples = 0
    for frames, phrases in tqdm(train_loader):
        frames = frames.to(device)
        phrases = phrases.to(device)
        tgt_input = phrases[:, :-1]
        tgt_output = phrases[:, 1:]
        # print(f"frames的形状为: {frames.shape}")
        # print(f"tgt_input形状为: {tgt_input.shape}")
        # print(f"tgt_output形状为: {tgt_output.shape}")
        optimizer.zero_grad()
        output = model(frames, tgt_input)
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        preds = output.argmax(dim=2)   #这里是取概率最大的
        for pred, target in zip(preds, tgt_output):
            pred_list = pred.tolist()
            target_list = target.tolist()
            if END_TOKEN_IDX in pred_list:
                pred_list = pred_list[:pred_list.index(END_TOKEN_IDX) + 1]
            if END_TOKEN_IDX in target_list:
                target_list = target_list[:target_list.index(END_TOKEN_IDX) + 1]
            edit_dist = editdistance.eval(pred_list, target_list)
            total_edit_dist += edit_dist
        total_loss += loss.item() * frames.size(0)
        total_samples += frames.size(0)
    avg_loss = total_loss / total_samples
    avg_edit_dist = total_edit_dist / total_samples
    return avg_loss, avg_edit_dist


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_edit_dist = 0
    total_samples = 0

    with torch.no_grad():
        for frames, phrases in val_loader:
            frames = frames.to(device)
            phrases = phrases.to(device)
            tgt_input = phrases[:, :-1]
            tgt_output = phrases[:, 1:]
            output = model(frames, tgt_input)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            preds = output.argmax(dim=2)
            for pred, target in zip(preds, tgt_output):
                pred_list = pred.tolist()
                target_list = target.tolist()
                if END_TOKEN_IDX in pred_list:
                    pred_list = pred_list[:pred_list.index(END_TOKEN_IDX) + 1]
                if END_TOKEN_IDX in target_list:
                    target_list = target_list[:target_list.index(END_TOKEN_IDX) + 1]

                edit_dist = editdistance.eval(pred_list, target_list)
                total_edit_dist += edit_dist

            total_loss += loss.item() * frames.size(0)
            total_samples += frames.size(0)

    avg_loss = total_loss / total_samples
    avg_edit_dist = total_edit_dist / total_samples
    return avg_loss, avg_edit_dist


def display_predictions(model, data_loader, num_examples=5, device='cpu'):
    model.eval()
    examples_shown = 0

    with torch.no_grad():
        for frames, phrases in data_loader:
            frames = frames.to(device)
            preds = model.generate(frames)
            for i in range(min(frames.size(0), num_examples - examples_shown)):
                pred_chars = []
                for token_id in preds[i].tolist():
                    if token_id == END_TOKEN_IDX:
                        break
                    pred_chars.append(num_to_char[token_id])
                pred_str = ''.join(pred_chars[1:])
                target_chars = []
                for token_id in phrases[i].tolist():
                    if token_id == END_TOKEN_IDX:
                        break
                    target_chars.append(num_to_char[token_id])
                target_str = ''.join(target_chars[1:])
                print(f"Target:     {target_str}")
                print(f"Prediction: {pred_str}")
                print("-" * 50)
                examples_shown += 1
                if examples_shown >= num_examples:
                    return





###################  7. 主训练循环  #######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = TransformerModel(
    # num_hid=1024,
    # num_head=8,
    # num_feed_forward=512,
    # source_maxlen=FRAME_LEN,
    # target_maxlen=64,
    # num_layers_enc=6,
    # num_layers_dec=5,
    # num_classes=62
    num_hid=num_hid,
    num_head=num_head,
    num_feed_forward=num_feed_forward,
    source_maxlen = source_maxlen,
    target_maxlen = target_maxlen,
    num_layers_enc=num_layers_enc,
    num_layers_dec=num_layers_dec,
    num_classes=num_classes
).to(device)
print(f"num_hid: {num_hid}")
print(f"num_head: {num_head}")
print(f"num_feed_forward: {num_feed_forward}")
print(f"source_maxlen: {source_maxlen}")
print(f"target_maxlen: {target_maxlen}")
print(f"num_layers_enc: {num_layers_enc}")
print(f"num_layers_dec: {num_layers_dec}")
print(f"num_classes: {num_classes}")


criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


best_val_edit_dist = float('inf')

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 50)
    train_loss, train_edit_dist = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Edit Distance: {train_edit_dist:.4f}")
    val_loss, val_edit_dist = evaluate(model, valid_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f} | Val Edit Distance: {val_edit_dist:.4f}")
    if val_edit_dist < best_val_edit_dist:
        best_val_edit_dist = val_edit_dist
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
    if (epoch + 1) % 1 == 0:
        print("\nSample Predictions:")
        display_predictions(model, valid_loader, num_examples=5, device=device)
    print()

print("Training complete!")

















# rhand = x[:, RHAND_IDX]
#         lhand = x[:, LHAND_IDX]
#         rpose = x[:, RPOSE_IDX]
#         lpose = x[:, LPOSE_IDX]
#         rnans = torch.sum(torch.any(torch.isnan(rhand), dim=1)).item()
#         lnans = torch.sum(torch.any(torch.isnan(lhand), dim=1)).item()
#         # 判断哪个是优势手
#         if rnans > lnans:
#             hand = lhand
#             pose = lpose
#             hand_x = hand[:, :len(LHAND_IDX) // 3]
#             hand_y = hand[:, len(LHAND_IDX) // 3: 2 * len(LHAND_IDX) // 3]
#             hand_z = hand[:, 2 * len(LHAND_IDX) // 3:]
#             hand = torch.cat([1 - hand_x, hand_y, hand_z], dim=1)
#             pose_x = pose[:, :len(LPOSE_IDX) // 3]
#             pose_y = pose[:, len(LPOSE_IDX) // 3: 2 * len(LPOSE_IDX) // 3]
#             pose_z = pose[:, 2 * len(LPOSE_IDX) // 3:]
#             pose = torch.cat([1 - pose_x, pose_y, pose_z], dim=1)
#         else:
#             hand = rhand
#             pose = rpose
#             hand_x = hand[:, :len(LHAND_IDX) // 3]
#             hand_y = hand[:, len(LHAND_IDX) // 3: 2 * len(LHAND_IDX) // 3]
#             hand_z = hand[:, 2 * len(LHAND_IDX) // 3:]
#             hand = torch.cat([hand_x, hand_y, hand_z], dim=1)
#             pose_x = pose[:, :len(LPOSE_IDX) // 3]
#             pose_y = pose[:, len(LPOSE_IDX) // 3: 2 * len(LPOSE_IDX) // 3]
#             pose_z = pose[:, 2 * len(LPOSE_IDX) // 3:]
#             pose = torch.cat([pose_x, pose_y, pose_z], dim=1)
#         hand = hand.view(hand.shape[0], -1, 3)
#         mean = torch.mean(hand, dim=1, keepdim=True)
#         std = torch.std(hand, dim=1, keepdim=True)
#         hand = (hand - mean) / (std + 1e-4)
#         pose = pose.view(pose.shape[0], -1, 3)
#         x = torch.cat([hand, pose], dim=1)
#         # # print(x.shape)
#         # # reshape重塑 frame 的长度
#         x = self.resize_pad(x)
#         # if x.shape[0] < self.frame_len:
#         #     pad_len = self.frame_len - x.shape[0]
#         #     x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
#         # else:
#         #     x = x[:self.frame_len]
#         x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
#         x = x.view(self.frame_len, -1)
#         return x