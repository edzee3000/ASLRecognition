import os
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import mediapipe
import matplotlib
import matplotlib.pyplot as plt
import random
import re

from skimage.transform import resize
from mediapipe.framework.formats import landmark_pb2
from tqdm.notebook import tqdm
from matplotlib import animation, rc

# 设置 matplotlib 使用交互式后端
matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg' 或其他支持交互式的后端

dataset_df = pd.read_csv(f'./Data/SimpleData/450474571.csv')
# dataset_df = pd.read_csv(f'./Data/SimpleData/5414471.csv')
# print("Full train dataset shape is {}".format(dataset_df.shape))
print("450474571.csv的形状为：{}".format(dataset_df.shape))
print(f"csv中前几行数据为：\n{dataset_df.head()}")



# Fetch sequence_id, file_id, phrase from first row
sequence_id, file_id, phrase = dataset_df.iloc[0][['sequence_id', 'file_id', 'phrase']]
print(f"sequence_id: {sequence_id}, file_id: {file_id}, phrase: {phrase}")
# Fetch data from parquet file
sample_sequence_df = pq.read_table(f"./Data/SimpleData/{str(file_id)}.parquet",
    filters=[[('sequence_id', '=', sequence_id)],]).to_pandas()
# print("Full sequence dataset shape is {}".format(sample_sequence_df.shape))
print(f"这里我们取sequence_id为：{sequence_id}")
print(f"{str(file_id)}.parquet文件中前几行数据为：\n{sample_sequence_df.head(10)}")
print(f"{str(file_id)}.parquet的形状为：{sample_sequence_df.shape}")
# print(f"{str(file_id)}.parquet所有的列名为：{sample_sequence_df.columns}")
column_names = sample_sequence_df.columns.tolist()
cleaned_names = [re.sub(r'_\d+$', '', col) for col in column_names]
unique_names = list(set(cleaned_names))
unique_names.sort()
print(f"去重后的前缀名称：{unique_names}")

# Function create animation from images.
matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
matplotlib.rcParams['savefig.pad_inches'] = 0
rc('animation', html='jshtml')
def create_animation(images):
    fig = plt.figure(figsize=(6, 9))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(images[0], cmap="gray")
    plt.close(fig)
    def animate_func(i):
        im.set_array(images[i])
        return [im]
    return animation.FuncAnimation(fig, animate_func, frames=len(images), interval=1000 / 10)
    # return animation.FuncAnimation(fig, animate_func, frames=len(images), interval=1000 / 24)


# Extract the landmark data and convert it to an image using medipipe library.
# This function extracts the data for both hands.
mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
def get_hands(seq_df):
    images = []
    all_hand_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_right_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_right_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_right_hand.*").values
        right_hand_image = np.zeros((600, 600, 3))
        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_hand, y_hand, z_hand):
            right_hand_landmarks.landmark.add(x=x, y=y, z=z)
        mp_drawing.draw_landmarks(
            right_hand_image,
            right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_left_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_left_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_left_hand.*").values
        left_hand_image = np.zeros((600, 600, 3))
        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_hand, y_hand, z_hand):
            left_hand_landmarks.landmark.add(x=x, y=y, z=z)
        mp_drawing.draw_landmarks(
            left_hand_image,
            left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        images.append([right_hand_image.astype(np.uint8), left_hand_image.astype(np.uint8)])
        all_hand_landmarks.append([right_hand_landmarks, left_hand_landmarks])
    return images, all_hand_landmarks



# Get the images created using mediapipe apis
hand_images, hand_landmarks = get_hands(sample_sequence_df)
# Fetch and show the data for right hand
anim = create_animation(np.array(hand_images)[:, 0])
anim.save("animation.mp4")
print("视频保存成功")





