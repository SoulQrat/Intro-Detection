import os
import cv2
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from collections import defaultdict

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def read_labels(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
    data = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "file_name"})
    data[['name', 'episode']] = data['name'].str.split('.', n=1, expand=True)
    return data

def str_to_time(s: str) -> int:
    h, m, s = map(int, s.split(':'))
    return h * 3600 + m * 60 + s

def extract_frames_with_labels(video_folder: str, label_df):
    dataset = []

    for _, row in label_df.iterrows():
        file_name = row['file_name']
        video_path = os.path.join(video_folder, file_name, f"{file_name}.mp4")

        start = str_to_time(row['start'])
        end = str_to_time(row['end'])

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        for sec in tqdm(range(duration), total=duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            tensor = transform(pil_img)
            label = int(start <= sec < end)
            dataset.append((tensor, label, file_name))

        cap.release()

    return dataset

class VideoFramesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        self.grouped = defaultdict(list)
        for tensor, label, fname in data:
            self.grouped[fname].append((tensor, label))

        self.videos = list(self.grouped.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        fname = self.videos[idx]
        frames_labels = self.grouped[fname]
        
        frames = [x[0] for x in frames_labels]
        labels = [x[1] for x in frames_labels]
        
        frames_tensor = torch.stack(frames)
        labels_tensor = torch.tensor(labels)
        
        return frames_tensor, labels_tensor

def collate_fn(batch):
    max_len = max(frames.shape[0] for frames, _ in batch)
    
    batch_frames = []
    batch_labels = []

    for frames, labels in batch:
        T, C, H, W = frames.shape
        pad_len = max_len - T
        
        if pad_len > 0:
            pad_frames = torch.zeros((pad_len, C, H, W), dtype=frames.dtype)
            pad_labels = torch.zeros(pad_len, dtype=labels.dtype)
            frames = torch.cat([frames, pad_frames], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
        
        batch_frames.append(frames)
        batch_labels.append(labels)

    batch_frames = torch.stack(batch_frames)
    batch_labels = torch.stack(batch_labels)
    
    return batch_frames, batch_labels