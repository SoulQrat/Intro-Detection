from model import IntroDetectionModel
from preprocess import extract_frames_with_labels, read_labels, VideoFramesDataset, collate_fn
from train import train
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_path = 'labels.json'
video_folder_path = 'video'

labels_df = read_labels('labels.json')
print(len(labels_df))
dataset = VideoFramesDataset(extract_frames_with_labels(f'{video_folder_path}/', labels_df))
print(len(dataset))
train_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
print(len(train_loader))

model = IntroDetectionModel(hidden_size=16, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, None, optimizer, device, 2)