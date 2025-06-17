from model import IntroDetectionModel
from preprocess import extract_frames_with_labels, read_labels, VideoFramesDataset, collate_fn
from train import train
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_path = 'labels.json'
video_folder_path = 'test'

labels_df = read_labels('labels.json')
dataset = VideoFramesDataset(extract_frames_with_labels(f'{video_folder_path}/', labels_df))
train_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

model = IntroDetectionModel(hidden_size=256, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, val_loader=None, optimizer=optimizer, device=device, n_epochs=10)