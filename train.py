import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def compute_metrics(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    epoch_loss = 0.0

    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    precision, recall, f1 = compute_metrics(all_targets, all_preds)

    return epoch_loss / len(loader), precision, recall, f1


def test(model, loader, device, loss_fn):
    model.eval()
    epoch_loss = 0.0

    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.float().to(device)

        with torch.no_grad():
            outputs = model(inputs).squeeze(1)
            loss = loss_fn(outputs, targets)

        epoch_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    precision, recall, f1 = compute_metrics(all_targets, all_preds)

    return epoch_loss / len(loader), precision, recall, f1

def train(model, train_loader, val_loader, optimizer, device, n_epochs=10):
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        train_loss, train_precision, train_recall, train_f1 = train_epoch(model, train_loader, optimizer, device, criterion)  
        print(f"[Train] Loss: {train_loss:.4f} | Precision: {train_precision:.3f} | Recall: {train_recall:.3f} | F1: {train_f1:.3f}")

        if val_loader is not None:
            val_loss, val_precision, val_recall, val_f1 = test(model, val_loader, device, criterion)
            print(f"[Val]   Loss: {val_loss:.4f} | Precision: {val_precision:.3f} | Recall: {val_recall:.3f} | F1: {val_f1:.3f}")
