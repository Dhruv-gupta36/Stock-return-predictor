import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.logging_config import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.X[idx: idx + self.seq_len],
            self.y[idx + self.seq_len],
        )


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.batch_norm(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int = 30,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    gradient_clip: float = 1.0,
    model_dir: str = "models",
) -> LSTMPredictor:
    logger.info(f"LSTM training on device: {_DEVICE}")

    train_ds = StockDataset(X_train, y_train, seq_len)
    val_ds = StockDataset(X_val, y_val, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if len(train_loader) == 0:
        raise ValueError(
            f"Training dataset too small for seq_len={seq_len}. "
            f"Need at least {seq_len + batch_size} samples."
        )

    input_size = X_train.shape[1]
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Path(model_dir) / "lstm_best.pt"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(_DEVICE)
            y_batch = y_batch.to(_DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(_DEVICE)
                y_batch = y_batch.to(_DEVICE)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=_DEVICE))
    logger.info(f"LSTM training complete. Best val loss: {best_val_loss:.6f}")
    return model


def predict_lstm(
    model: LSTMPredictor,
    X: np.ndarray,
    seq_len: int = 30,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    dataset = StockDataset(X, np.zeros(len(X)), seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(_DEVICE)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)
