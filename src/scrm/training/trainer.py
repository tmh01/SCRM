# Transformer 训练、早停和前缀预测；参考模型仅用于元开发误差标准化。
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import numpy as np
import pandas as pd
from scrm.models.backbones import ActivityVocab, TimeScaler, TransformerRegressor

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass

def _prediction_frame(samples, pred_h: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({'case_id': [s.case_id for s in samples], 'case_key': [s.business_key for s in samples], 'process_id': [s.process_id for s in samples], 'prefix_len': [s.prefix_len for s in samples], 'elapsed_hours': [s.elapsed_hours for s in samples], 'case_duration_hours': [s.case_duration_hours for s in samples], 'y_true_hours': [s.remaining_hours for s in samples], 'y_pred_hours': np.maximum(np.asarray(pred_h, dtype=float), 0.0)})

class _TorchPrefixDataset:

    def __init__(self, samples, vocab, scaler):
        self.samples = list(samples)
        self.vocab = vocab
        self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {'ids': np.asarray(self.vocab.encode(s.activity_names), dtype=np.int64), 'time': self.scaler.transform(s.time_features), 'y_log': np.float32(np.log1p(s.remaining_hours)), 'sample_idx': idx}

def _model_sequence_limit(model) -> int | None:
    positional_embedding = getattr(model, 'pos_embedding', None)
    if positional_embedding is None:
        return None
    return int(positional_embedding.num_embeddings)

def _make_collate(max_sequence_length: int | None=None):
    import torch

    def collate(batch):
        limit = int(max_sequence_length) if max_sequence_length is not None else None
        prepared = []
        for item in batch:
            if limit is not None and limit > 0 and (len(item['ids']) > limit):
                prepared.append({**item, 'ids': item['ids'][-limit:], 'time': item['time'][-limit:]})
            else:
                prepared.append(item)
        lengths = torch.tensor([len(x['ids']) for x in prepared], dtype=torch.long)
        max_len = int(lengths.max().item())
        ids = torch.zeros((len(prepared), max_len), dtype=torch.long)
        tf = torch.zeros((len(prepared), max_len, 6), dtype=torch.float32)
        y = torch.tensor([x['y_log'] for x in prepared], dtype=torch.float32)
        idxs = []
        for i, x in enumerate(prepared):
            l = len(x['ids'])
            ids[i, :l] = torch.from_numpy(x['ids'])
            tf[i, :l] = torch.from_numpy(x['time'])
            idxs.append(x['sample_idx'])
        return (ids, tf, lengths, y, idxs)
    return collate

def _build_torch_model(model_name, vocab_size, cfg):
    if model_name == 'transformer':
        return TransformerRegressor(vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.transformer_layers, n_heads=cfg.transformer_heads, dropout=cfg.dropout)
    raise ValueError(model_name)

def _resolve_device(device: str):
    import torch
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def _predict_torch(model, samples, vocab, scaler, batch_size, device):
    import torch
    from torch.utils.data import DataLoader
    ds = _TorchPrefixDataset(samples, vocab, scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_make_collate(_model_sequence_limit(model)))
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, tf, lengths, _, _ in loader:
            ids, tf, lengths = (ids.to(device), tf.to(device), lengths.to(device))
            out = model(ids, tf, lengths)
            preds.extend(torch.expm1(out).clamp_min(0).cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)

def _fit_for_epochs(model, samples, vocab, scaler, epochs, cfg, seed, device):
    import torch
    from torch.utils.data import DataLoader
    set_global_seed(seed)
    ds = _TorchPrefixDataset(samples, vocab, scaler)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(cfg.batch_size, max(1, len(ds))), shuffle=True, collate_fn=_make_collate(_model_sequence_limit(model)), generator=generator)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    for _ in range(max(1, int(epochs))):
        model.train()
        for ids, tf, lengths, y, _ in loader:
            ids, tf, lengths, y = (ids.to(device), tf.to(device), lengths.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            pred = model(ids, tf, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model

def select_best_epoch(model_name, inner_train_samples, inner_val_samples, cfg, seed, device_str='auto') -> tuple[int, float]:
    import torch
    from torch.utils.data import DataLoader
    set_global_seed(seed)
    vocab = ActivityVocab().fit(inner_train_samples)
    scaler = TimeScaler.fit(inner_train_samples)
    model = _build_torch_model(model_name, len(vocab), cfg)
    device = _resolve_device(device_str)
    model = model.to(device)
    ds = _TorchPrefixDataset(inner_train_samples, vocab, scaler)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(cfg.batch_size, max(1, len(ds))), shuffle=True, collate_fn=_make_collate(_model_sequence_limit(model)), generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    best_epoch = 1
    best_val = float('inf')
    bad = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for ids, tf, lengths, y, _ in loader:
            ids, tf, lengths, y = (ids.to(device), tf.to(device), lengths.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            pred = model(ids, tf, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_pred = _predict_torch(model, inner_val_samples, vocab, scaler, cfg.batch_size, device)
        val_true = np.asarray([s.remaining_hours for s in inner_val_samples], dtype=float)
        val_mae = float(np.mean(np.abs(val_true - val_pred))) if len(val_true) else float('inf')
        if val_mae < best_val - 1e-08:
            best_val = val_mae
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    return (int(best_epoch), float(best_val))

def fit_target_reference(model_name, all_support_samples, inner_train_samples, inner_val_samples, test_samples, cfg, seed, device_str='auto'):
    if not inner_train_samples or not inner_val_samples:
        raise ValueError('Deep baseline requires non-empty case-level inner train/validation support.')
    best_epoch, inner_val_mae = select_best_epoch(model_name, inner_train_samples, inner_val_samples, cfg, seed, device_str=device_str)
    set_global_seed(seed + 100003)
    vocab = ActivityVocab().fit(all_support_samples)
    scaler = TimeScaler.fit(all_support_samples)
    model = _build_torch_model(model_name, len(vocab), cfg)
    device = _resolve_device(device_str)
    model = _fit_for_epochs(model, all_support_samples, vocab, scaler, best_epoch, cfg, seed + 100003, device)
    pred = _predict_torch(model, test_samples, vocab, scaler, cfg.batch_size, device)
    meta = {'best_epoch': int(best_epoch), 'inner_val_mae_h': float(inner_val_mae), 'vocab_size': int(len(vocab)), 'sequence_window_events': _model_sequence_limit(model), 'device': str(device)}
    return (_prediction_frame(test_samples, pred), meta)
