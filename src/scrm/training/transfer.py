# 源模型预训练与目标少样本适配。
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scrm.models.backbones import ActivityVocab, TimeScaler
from scrm.training.trainer import _TorchPrefixDataset, _build_torch_model, _make_collate, _model_sequence_limit, _predict_torch, _prediction_frame, _resolve_device, set_global_seed

def _training_cfg(cfg, learning_rate: float):
    return SimpleNamespace(embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, transformer_layers=cfg.transformer_layers, transformer_heads=cfg.transformer_heads, dropout=cfg.dropout, batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, learning_rate=float(learning_rate))

def _fit_epochs(model, samples, vocab, scaler, epochs: int, cfg, seed: int, device, learning_rate: float):
    import torch
    from torch.utils.data import DataLoader
    set_global_seed(seed)
    ds = _TorchPrefixDataset(samples, vocab, scaler)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(ds, batch_size=min(int(cfg.batch_size), max(1, len(ds))), shuffle=True, collate_fn=_make_collate(_model_sequence_limit(model)), generator=generator)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(cfg.weight_decay))
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

def _select_epochs_from_scratch(model_name, train_samples, val_samples, cfg, seed: int, device_str: str):
    import torch
    from torch.utils.data import DataLoader
    if not train_samples or not val_samples:
        raise ValueError('Source pretraining requires non-empty train and validation prefixes.')
    set_global_seed(seed)
    vocab = ActivityVocab().fit(train_samples)
    scaler = TimeScaler.fit(train_samples)
    tcfg = _training_cfg(cfg, cfg.pretrain_learning_rate)
    model = _build_torch_model(model_name, len(vocab), tcfg)
    device = _resolve_device(device_str)
    model = model.to(device)
    ds = _TorchPrefixDataset(train_samples, vocab, scaler)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(ds, batch_size=min(int(cfg.batch_size), max(1, len(ds))), shuffle=True, collate_fn=_make_collate(_model_sequence_limit(model)), generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.pretrain_learning_rate), weight_decay=float(cfg.weight_decay))
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    best_epoch, best_val, bad = (1, float('inf'), 0)
    for epoch in range(1, int(cfg.pretrain_max_epochs) + 1):
        model.train()
        for ids, tf, lengths, y, _ in loader:
            ids, tf, lengths, y = (ids.to(device), tf.to(device), lengths.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            pred = model(ids, tf, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pred = _predict_torch(model, val_samples, vocab, scaler, cfg.batch_size, device)
        true = np.asarray([s.remaining_hours for s in val_samples], dtype=float)
        val_mae = float(np.mean(np.abs(true - pred)))
        if val_mae < best_val - 1e-08:
            best_epoch, best_val, bad = (epoch, val_mae, 0)
        else:
            bad += 1
        if bad >= int(cfg.pretrain_patience):
            break
    return (int(best_epoch), float(best_val), vocab, scaler)

def train_source_checkpoint(model_name, source_train_samples, source_val_samples, cfg, seed: int, device_str: str='auto') -> dict:
    import torch
    best_epoch, best_val, _, _ = _select_epochs_from_scratch(model_name, source_train_samples, source_val_samples, cfg, int(seed), device_str)
    refit_seed = int(seed) + 200003
    set_global_seed(refit_seed)
    vocab = ActivityVocab().fit(source_train_samples)
    scaler = TimeScaler.fit(source_train_samples)
    tcfg = _training_cfg(cfg, cfg.pretrain_learning_rate)
    model = _build_torch_model(model_name, len(vocab), tcfg)
    device = _resolve_device(device_str)
    model = _fit_epochs(model, source_train_samples, vocab, scaler, best_epoch, cfg, refit_seed, device, cfg.pretrain_learning_rate)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {'model_name': str(model_name), 'seed': int(seed), 'best_epoch': int(best_epoch), 'source_val_mae_h': float(best_val), 'vocab': dict(vocab.token_to_id), 'scaler_mean': np.asarray(scaler.mean, dtype=np.float32), 'scaler_std': np.asarray(scaler.std, dtype=np.float32), 'state_dict': state, 'source_train_prefixes': int(len(source_train_samples)), 'source_val_prefixes': int(len(source_val_samples)), 'sequence_window_events': _model_sequence_limit(model)}

def save_source_checkpoint(checkpoint: dict, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_source_checkpoint(path: Path) -> dict:
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def _vocab_from_mapping(mapping: dict[str, int]) -> ActivityVocab:
    vocab = ActivityVocab()
    vocab.token_to_id = {str(k): int(v) for k, v in mapping.items()}
    return vocab

def extend_vocab(source_mapping: dict[str, int], samples) -> ActivityVocab:
    vocab = _vocab_from_mapping(source_mapping)
    tokens = sorted({str(t) for s in samples for t in s.activity_names})
    for token in tokens:
        if token not in vocab.token_to_id:
            vocab.token_to_id[token] = len(vocab.token_to_id)
    return vocab

def instantiate_transfer_model(checkpoint: dict, model_name: str, extra_samples, cfg, init_seed: int):
    set_global_seed(int(init_seed))
    vocab = extend_vocab(checkpoint['vocab'], extra_samples)
    scaler = TimeScaler(np.asarray(checkpoint['scaler_mean'], dtype=np.float32), np.asarray(checkpoint['scaler_std'], dtype=np.float32))
    tcfg = _training_cfg(cfg, cfg.finetune_learning_rate)
    model = _build_torch_model(model_name, len(vocab), tcfg)
    fresh = model.state_dict()
    old = checkpoint['state_dict']
    for key, target_tensor in fresh.items():
        if key == 'embedding.weight':
            src = old[key]
            n = min(src.shape[0], target_tensor.shape[0])
            target_tensor[:n].copy_(src[:n])
        elif key in old and tuple(old[key].shape) == tuple(target_tensor.shape):
            target_tensor.copy_(old[key])
    model.load_state_dict(fresh)
    return (model, vocab, scaler)

def select_finetune_epoch(checkpoint, model_name, inner_train_samples, inner_val_samples, cfg, seed: int, device_str: str='auto'):
    import torch
    from torch.utils.data import DataLoader
    if not inner_train_samples or not inner_val_samples:
        raise ValueError('Target fine-tuning requires non-empty case-level inner train/validation support.')
    model, vocab, scaler = instantiate_transfer_model(checkpoint, model_name, inner_train_samples, cfg, init_seed=int(seed) + 300007)
    device = _resolve_device(device_str)
    model = model.to(device)
    ds = _TorchPrefixDataset(inner_train_samples, vocab, scaler)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 300007)
    loader = DataLoader(ds, batch_size=min(int(cfg.batch_size), max(1, len(ds))), shuffle=True, collate_fn=_make_collate(_model_sequence_limit(model)), generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.finetune_learning_rate), weight_decay=float(cfg.weight_decay))
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    best_epoch, best_val, bad = (1, float('inf'), 0)
    for epoch in range(1, int(cfg.finetune_max_epochs) + 1):
        model.train()
        for ids, tf, lengths, y, _ in loader:
            ids, tf, lengths, y = (ids.to(device), tf.to(device), lengths.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            pred = model(ids, tf, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pred = _predict_torch(model, inner_val_samples, vocab, scaler, cfg.batch_size, device)
        true = np.asarray([s.remaining_hours for s in inner_val_samples], dtype=float)
        val_mae = float(np.mean(np.abs(true - pred)))
        if val_mae < best_val - 1e-08:
            best_epoch, best_val, bad = (epoch, val_mae, 0)
        else:
            bad += 1
        if bad >= int(cfg.finetune_patience):
            break
    return (int(best_epoch), float(best_val))

def run_transfer_finetune(checkpoint, model_name, all_support_samples, inner_train_samples, inner_val_samples, test_samples, cfg, seed: int, device_str: str='auto'):
    best_epoch, inner_val_mae = select_finetune_epoch(checkpoint, model_name, inner_train_samples, inner_val_samples, cfg, int(seed), device_str)
    final_seed = int(seed) + 400009
    model, vocab, scaler = instantiate_transfer_model(checkpoint, model_name, all_support_samples, cfg, init_seed=final_seed)
    device = _resolve_device(device_str)
    model = _fit_epochs(model, all_support_samples, vocab, scaler, best_epoch, cfg, final_seed, device, cfg.finetune_learning_rate)
    pred = _predict_torch(model, test_samples, vocab, scaler, cfg.batch_size, device)
    pred_df = _prediction_frame(test_samples, pred)
    source_vocab_size = len(checkpoint['vocab'])
    return (pred_df, {'finetune_best_epoch': int(best_epoch), 'target_inner_val_mae_h': float(inner_val_mae), 'source_pretrain_best_epoch': int(checkpoint['best_epoch']), 'source_val_mae_h': float(checkpoint['source_val_mae_h']), 'source_vocab_size': int(source_vocab_size), 'adapted_vocab_size': int(len(vocab)), 'target_new_activity_tokens': int(max(len(vocab) - source_vocab_size, 0)), 'sequence_window_events': _model_sequence_limit(model), 'device': str(device)})
