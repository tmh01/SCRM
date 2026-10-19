# 公开事件日志读取、案例划分、前缀与六视图特征。
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scrm.data.records import PrefixSample
SUPPORTED_SUFFIXES = ('.csv', '.xes', '.xes.gz')

@dataclass(frozen=True)
class PublicSchema:
    case_col: str
    activity_col: str
    timestamp_col: str
    resource_col: str | None = None
    lifecycle_col: str | None = None

@dataclass(frozen=True)
class PublicLogSpec:
    process_name: str
    path: Path
    family: str
    schema: PublicSchema | None = None
    terminal_activities: tuple[str, ...] = ()

def _norm_col(x: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(x).strip().lower())
CASE_CANDIDATES = ('case:concept:name', 'case_id', 'case id', 'caseid', 'case', 'trace_id', 'trace id', 'traceid', 'case concept name', 'case:case', 'case_identifier', 'caseidentifier', 'casekey', 'case_key')
ACTIVITY_CANDIDATES = ('concept:name', 'activity', 'activity_name', 'activity name', 'activityid', 'activity_id', 'event', 'event_name', 'task', 'task_name', 'eventclass', 'event_class')
TIMESTAMP_CANDIDATES = ('time:timestamp', 'timestamp', 'complete timestamp', 'complete_timestamp', 'event_time', 'event time', 'datetime', 'date_time', 'time', 'start_timestamp', 'end_timestamp', 'completion_time')
RESOURCE_CANDIDATES = ('org:resource', 'resource', 'resource_name', 'resource name', 'user', 'performer', 'actor', 'org:role', 'role')
LIFECYCLE_CANDIDATES = ('lifecycle:transition', 'lifecycle', 'transition', 'lifecycle_transition', 'event_type', 'eventtype')

def _pick_column(columns: Sequence[str], candidates: Sequence[str], required: bool, role: str) -> str | None:
    by_norm: dict[str, list[str]] = {}
    for c in columns:
        by_norm.setdefault(_norm_col(c), []).append(str(c))
    for cand in candidates:
        key = _norm_col(cand)
        if key in by_norm:
            return by_norm[key][0]
    patterns = {'case': ('case', 'trace'), 'activity': ('activity', 'event', 'task'), 'timestamp': ('timestamp', 'datetime', 'time'), 'resource': ('resource', 'performer', 'user', 'actor'), 'lifecycle': ('lifecycle', 'transition')}.get(role, ())
    for c in columns:
        n = _norm_col(c)
        if any((p in n for p in patterns)):
            return str(c)
    if required:
        raise ValueError(f'Could not detect required public-log {role} column from: {list(columns)}')
    return None

def detect_schema(columns: Sequence[str]) -> PublicSchema:
    cols = [str(c) for c in columns]
    return PublicSchema(case_col=str(_pick_column(cols, CASE_CANDIDATES, True, 'case')), activity_col=str(_pick_column(cols, ACTIVITY_CANDIDATES, True, 'activity')), timestamp_col=str(_pick_column(cols, TIMESTAMP_CANDIDATES, True, 'timestamp')), resource_col=_pick_column(cols, RESOURCE_CANDIDATES, False, 'resource'), lifecycle_col=_pick_column(cols, LIFECYCLE_CANDIDATES, False, 'lifecycle'))

def _read_csv_flexible(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin1'):
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding=enc)
        except Exception as e:
            errors.append(f'{enc}: {type(e).__name__}: {e}')
    raise RuntimeError(f"Unable to read CSV {path}: {' | '.join(errors)}")

def _strip_ns(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]

def _xes_attr_value(el: ET.Element):
    tag = _strip_ns(el.tag)
    value = el.attrib.get('value')
    if tag in {'int', 'long'}:
        try:
            return int(value) if value is not None else None
        except Exception:
            return value
    if tag in {'float', 'double'}:
        try:
            return float(value) if value is not None else None
        except Exception:
            return value
    if tag == 'boolean':
        return str(value).lower() == 'true'
    return value

def _read_xes(path: Path) -> pd.DataFrame:
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    rows: list[dict] = []
    with opener(path, 'rb') as fh:
        context = ET.iterparse(fh, events=('end',))
        for _, elem in context:
            if _strip_ns(elem.tag) != 'trace':
                continue
            trace_attrs: dict[str, object] = {}
            children = list(elem)
            for child in children:
                if _strip_ns(child.tag) == 'event':
                    continue
                key = child.attrib.get('key')
                if key:
                    trace_attrs[str(key)] = _xes_attr_value(child)
            case_id = trace_attrs.get('concept:name') or trace_attrs.get('case:concept:name')
            for event_pos, event in enumerate([x for x in children if _strip_ns(x.tag) == 'event']):
                r = {'__trace_case_id': case_id, '__event_pos': event_pos}
                for attr in list(event):
                    key = attr.attrib.get('key')
                    if key:
                        r[str(key)] = _xes_attr_value(attr)
                rows.append(r)
            elem.clear()
    if not rows:
        raise ValueError(f'No traces/events parsed from XES: {path}')
    df = pd.DataFrame(rows)
    if 'case:concept:name' not in df.columns and '__trace_case_id' in df.columns:
        df['case:concept:name'] = df['__trace_case_id']
    return df

def read_raw_public_log(path: Path) -> pd.DataFrame:
    low = str(path).lower()
    if low.endswith('.csv'):
        return _read_csv_flexible(path)
    if low.endswith('.xes') or low.endswith('.xes.gz'):
        return _read_xes(path)
    raise ValueError(f'Unsupported public event log: {path}')

def truncate_at_last_terminal_event(raw: pd.DataFrame, terminal_activities: Sequence[str], schema: PublicSchema | None=None) -> tuple[pd.DataFrame, dict]:
    """仅保留存在终止活动的案例，并在最后一次终止事件处截断。

    某些服务台日志在 ``Closed`` 之后仍记录关系更新或质量标记。
    直接要求最后一行必须是 ``Closed`` 会错误丢弃这些已完成案例，因此这里
    选择每个案例最后一次终止活动作为剩余时间终点，并删除其后的事件。
    """
    resolved = schema or detect_schema(raw.columns)
    allowed = {str(value).strip().casefold() for value in terminal_activities if str(value).strip()}
    if not allowed:
        return (raw.copy(), {'completion_filter': False, 'terminal_activities': [], 'excluded_without_terminal_cases': 0, 'events_after_terminal_removed': 0})
    view = pd.DataFrame({'_row': np.arange(len(raw), dtype=np.int64), '_case': raw[resolved.case_col].astype(str).str.strip(), '_activity': raw[resolved.activity_col].astype(str).str.strip(), '_timestamp': _to_naive_datetime(raw[resolved.timestamp_col])})
    valid_case = ~view['_case'].str.casefold().isin({'', 'nan', 'none', 'null'})
    valid_time = view['_timestamp'].notna()
    candidates = view[valid_case & valid_time & view['_activity'].str.casefold().isin(allowed)].sort_values(['_case', '_timestamp', '_row'], kind='mergesort')
    terminal = candidates.groupby('_case', sort=False).tail(1).set_index('_case')
    case_count = int(view.loc[valid_case, '_case'].nunique())
    if terminal.empty:
        raise ValueError(f'No case contains a configured terminal activity: {sorted(terminal_activities)}')
    end_time = view['_case'].map(terminal['_timestamp'])
    end_order = view['_case'].map(terminal['_row'])
    has_terminal = end_time.notna()
    at_or_before_terminal = (view['_timestamp'] < end_time) | (view['_timestamp'] == end_time) & (view['_row'] <= end_order)
    keep = has_terminal & at_or_before_terminal
    return (raw.loc[keep.to_numpy()].copy(), {'completion_filter': True, 'terminal_activities': sorted(map(str, terminal_activities)), 'excluded_without_terminal_cases': int(case_count - len(terminal)), 'events_after_terminal_removed': int((has_terminal & ~at_or_before_terminal).sum())})

def _to_naive_datetime(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, errors='coerce', utc=True)
    try:
        return x.dt.tz_convert(None)
    except Exception:
        try:
            return x.dt.tz_localize(None)
        except Exception:
            return pd.to_datetime(s, errors='coerce')

def _stable_process_int(name: str) -> int:
    h = hashlib.sha1(str(name).encode('utf-8')).hexdigest()[:8]
    return (int(h, 16) & 2147483647) % 900000000

def normalize_public_log(raw: pd.DataFrame, process_name: str, schema: PublicSchema | None=None, activity_lifecycle: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    schema = schema or detect_schema(raw.columns)
    x = raw.copy()
    case = x[schema.case_col].astype(str).str.strip()
    act = x[schema.activity_col].astype(str).str.strip()
    ts = _to_naive_datetime(x[schema.timestamp_col])
    res = x[schema.resource_col].astype(str).str.strip() if schema.resource_col and schema.resource_col in x else pd.Series('', index=x.index)
    life = x[schema.lifecycle_col].astype(str).str.strip() if schema.lifecycle_col and schema.lifecycle_col in x else pd.Series('', index=x.index)
    bad_case = case.str.lower().isin({'', 'nan', 'none', 'null'})
    bad_act = act.str.lower().isin({'', 'nan', 'none', 'null'})
    keep = ~bad_case & ~bad_act & ts.notna()
    kept_idx = np.flatnonzero(keep.to_numpy())
    case, act, ts, res, life = (case[keep], act[keep], ts[keep], res[keep], life[keep])
    if activity_lifecycle:
        valid_life = ~life.str.lower().isin({'', 'nan', 'none', 'null'})
        labels = act.where(~valid_life, act + '::' + life)
    else:
        labels = act
    e = pd.DataFrame({'raw_case_id': case.to_numpy(), 'activity_name': labels.astype(str).to_numpy(), 'timestamp': ts.to_numpy(), 'resource': res.astype(str).to_numpy(), 'lifecycle': life.astype(str).to_numpy(), '_raw_order': kept_idx.astype(np.int64)})
    unique_cases = sorted(e['raw_case_id'].astype(str).unique().tolist())
    case_map = {c: i + 1 for i, c in enumerate(unique_cases)}
    process_int = _stable_process_int(process_name)
    e['local_case_id'] = e['raw_case_id'].map(case_map).astype(np.int64)
    e['case_uid'] = np.int64(process_int) * np.int64(10000000) + e['local_case_id'].astype(np.int64)
    e['process_name'] = str(process_name)
    e['process_id'] = int(process_int)
    e = e.sort_values(['case_uid', 'timestamp', '_raw_order'], kind='mergesort').reset_index(drop=True)
    e['event_order'] = e.groupby('case_uid', sort=False).cumcount().astype(int)
    c = e.groupby('case_uid', sort=False).agg(process_name=('process_name', 'first'), process_id=('process_id', 'first'), raw_case_id=('raw_case_id', 'first'), case_start=('timestamp', 'min'), case_end=('timestamp', 'max'), trace_length=('activity_name', 'size')).reset_index()
    c['duration_hours'] = (c['case_end'] - c['case_start']).dt.total_seconds() / 3600.0
    c['usable_rtp'] = (c['trace_length'] >= 2) & c['duration_hours'].notna() & (c['duration_hours'] > 0)
    valid_ids = set(c.loc[c['usable_rtp'], 'case_uid'].astype(np.int64))
    e['usable_rtp'] = e['case_uid'].isin(valid_ids)
    audit = {'process_name': str(process_name), 'raw_rows': int(len(raw)), 'parsed_events': int(len(e)), 'raw_cases': int(len(unique_cases)), 'usable_cases': int(c['usable_rtp'].sum()), 'activities': int(e.loc[e['usable_rtp'], 'activity_name'].nunique()), 'resources': int(e.loc[e['usable_rtp'] & ~e['resource'].astype(str).str.lower().isin({'', 'nan', 'null', 'none'}), 'resource'].nunique()), 'schema_case_col': schema.case_col, 'schema_activity_col': schema.activity_col, 'schema_timestamp_col': schema.timestamp_col, 'schema_resource_col': schema.resource_col or '', 'schema_lifecycle_col': schema.lifecycle_col or ''}
    return (e, c, audit)

def chronological_case_split(cases: pd.DataFrame, train_ratio: float=0.7, val_ratio: float=0.1) -> pd.DataFrame:
    c = cases[cases['usable_rtp']].copy().sort_values(['case_start', 'case_uid'], kind='mergesort')
    n = len(c)
    if n < 3:
        raise ValueError('Need at least 3 usable cases for train/val/test split.')
    n_train = max(1, int(math.floor(n * float(train_ratio))))
    n_val = max(1, int(math.floor(n * float(val_ratio))))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    labels = np.array(['test'] * n, dtype=object)
    labels[:n_train] = 'train'
    labels[n_train:n_train + n_val] = 'val'
    out = c[['case_uid', 'process_name', 'raw_case_id', 'case_start']].copy()
    out['split'] = labels
    return out

def select_support_case_ids(split: pd.DataFrame, process_name: str, seed: int, k: int) -> list[int]:
    pool = split[(split['process_name'].astype(str) == str(process_name)) & (split['split'] == 'train')]['case_uid'].astype(np.int64).to_numpy()
    pool = np.sort(np.unique(pool))
    if len(pool) < int(k):
        return []
    rng = np.random.default_rng(int(seed))
    arr = pool.copy()
    rng.shuffle(arr)
    return [int(x) for x in arr[:int(k)]]

def split_support_case_ids_generic(case_ids: Sequence[int], seed: int, val_ratio: float=0.2) -> tuple[list[int], list[int]]:
    ids = np.asarray(sorted(set((int(x) for x in case_ids))), dtype=np.int64)
    if len(ids) < 2:
        return (ids.tolist(), [])
    rng = np.random.default_rng(int(seed) + 991)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(val_ratio))))
    n_val = min(n_val, len(ids) - 1)
    return ([int(x) for x in ids[n_val:]], [int(x) for x in ids[:n_val]])

def _event_time_features(times: list[pd.Timestamp], case_start: pd.Timestamp) -> np.ndarray:
    out = []
    prev = case_start
    for t in times:
        elapsed = max((t - case_start).total_seconds() / 3600.0, 0.0)
        delta = max((t - prev).total_seconds() / 3600.0, 0.0)
        hour = t.hour + t.minute / 60.0
        dow = t.dayofweek
        out.append([np.log1p(elapsed), np.log1p(delta), np.sin(2.0 * np.pi * hour / 24.0), np.cos(2.0 * np.pi * hour / 24.0), np.sin(2.0 * np.pi * dow / 7.0), np.cos(2.0 * np.pi * dow / 7.0)])
        prev = t
    return np.asarray(out, dtype=np.float32)

def build_public_prefix_samples(events: pd.DataFrame, cases: pd.DataFrame, case_ids: Iterable[int], min_prefix_len: int=1) -> list[PrefixSample]:
    ids = set((int(x) for x in case_ids))
    if not ids:
        return []
    ctab = cases[cases['case_uid'].isin(ids) & cases['usable_rtp']].set_index('case_uid')
    ev = events[events['case_uid'].isin(ctab.index) & events['timestamp'].notna()].copy()
    ev = ev.sort_values(['case_uid', 'timestamp', 'event_order'], kind='mergesort')
    samples: list[PrefixSample] = []
    for cid, g in ev.groupby('case_uid', sort=False):
        cid = int(cid)
        if cid not in ctab.index:
            continue
        crow = ctab.loc[cid]
        start, end = (pd.Timestamp(crow['case_start']), pd.Timestamp(crow['case_end']))
        g = g[g['timestamp'] <= end]
        names = g['activity_name'].fillna('<MISSING_ACTIVITY>').astype(str).tolist()
        times = [pd.Timestamp(x) for x in g['timestamp'].tolist()]
        if not names:
            continue
        tf = _event_time_features(times, start)
        for j in range(len(names)):
            plen = j + 1
            if plen < int(min_prefix_len):
                continue
            current = times[j]
            remaining = (end - current).total_seconds() / 3600.0
            if remaining <= 0:
                continue
            elapsed = max((current - start).total_seconds() / 3600.0, 0.0)
            delta = 0.0 if j == 0 else max((current - times[j - 1]).total_seconds() / 3600.0, 0.0)
            samples.append(PrefixSample(case_id=cid, business_key=f"{crow['process_name']}::{crow['raw_case_id']}", process_id=int(crow['process_id']), activity_names=names[:plen], time_features=tf[:plen].copy(), prefix_len=plen, elapsed_hours=float(elapsed), delta_hours=float(delta), remaining_hours=float(remaining), case_duration_hours=float(crow['duration_hours']), current_time=current))
    return samples

def _set_jaccard(a, b) -> float:
    a, b = (set(a), set(b))
    if not a and (not b):
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))

def _weighted_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum((min(a.get(k, 0), b.get(k, 0)) for k in keys))
    den = sum((max(a.get(k, 0), b.get(k, 0)) for k in keys))
    return float(num / den) if den else 1.0

def _robust_log_dist_sim(a, b) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]
    b = b[np.isfinite(b) & (b >= 0)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    la, lb = (np.log1p(a), np.log1p(b))
    ma, mb = (float(np.median(la)), float(np.median(lb)))
    qa, qb = (np.quantile(la, [0.25, 0.75]), np.quantile(lb, [0.25, 0.75]))
    dist = abs(ma - mb) + 0.5 * abs(float(qa[1] - qa[0]) - float(qb[1] - qb[0]))
    return float(math.exp(-dist))

def build_public_signature(events: pd.DataFrame, cases: pd.DataFrame, case_ids: Sequence[int]) -> dict:
    ids = set((int(x) for x in case_ids))
    c = cases[cases['case_uid'].isin(ids) & cases['usable_rtp']].copy()
    ev = events[events['case_uid'].isin(c['case_uid'])].copy().sort_values(['case_uid', 'timestamp', 'event_order'], kind='mergesort')
    activity_counts, dfg_counts, variant_counts = (Counter(), Counter(), Counter())
    gaps, trace_lengths = ([], [])
    for _, g in ev.groupby('case_uid', sort=False):
        names = g['activity_name'].astype(str).tolist()
        times = [pd.Timestamp(x) for x in g['timestamp']]
        if not names:
            continue
        activity_counts.update(names)
        dfg_counts.update(zip(names[:-1], names[1:]))
        variant_counts.update([tuple(names)])
        trace_lengths.append(len(names))
        gaps.extend((max((times[i] - times[i - 1]).total_seconds() / 3600.0, 0.0) for i in range(1, len(times))))
    return {'n_cases': int(len(c)), 'activity_counts': activity_counts, 'dfg_counts': dfg_counts, 'variant_counts': variant_counts, 'gaps_h': np.asarray(gaps, dtype=float), 'durations_h': pd.to_numeric(c['duration_hours'], errors='coerce').dropna().to_numpy(float), 'trace_lengths': np.asarray(trace_lengths, dtype=float)}
SIM_COMPONENTS = ('activity_similarity', 'dfg_similarity', 'temporal_similarity', 'duration_similarity', 'trace_length_similarity', 'variant_similarity')

def compare_public_signatures(source_sig: dict, target_sig: dict) -> dict:
    vals = {'activity_similarity': _set_jaccard(source_sig['activity_counts'], target_sig['activity_counts']), 'dfg_similarity': _weighted_jaccard(source_sig['dfg_counts'], target_sig['dfg_counts']), 'temporal_similarity': _robust_log_dist_sim(source_sig['gaps_h'], target_sig['gaps_h']), 'duration_similarity': _robust_log_dist_sim(source_sig['durations_h'], target_sig['durations_h']), 'trace_length_similarity': _robust_log_dist_sim(source_sig['trace_lengths'], target_sig['trace_lengths']), 'variant_similarity': _weighted_jaccard(source_sig['variant_counts'], target_sig['variant_counts'])}
    a = np.asarray([vals[x] for x in SIM_COMPONENTS], dtype=float)
    good = a[np.isfinite(a)]
    vals['equal6_similarity'] = float(good.mean()) if len(good) else np.nan
    return vals

def _public_file_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith('.xes.gz') else path.stem
_BP2020_ALIAS_STEMS = {'bpic2020dom': 'BPIC2020_Dom', 'bpic2020inter': 'BPIC2020_Inter', 'bpic2020per': 'BPIC2020_Per', 'bpic2020pre': 'BPIC2020_Pre', 'bpic2020req': 'BPIC2020_Req'}
_SKIP_DISCOVERY_DIRS = {'outputs', 'output', 'artifacts', 'artifact', 'checkpoints', 'checkpoint', '.git', '.idea', '.pytest_cache', '__pycache__', 'venv', '.venv'}

def _normalized_file_stem(path: Path) -> str:
    name = _public_file_name(path)
    return re.sub('[^a-z0-9]+', '', name.lower())

def discover_exact_bpic2020_aliases(data_dir: Path, max_depth: int=2) -> list[Path]:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if p.is_dir():
                    if depth < int(max_depth) and p.name.lower() not in _SKIP_DISCOVERY_DIRS:
                        stack.append((p, depth + 1))
                    continue
                if not p.is_file():
                    continue
            except OSError:
                continue
            low = p.name.lower()
            if not low.endswith(SUPPORTED_SUFFIXES):
                continue
            if _normalized_file_stem(p) in _BP2020_ALIAS_STEMS:
                out.append(p.resolve())
    return sorted(set(out), key=lambda p: str(p).lower())

def discover_public_files(data_dir: Path, recursive: bool=True) -> list[Path]:
    if not data_dir.exists():
        return []
    out: list[Path] = []
    if not recursive:
        iterator = ((p, ()) for p in data_dir.glob('*'))
        for p, _ in iterator:
            if p.is_file() and str(p).lower().endswith(SUPPORTED_SUFFIXES):
                out.append(p)
    else:
        import os
        for base, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DISCOVERY_DIRS]
            b = Path(base)
            for name in files:
                p = b / name
                if str(p).lower().endswith(SUPPORTED_SUFFIXES):
                    out.append(p)
    return sorted(set((x.resolve() for x in out)), key=lambda p: str(p).lower())

def infer_family_from_name(name: str) -> str | None:
    n = re.sub('[^a-z0-9]+', '', name.lower())
    bpic2020_official = ('domesticdeclarations', 'internationaldeclarations', 'prepaidtravelcost', 'prepaidtravelcosts', 'requestforpayment', 'permitlog', 'travelpermit')
    if any((tok in n for tok in bpic2020_official)):
        return 'bpic2020'
    if ('bpic2020' in n or 'bpi2020' in n or '2020' in n) and any((t in n for t in ('dom', 'inter', 'per', 'pre', 'req', 'permit'))):
        return 'bpic2020'
    if 'bpic2015' in n or 'bpi2015' in n or 'bpic15' in n or ('bpi15' in n) or ('2015' in n):
        return 'bpic2015'
    return None

def load_benchmark_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def resolve_specs(data_dir: Path, manifest_path: Path | None=None, min_processes: int=4) -> tuple[list[PublicLogSpec], dict]:
    manifest_path = manifest_path or data_dir / 'benchmark_manifest.json'
    manifest = load_benchmark_manifest(manifest_path)
    specs: list[PublicLogSpec] = []
    if manifest:
        source = manifest.get('benchmarks', manifest)
        for family, entries in source.items():
            for ent in entries:
                if isinstance(ent, str):
                    rel, overrides = (ent, {})
                else:
                    rel, overrides = (ent.get('file'), ent)
                if not rel:
                    continue
                p = Path(rel)
                if not p.is_absolute():
                    p = data_dir / p
                if not p.exists():
                    raise FileNotFoundError(f'Manifest file missing: {p}')
                pname = str(overrides.get('process_name') or _public_file_name(p))
                schema = None
                if overrides.get('case_col') and overrides.get('activity_col') and overrides.get('timestamp_col'):
                    schema = PublicSchema(str(overrides['case_col']), str(overrides['activity_col']), str(overrides['timestamp_col']), overrides.get('resource_col'), overrides.get('lifecycle_col'))
                terminal_activities = tuple((str(value) for value in overrides.get('terminal_activities', ())))
                specs.append(PublicLogSpec(pname, p.resolve(), str(family), schema, terminal_activities))
    else:
        priority = discover_exact_bpic2020_aliases(data_dir, max_depth=2)
        priority_set = {str(p).lower() for p in priority}
        candidates = priority + [p for p in discover_public_files(data_dir, recursive=True) if str(p).lower() not in priority_set]
        for p in candidates:
            stem = _public_file_name(p)
            family = infer_family_from_name(stem)
            if family:
                norm = _normalized_file_stem(p)
                pname = _BP2020_ALIAS_STEMS.get(norm, stem)
                specs.append(PublicLogSpec(pname, p, family, None))
    uniq = {}
    for s in specs:
        uniq[s.family, str(s.path).lower()] = s
    specs = list(uniq.values())
    families: dict[str, list[str]] = {}
    for s in specs:
        families.setdefault(s.family, []).append(s.process_name)
    valid = {f: sorted(v) for f, v in families.items() if len(set(v)) >= int(min_processes)}
    meta = {'manifest_path': str(manifest_path), 'manifest_used': bool(manifest), 'families_discovered': {k: sorted(v) for k, v in families.items()}, 'families_valid_by_count': valid}
    return (specs, meta)
