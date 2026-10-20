"""Small I/O helpers: deterministic files (UTF-8, LF) and an offline guard."""

from __future__ import annotations

import contextlib
import json
import math
import socket
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 10)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, sort_keys=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path, float_format: str = "%.6f") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format=float_format)
    return path


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


class NetworkDisabled(RuntimeError):
    pass


@contextlib.contextmanager
def no_network():
    """Fail loudly if anything inside tries to open a network connection."""
    orig_connect, orig_create = socket.socket.connect, socket.create_connection

    def refuse(*args, **kwargs):
        raise NetworkDisabled("network access attempted inside an offline wattline step")

    socket.socket.connect = refuse          # type: ignore[method-assign]
    socket.create_connection = refuse       # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.create_connection = orig_create
