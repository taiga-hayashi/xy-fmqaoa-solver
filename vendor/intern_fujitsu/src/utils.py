import sys
import torch
import itertools

try:  # Supports both ``python -m src...`` and legacy direct script execution.
    from .fm import TorchFM
except ImportError:  # pragma: no cover - exercised by direct script execution
    from fm import TorchFM

class Logger:
    """A small stdout tee that can also be used safely as a context manager."""

    def __init__(self, filename, terminal=None):
        self.terminal = terminal or sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def get_fm_minimum(model: TorchFM, d: int):
    """FMサロゲートモデル上の最小値とその入力を全探索で求める"""
    all_patterns = list(itertools.product([0, 1], repeat=d))
    x_tensor = torch.tensor(all_patterns, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
    min_idx = torch.argmin(outputs).item()
    return all_patterns[min_idx], outputs[min_idx].item()
