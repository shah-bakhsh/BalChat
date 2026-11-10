from pathlib import Path
import subprocess
import sys


def test_training_requires_explicit_experimental_mode_before_any_ml_download(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "train.py"), "--purpose", "commercial", "--prepared", str(tmp_path / "missing"), "--output", str(tmp_path / "run")], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "--experimental is required" in result.stderr
    assert not (tmp_path / "run").exists()
