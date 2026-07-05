import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_engine import iter_mastering_chunks


def test_iter_mastering_chunks_returns_finite_audio_and_metrics():
    sr = 16000
    audio = np.sin(2 * np.pi * 220 * np.arange(sr * 0.2) / sr).astype(np.float32)
    audio = np.stack([audio, audio * 0.9], axis=0)

    outputs = []
    for processed, metrics in iter_mastering_chunks(audio, sr, chunk_seconds=0.05):
        outputs.append(processed)
        assert np.isfinite(processed).all()
        assert metrics["lufs_momentary"] is not None
        assert metrics["peak_db"] is not None

    total_samples = sum(block.shape[-1] for block in outputs)
    assert total_samples >= audio.shape[-1]
