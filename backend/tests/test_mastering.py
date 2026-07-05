import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mastering import apply_mastering_chain


def test_apply_mastering_chain_handles_lufs_normalization_without_name_error():
    sr = 22050
    audio = np.random.randn(2, sr * 2).astype(np.float32) * 0.1

    audio_out, meters = apply_mastering_chain(
        audio,
        sr,
        use_lufs_normalize=True,
        target_lufs=-20.0,
        input_gain_db=0.0,
        oversample_mode="draft",
        comp_threshold=0.6,
        comp_ratio=2.0,
        comp_attack_ms=10.0,
        comp_release_ms=80.0,
        comp_makeup_db=0.0,
    )

    assert audio_out.shape == audio.shape
    assert meters["post_limiter"].get("lufs") is not None
