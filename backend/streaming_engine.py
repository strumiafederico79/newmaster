"""
Motor de mastering en tiempo real / streaming.
Procesa el audio en bloques (chunks) entregando PCM16 y métricas completas
(LUFS, peak, RMS, correlación estéreo, GR multibanda, GR banda ancha y
glue) para visualización en vivo sin bloquear la interfaz.
"""
import numpy as np
from mastering import apply_mastering_chain, measure_lufs_integrated, stereo_correlation

CHUNK_SECONDS_DEFAULT = 2.0


def iter_mastering_chunks(audio: np.ndarray, sr: int,
                          chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                          **chain_params):
    """
    Yields (processed_block, metrics_dict) for each chunk.
    metrics_dict now includes gain-reduction data from chain_meters for las
    tres etapas de dinámica (multibanda, banda ancha, glue) más el snapshot
    VU pre/post limiter.
    """
    total_samples = audio.shape[-1]
    chunk_samples = max(1, int(chunk_seconds * sr))
    n_chunks = int(np.ceil(total_samples / chunk_samples))

    for i in range(n_chunks):
        start = i * chunk_samples
        end   = min(start + chunk_samples, total_samples)
        block = audio[:, start:end] if audio.ndim == 2 else audio[start:end]
        if block.shape[-1] == 0:
            continue

        # apply_mastering_chain now returns (audio, chain_meters)
        processed, chain_meters = apply_mastering_chain(block, sr, **chain_params)

        mono     = processed.mean(axis=0) if processed.ndim == 2 else processed
        peak     = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms      = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
        peak_db  = float(20.0 * np.log10(peak + 1e-9))
        rms_db   = float(20.0 * np.log10(rms  + 1e-9))

        try:
            lufs_chunk = measure_lufs_integrated(processed, sr)
        except Exception:
            lufs_chunk = rms_db - 0.691

        # Stereo correlation
        corr = stereo_correlation(processed)

        metrics = {
            "chunk_index":        i,
            "n_chunks":           n_chunks,
            "progress_pct":       round(((i + 1) / n_chunks) * 100.0, 1),
            "peak_db":            round(peak_db, 2),
            "rms_db":             round(rms_db, 2),
            "lufs_momentary":     round(lufs_chunk, 2),
            "stereo_correlation": round(corr, 3),
            "time_sec":           round(start / sr, 2),
            # Multiband compressor gain-reduction metering for front-end rendering
            "mb_meters": chain_meters.get("mb", {}),
            # Compresor de banda ancha ("Dinámica") y glue compressor
            "comp_meters": chain_meters.get("comp", {}),
            "glue_meters": chain_meters.get("glue", {}),
            # Pre/post limiter VU snapshot
            "pre_limiter":        chain_meters.get("pre_limiter", {}),
            "post_limiter":       chain_meters.get("post_limiter", {}),
        }
        yield processed, metrics


def master_stream_to_pcm16(audio: np.ndarray, sr: int,
                           chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                           **chain_params):
    """
    Yields (pcm16_bytes, metrics_dict) for each processed chunk.
    pcm16_bytes: interleaved int16 PCM (little-endian).
    """
    for processed, metrics in iter_mastering_chunks(audio, sr,
                                                    chunk_seconds=chunk_seconds,
                                                    **chain_params):
        block  = processed.T if processed.ndim == 2 else processed
        pcm16  = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
        yield pcm16.tobytes(), metrics
