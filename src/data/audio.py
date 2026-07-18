"""Audio signal primitives for the gugak stem-separation pipeline.

Low-level, reusable audio I/O + DSP ops shared by EDA, preprocessing, and the
future stem-summing dataloader. One-off analyses (e.g. the master ≈ Σ(stems)
residual study) and the notebook import from here rather than reimplementing.

Design note: this is the *audio primitives* layer. Mixing/grouping (stem → 4-stem
mixture) and the Dataset class live in sibling modules and import these.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path, start: int = 0, frames: int = -1, dtype: str = "float32"):
    """Read a WAV as ``(frames, channels)`` float array + sample rate (always 2-D)."""
    audio, sr = sf.read(str(path), start=start, frames=frames, dtype=dtype, always_2d=True)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix ``(frames, channels)`` -> mono ``(frames,)`` by channel average."""
    return audio.mean(axis=1)


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude (float64 accumulation)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak(x: np.ndarray) -> float:
    """Peak absolute amplitude."""
    x = np.asarray(x)
    return float(np.abs(x).max()) if x.size else 0.0


def to_db(amp: float, floor_db: float = -120.0) -> float:
    """Amplitude ratio -> dB, clamped to ``floor_db`` for silence."""
    amp = float(amp)
    return floor_db if amp <= 0 else max(floor_db, 20.0 * float(np.log10(amp)))


def prefix_energy(mono: np.ndarray) -> np.ndarray:
    """Prefix sum of squared samples (length N+1, ``[0]=0``) for O(1) window energy.

    ``window_energy = prefix[start + win] - prefix[start]``.
    """
    m = np.asarray(mono, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(m * m)))


def window_rms(prefix: np.ndarray, start: int, win: int) -> float:
    """RMS of the length-``win`` window at ``start``, from a prefix-energy array."""
    return float(np.sqrt((prefix[start + win] - prefix[start]) / win))


def loudest_window(prefix: np.ndarray, win: int, hop: int) -> int:
    """Start index of the highest-energy length-``win`` window (stride ``hop``).

    Returns 0 if the signal is shorter than ``win``.
    """
    n = len(prefix) - 1
    if n <= win:
        return 0
    starts = np.arange(0, n - win + 1, hop)
    energies = prefix[starts + win] - prefix[starts]
    return int(starts[int(energies.argmax())])


# --- mixing / residual analysis (master ≈ Σ(stems)) -------------------------

def estimate_lag(ref: np.ndarray, est: np.ndarray, sr: int,
                 max_lag_ms: float = 50.0, probe_s: float = 30.0) -> int:
    """Integer sample lag ``d`` s.t. ``ref[k] ≈ est[k - d]`` (ref delayed by d>0).

    FFT cross-correlation on a central mono probe; search limited to ±max_lag_ms.
    Guards against a spurious large shift when the true offset is small (recording
    latency between the mastered mix and the stems is typically a handful of samples).
    """
    a, b = to_mono(ref), to_mono(est)
    length = min(len(a), len(b))
    w = min(int(probe_s * sr), length)
    if w < 2:
        return 0
    s = (length - w) // 2
    a, b = a[s:s + w], b[s:s + w]
    n = 1 << int(np.ceil(np.log2(2 * w)))
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    max_lag = int(max_lag_ms / 1000 * sr)
    if max_lag < 1:
        return 0
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([xc[:max_lag + 1], xc[-max_lag:]])
    return int(lags[int(vals.argmax())])


def fit_scalar_gain(ref: np.ndarray, est: np.ndarray) -> float:
    """Least-squares scalar ``g`` minimizing ‖ref − g·est‖ (mastering level change)."""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    denom = float(np.sum(est * est))
    return float(np.sum(ref * est) / denom) if denom > 0 else 0.0


def fit_linear_mix(stems, target, ridge: float = 1e-9):
    """Least-squares per-stem scalar gains ``w`` minimizing ‖target − Σ wᵢ·stemᵢ‖.

    ``stems``: list of aligned equal-length ``(L, ch)`` arrays; ``target``: ``(L, ch)``.
    Returns ``(weights, reconstruction)``. NOTE: gugak stems are heterophonic and thus
    often collinear, so the fitted weights can be non-physical (huge/negative) even when
    the residual is small — interpret the RESIDUAL, not the individual weights.
    """
    n = len(stems)
    X = [np.asarray(s, dtype=np.float64).reshape(-1) for s in stems]
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    G = np.array([[X[i] @ X[j] for j in range(n)] for i in range(n)])
    b = np.array([X[i] @ y for i in range(n)])
    w = np.linalg.solve(G + ridge * np.eye(n) * (np.trace(G) / n), b)
    recon = sum(wi * si for wi, si in zip(w, stems))
    return w, recon


def read_and_sum(paths, dtype: str = "float32"):
    """Read WAVs and sum them sample-aligned, trimming to the shortest (mix, sr, lens).

    Handles small inter-stem length mismatches (e.g. 판소리) by trimming to the min.
    """
    sigs, sr = [], None
    for p in paths:
        a, s = read_audio(p, dtype=dtype)
        sr = s if sr is None else sr
        sigs.append(a)
    length = min(len(a) for a in sigs)
    mix = np.zeros((length, sigs[0].shape[1]), dtype=np.float64)
    for a in sigs:
        mix += a[:length]
    return mix, sr, [len(a) for a in sigs]


def residual_report(stem_paths, master_path,
                    max_lag_ms: float = 50.0, probe_s: float = 30.0) -> dict:
    """Compute master ≈ Σ(stems) residual for one song.

    Sums the stems, aligns the master to the sum (integer lag), fits an optimal scalar
    gain, and reports the residual RMS relative to the master, in dB — both raw and
    gain-compensated. Lower dB = closer to linear mixing (summing stems is safe).
    """
    mix, sr, stem_lens = read_and_sum(stem_paths)
    master, _ = read_audio(master_path, dtype="float32")
    master = master.astype(np.float64)

    d = estimate_lag(master, mix, sr, max_lag_ms=max_lag_ms, probe_s=probe_s)
    if d >= 0:
        ref_a, est_a = master[d:], mix
    else:
        ref_a, est_a = master, mix[-d:]
    L = min(len(ref_a), len(est_a))
    ref_a, est_a = ref_a[:L], est_a[:L]

    g = fit_scalar_gain(ref_a, est_a)
    ref_rms = rms(ref_a)
    resid_raw = rms(ref_a - est_a)
    resid_gain = rms(ref_a - g * est_a)
    to_rel_db = lambda r: (to_db(r / ref_rms) if ref_rms > 0 else float("nan"))
    return dict(
        n_stems=len(stem_paths),
        sr=sr,
        lag_samples=d,
        lag_ms=round(1000 * d / sr, 3),
        gain=round(g, 4),
        resid_db_raw=round(to_rel_db(resid_raw), 2),
        resid_db_gain=round(to_rel_db(resid_gain), 2),
        master_frames=int(len(master)),
        mix_frames=int(len(mix)),
        len_diff_ms=round(1000 * (len(master) - min(stem_lens)) / sr, 1),
        master_rms=round(ref_rms, 6),
    )
