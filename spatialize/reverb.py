from __future__ import annotations

import math

import numpy as np


_REFERENCE_RATE = 44_100
_COMB_DELAYS_SAMPLES = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_STEREO_SPREAD_SAMPLES = 23
_ALLPASS_DELAYS_SAMPLES = (556, 441, 341, 225)
_ALLPASS_GAIN = 0.5
_ENVELOPE_RELEASE_S = 0.02


def _scale_delay(base_samples: int, sample_rate: int) -> int:
    return max(1, round(base_samples * sample_rate / _REFERENCE_RATE))


def comb_feedback_gain(delay_samples: int, sample_rate: int, rt60_s: float) -> float:
    if delay_samples <= 0:
        raise ValueError("delay_samples must be positive")
    if rt60_s <= 0:
        raise ValueError("rt60_s must be positive")
    exponent = -3.0 * delay_samples / (sample_rate * rt60_s)
    return 10.0**exponent


def feedback_comb(x: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    if delay_samples <= 0:
        raise ValueError("delay_samples must be positive")
    if not abs(feedback) < 1.0:
        raise ValueError("feedback gain must satisfy |feedback| < 1 for stability")

    y = np.empty_like(x)
    buffer = np.zeros(delay_samples, dtype=x.dtype)
    pos = 0
    for i in range(x.shape[0]):
        value = x[i] + feedback * buffer[pos]
        y[i] = value
        buffer[pos] = value
        pos += 1
        if pos == delay_samples:
            pos = 0
    return y


def allpass_filter(x: np.ndarray, delay_samples: int, gain: float = _ALLPASS_GAIN) -> np.ndarray:
    if delay_samples <= 0:
        raise ValueError("delay_samples must be positive")
    if not 0.0 <= gain < 1.0:
        raise ValueError("allpass gain must be in [0, 1)")

    y = np.empty_like(x)
    x_buffer = np.zeros(delay_samples, dtype=x.dtype)
    y_buffer = np.zeros(delay_samples, dtype=x.dtype)
    pos = 0
    for i in range(x.shape[0]):
        delayed_x = x_buffer[pos]
        delayed_y = y_buffer[pos]
        value = -gain * x[i] + delayed_x + gain * delayed_y
        y[i] = value
        x_buffer[pos] = x[i]
        y_buffer[pos] = value
        pos += 1
        if pos == delay_samples:
            pos = 0
    return y


def _damp(x: np.ndarray, damping: float) -> np.ndarray:
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")
    if damping == 0.0:
        return x
    y = np.empty_like(x)
    state = 0.0
    gain = 1.0 - damping
    for i in range(x.shape[0]):
        state = gain * x[i] + damping * state
        y[i] = state
    return y


def _peak_envelope(x: np.ndarray, sample_rate: int, release_s: float = _ENVELOPE_RELEASE_S) -> np.ndarray:
    release_coeff = math.exp(-1.0 / (release_s * sample_rate))
    envelope = np.empty_like(x)
    level = 0.0
    for i in range(x.shape[0]):
        level = max(abs(x[i]), level * release_coeff)
        envelope[i] = level
    return envelope


def _independent_diffuse_noise(
    dry_channel: np.ndarray, diffusion: float, rng: np.random.Generator, sample_rate: int
) -> np.ndarray:
    envelope = _peak_envelope(dry_channel, sample_rate)
    return rng.standard_normal(dry_channel.shape[0]) * diffusion * envelope


def _channel_reverb(
    dry: np.ndarray,
    sample_rate: int,
    rt60_s: float,
    damping: float,
    stereo_offset_samples: int,
    diffuse_noise: np.ndarray,
) -> np.ndarray:
    excited = dry + diffuse_noise
    wet = np.zeros_like(dry)
    for base_delay in _COMB_DELAYS_SAMPLES:
        delay_samples = _scale_delay(base_delay, sample_rate) + stereo_offset_samples
        feedback = comb_feedback_gain(delay_samples, sample_rate, rt60_s)
        wet += _damp(feedback_comb(excited, delay_samples, feedback), damping)
    wet /= len(_COMB_DELAYS_SAMPLES)

    for base_delay in _ALLPASS_DELAYS_SAMPLES:
        delay_samples = _scale_delay(base_delay, sample_rate)
        wet = allpass_filter(wet, delay_samples, _ALLPASS_GAIN)
    return wet


def schroeder_reverb(
    stereo: np.ndarray,
    sample_rate: int,
    rt60_s: float,
    wet: float,
    damping: float = 0.25,
    diffusion: float = 0.6,
    seed: int = 0,
    tail_s: float | None = None,
) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")
    if not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be in [0, 1]")
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must be in [0, 1]")

    if tail_s is None:
        tail_s = 1.2 * rt60_s
    tail_samples = int(math.ceil(tail_s * sample_rate))
    padded = np.pad(stereo, ((0, tail_samples), (0, 0)))

    if wet == 0.0:
        return padded

    mono_excitation = 0.5 * (padded[:, 0] + padded[:, 1])

    rng = np.random.default_rng(seed)
    noise_left = _independent_diffuse_noise(mono_excitation, diffusion, rng, sample_rate)
    noise_right = _independent_diffuse_noise(mono_excitation, diffusion, rng, sample_rate)

    stereo_offset = _scale_delay(_STEREO_SPREAD_SAMPLES, sample_rate)
    left_wet = _channel_reverb(mono_excitation, sample_rate, rt60_s, damping, 0, noise_left)
    right_wet = _channel_reverb(
        mono_excitation, sample_rate, rt60_s, damping, stereo_offset, noise_right
    )
    wet_signal = np.column_stack((left_wet, right_wet))

    return (1.0 - wet) * padded + wet * wet_signal
