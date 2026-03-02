"""
Audio analyzer: FFT per frame + band energies (bass / mid / high).

Theory (short version):
-----------------------
1. Frames
   We cut the waveform into overlapping windows of FRAME_LEN samples (e.g. 2048).
   Each window is one "frame". We advance by HOP_LEN (e.g. 512) so frames overlap.
   So we get one snapshot every ~11.6 ms.

2. FFT per frame
   For each frame we run a "real FFT" (rfft) on the windowed samples.
   - Input: 2048 time-domain samples (amplitude over time).
   - Output: 1025 "bins" (for 2048 samples). Each bin k corresponds to a frequency
     f = k * (sample_rate / FRAME_LEN). So bin 0 = 0 Hz, bin 1 ≈ 21.5 Hz, ...
     up to bin 1024 = 22050 Hz (Nyquist). The value in each bin is the magnitude
     (strength) of that frequency in this frame.

3. Frequency → bin index
   Bin index k for frequency f (Hz):  k = f * FRAME_LEN / sample_rate
   So we can slice the FFT magnitude array to get only the bins that fall in
   [20, 250], [250, 4000], [4000, 22050] Hz.

4. Band energies
   For each frame we sum the FFT magnitudes in each band. That gives three
   numbers per frame: (bass_energy, mid_energy, high_energy). No need to keep
   every bin—just these three sums for the visualizer.
"""

import numpy as np
from utils import FRAME_LEN, HOP_LEN, BASS, MID, HIGH


def _hz_to_bin(f_hz: float, sr: int) -> int:
    """Convert frequency in Hz to FFT bin index (for rfft of length FRAME_LEN)."""
    return int(f_hz * FRAME_LEN / sr)


def _band_bin_ranges(sr: int):
    """Return (start, end) bin indices for BASS, MID, HIGH (end exclusive)."""
    bass_lo, bass_hi = BASS
    mid_lo, mid_hi = MID
    high_lo, high_hi = HIGH
    return (
        (_hz_to_bin(bass_lo, sr), _hz_to_bin(bass_hi, sr) + 1),
        (_hz_to_bin(mid_lo, sr), _hz_to_bin(mid_hi, sr) + 1),
        (_hz_to_bin(high_lo, sr), _hz_to_bin(high_hi, sr) + 1),
    )


def analyze_track(audio: np.ndarray, sr: int):
    """
    Run FFT per frame and sum magnitudes in bass/mid/high bands.

    Returns:
        band_energies: shape (num_frames, 3), each row is (bass, mid, high).
        frame_times: shape (num_frames,), time in seconds for each frame.
    """
    bass_range, mid_range, high_range = _band_bin_ranges(sr)
    window = np.hanning(FRAME_LEN)

    # Number of full frames that fit in the audio
    num_frames = max(0, (len(audio) - FRAME_LEN) // HOP_LEN + 1)
    band_energies = np.zeros((num_frames, 3))
    frame_times = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * HOP_LEN
        frame = audio[start : start + FRAME_LEN].astype(np.float64) * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)

        band_energies[i, 0] = np.sum(mag[bass_range[0] : bass_range[1]])
        band_energies[i, 1] = np.sum(mag[mid_range[0] : mid_range[1]])
        band_energies[i, 2] = np.sum(mag[high_range[0] : high_range[1]])

        frame_times[i] = start / sr

    return band_energies, frame_times


def get_frame_index_for_time(t_sec: float, sr: int, hop_len: int = HOP_LEN) -> int:
    """
    Given playback time in seconds, return the frame index that contains this time.
    Frame i covers time [ frame_time[i], frame_time[i] + FRAME_LEN/sr ).
    """
    sample_index = t_sec * sr
    frame_index = int(sample_index // hop_len)
    return max(0, frame_index)


def is_beat_near(t_sec: float, window_sec: float = 0.1) -> bool:
    """
    Placeholder: returns False until beat detection is implemented.
    Intended: True if there is a beat within window_sec of t_sec.
    """
    return False
