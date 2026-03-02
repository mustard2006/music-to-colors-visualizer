import numpy as np
#import librosa
import subprocess
import json

# Main methods
def decode_to_pcm(file):
    cmd = [
        "ffmpeg",
        "-i", file,
        "-f", "f32le",
        "-ac", "1",
        "-ar", "44100",
        "-"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    audio_bytes = process.stdout.read()

    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    return audio


def get_metadata(file):
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "a:0", 
        "-show_entries", "stream=duration,sample_rate,channels,channel_layout", 
        "-of", "json",
        file
    ]

    # ffprobe and capture stdout (capture_output=True)
    res = subprocess.run(cmd, capture_output=True, text=True)

    # parse json
    info = json.loads(res.stdout)
    stream = info['streams'][0]

    metadata = {
        "duration": float(stream["duration"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "channel_layout": stream.get("channel_layout", None)
    }

    return metadata



# FFT frame settings
FRAME_LEN = 2048   # samples per window
HOP_LEN = 512     # step between windows

# Frequency bands in Hz (for 44.1 kHz, Nyquist = 22050)
BASS = (20, 250)
MID = (250, 4000)
HIGH = (4000, 22050)


#def beat_and_onset_times(audio: np.ndarray, sr: int):