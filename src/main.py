from analyzer import analyze_track, get_frame_index_for_time, is_beat_near
from utils import decode_to_pcm, get_metadata

SR = 44100

def main():
    file_path = "./audio_samples/Deftones_My_Own_Summer.wav"

    pcm = decode_to_pcm(file_path)
    print("PCM shape:", pcm.shape, "samples")

    metadata = get_metadata(file_path)
    print("Metadata:", metadata)

    band_energies, frame_times = analyze_track(pcm, SR)
    print("\nAnalyzer output:")
    print("  band_energies shape:", band_energies.shape, "(frames, 3 = bass/mid/high)")
    print("  frame_times shape:", frame_times.shape)
    print("  first 5 frames (bass, mid, high):")
    print(band_energies[:5])
    print("  frame at t=10s -> index", get_frame_index_for_time(10.0, SR))
    print("  band energies at t=10s:", band_energies[get_frame_index_for_time(10.0, SR)])

if __name__ == "__main__":
    main()