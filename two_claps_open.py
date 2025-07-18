import time
import webbrowser
import os
import numpy as np
import pyaudio
from scipy import signal

SAMPLING_RATE = 44100
FRAME_DURATION = 0.02
FRAME_SIZE = int(SAMPLING_RATE * FRAME_DURATION)

FILTER_ORDER = 2
FREQ_LOW = 1400
FREQ_HIGH = 1800

AMPLITUDE_THRESHOLD = 0.2
MIN_PEAK_INTERVAL = 0.2


def initialize_audio_stream():
    p = pyaudio.PyAudio()
    return p, p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLING_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )


def create_bandpass_filter():
    return signal.butter(
        FILTER_ORDER,
        [FREQ_LOW, FREQ_HIGH],
        btype="bandpass",
        fs=SAMPLING_RATE,
        output="sos",
    )


def main():
    p, stream = initialize_audio_stream()
    sos = create_bandpass_filter()
    window = signal.windows.hann(FRAME_SIZE)

    clap_count = 0
    clap_times = []
    last_peak_time = -float("inf")
    filter_state = np.zeros((sos.shape[0], 2))

    print("Starting real-time processing. Please clap twice.")
    start_time = time.time()

    try:
        while clap_count < 2:
            frame = np.frombuffer(
                stream.read(FRAME_SIZE, exception_on_overflow=False), dtype=np.float32
            )
            frame = frame * window

            frame_filtered, filter_state = signal.sosfilt(sos, frame, zi=filter_state)

            peaks, _ = signal.find_peaks(
                np.abs(frame_filtered), height=AMPLITUDE_THRESHOLD
            )
            current_time = time.time() - start_time

            if peaks.size > 0 and (current_time - last_peak_time) >= MIN_PEAK_INTERVAL:
                clap_count += 1
                clap_times.append(current_time)
                print(f"Clap detected: {current_time:.2f} seconds")
                last_peak_time = current_time

                if clap_count == 2:
                    webbrowser.open("steam://rungameid/570")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    print(f"\nResults:")
    print(f"Sampling frequency: {SAMPLING_RATE} Hz")
    print(f"Number of detected claps: {clap_count}")
    print("Two claps have been detected!")

    if clap_times:
        print("Clap times (seconds):")
        for t in clap_times:
            print(f"{t:.4f}")


if __name__ == "__main__":
    main()
