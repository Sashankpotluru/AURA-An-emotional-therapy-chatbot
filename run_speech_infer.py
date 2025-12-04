# run_speech_infer.py
import sys
from speech_erc.infer import predict_from_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_speech_infer.py path/to/audio.wav")
        raise SystemExit(1)

    wav_path = sys.argv[1]
    predict_from_file(wav_path, top_k=3)

if __name__ == "__main__":
    main()
