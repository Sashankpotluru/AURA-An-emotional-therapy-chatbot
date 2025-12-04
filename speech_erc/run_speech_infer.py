# speech_erc/run_speech_infer.py

import sys
from .infer import predict


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m speech_erc.run_speech_infer path/to/audio.wav")
        raise SystemExit(1)
    path = sys.argv[1]
    predict(path, top_k=3)


if __name__ == "__main__":
    main()
