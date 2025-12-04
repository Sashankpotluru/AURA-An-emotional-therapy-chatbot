# run_fusion_infer.py

import sys
from fusion.infer import pretty_print_fusion

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_fusion_infer.py \"your text\" path/to/audio.wav [alpha_text]")
        raise SystemExit(1)

    text = sys.argv[1]
    wav_path = sys.argv[2]

    if len(sys.argv) >= 4:
        alpha_text = float(sys.argv[3])
    else:
        alpha_text = 0.6

    pretty_print_fusion(text, wav_path, alpha_text=alpha_text)

if __name__ == "__main__":
    main()
