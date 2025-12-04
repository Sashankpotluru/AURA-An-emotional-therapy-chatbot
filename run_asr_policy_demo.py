# run_asr_policy_demo.py
"""
Demo: audio-only input with ASR + fusion + style + crisis detection.

Usage:
    python3 run_asr_policy_demo.py data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav
"""

import sys

from emotion_router import analyze_audio_with_asr
from response_policy import build_safe_prompt  # you already created this

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_asr_policy_demo.py path/to/audio.wav")
        raise SystemExit(1)

    wav_path = sys.argv[1]

    emo_res, style, transcript = analyze_audio_with_asr(wav_path, alpha_text=0.6)

    print("=== TRANSCRIPT ===")
    print(transcript)
    print()

    print("=== EMOTION RESULT ===")
    print("mode:", emo_res.mode)
    print("probs:", list(zip(emo_res.labels, emo_res.probs)))
    print("crisis_flag:", emo_res.crisis_flag)
    print("crisis_reason:", emo_res.crisis_reason)
    print("style:", style)
    print()

    prompt = build_safe_prompt(transcript, emo_res, style)

    print("=== LLM PROMPT ===")
    print(prompt)

if __name__ == "__main__":
    main()
