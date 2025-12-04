# run_policy_demo.py (example usage)
from emotion_router import analyze_with_style

text = "I feel like nothing will ever get better."
wav_path = "data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav"

# Text-only
emo_res_text, style_text = analyze_with_style(text=text, wav_path=None)
print("=== TEXT-ONLY ===")
print("Top emotions:", list(zip(emo_res_text.labels, emo_res_text.probs)))
print("Style:", style_text)

# Speech-only
emo_res_speech, style_speech = analyze_with_style(text=None, wav_path=wav_path)
print("\n=== SPEECH-ONLY ===")
print("Top emotions:", list(zip(emo_res_speech.labels, emo_res_speech.probs)))
print("Style:", style_speech)


