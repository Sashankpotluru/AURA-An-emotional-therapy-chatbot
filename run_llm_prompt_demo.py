# run_llm_prompt_demo.py

from emotion_router import analyze_with_style
from response_policy import build_safe_prompt


def demo(text: str, wav_path: str | None = None):
    emo_res, style = analyze_with_style(text=text, wav_path=wav_path, alpha_text=0.6)
    prompt = build_safe_prompt(text, emo_res, style)

    print("=== EMOTION RESULT ===")
    print("mode:", emo_res.mode)
    print("probs:", list(zip(emo_res.labels, emo_res.probs)))
    print("crisis_flag:", emo_res.crisis_flag)
    print("crisis_reason:", emo_res.crisis_reason)
    print("style:", style)
    print("\n=== GENERATED PROMPT FOR LLM ===\n")
    print(prompt)


if __name__ == "__main__":
    # Example 1: non-crisis, sad text
    text = "I feel like nothing will ever get better."
    demo(text=text, wav_path=None)

    # Later you can add a speech example:
    # demo(text=None, wav_path="data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav")
