# run_llm_chat_cli.py

from llm_client import generate_reply


def main():
    print("AURA CLI chat (text + audio). Type 'quit' to exit.")
    print("For audio input, type:  audio path/to/file.wav\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"quit", "exit"}:
                print("Bye!")
                break

            # -------- AUDIO MODE --------
            if user_input.lower().startswith("audio "):
                wav_path = user_input[6:].strip()
                print(f"[debug] audio mode, wav_path={wav_path}")
                effective_text, emo_res, style, reply = generate_reply(
                    text=None,
                    wav_path=wav_path,
                    alpha_text=0.6,
                )
                print(f"[ASR transcript] {effective_text}")

            # -------- TEXT MODE --------
            else:
                effective_text, emo_res, style, reply = generate_reply(
                    text=user_input,
                    wav_path=None,
                    alpha_text=0.6,
                )

            # Pretty-print emotions & style
            probs_pretty = [
                (lab, float(p)) for lab, p in zip(emo_res.labels, emo_res.probs)
            ]
            print(f"[mode={emo_res.mode}] emotions={probs_pretty}")
            print(
                f"[style] tone={style.tone}, "
                f"verbosity={style.verbosity}, "
                f"directness={style.directness}"
            )
            if getattr(emo_res, "crisis_flag", False):
                print(f"[CRISIS FLAG] reason: {emo_res.crisis_reason}")

            print(f"\nAURA: {reply}\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")


if __name__ == "__main__":
    main()
