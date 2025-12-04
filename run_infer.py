# run_infer.py
import sys
from text_erc.infer import predict

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_infer.py \"your text here\"")
        raise SystemExit(1)

    text = " ".join(sys.argv[1:])
    predict(text, top_k=5)

if __name__ == "__main__":
    main()
