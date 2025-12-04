import matplotlib.pyplot as plt

# =========================
# 1) TEXT MODELS (ROBERTA)
# =========================

epochs_text = [1, 2, 3]

# RoBERTa-base (from your logs)
base_macro_f1 = [0.3189, 0.4254, 0.4456]
base_sample_acc = [0.4243, 0.4637, 0.4615]

# RoBERTa-large (from your logs)
large_macro_f1 = [0.4514, 0.4956, 0.5222]
large_sample_acc = [0.4456, 0.4513, 0.4606]

# ---- Plot: Macro-F1 over epochs for base vs large ----
plt.figure(figsize=(6, 4))
plt.plot(epochs_text, base_macro_f1, marker="o", label="RoBERTa-base Macro-F1")
plt.plot(epochs_text, large_macro_f1, marker="o", linestyle="--", label="RoBERTa-large Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1 (validation)")
plt.title("Text Emotion Model – Macro-F1 vs Epoch")
plt.xticks(epochs_text)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("text_roberta_macro_f1.png", dpi=300)
plt.close()

# ---- Plot: Sample Accuracy over epochs for base vs large ----
plt.figure(figsize=(6, 4))
plt.plot(epochs_text, base_sample_acc, marker="o", label="RoBERTa-base Sample Acc")
plt.plot(epochs_text, large_sample_acc, marker="o", linestyle="--", label="RoBERTa-large Sample Acc")
plt.xlabel("Epoch")
plt.ylabel("Sample Accuracy (validation)")
plt.title("Text Emotion Model – Sample Accuracy vs Epoch")
plt.xticks(epochs_text)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("text_roberta_sample_acc.png", dpi=300)
plt.close()

# ==================================================
# 2) SPEECH MODEL (WAV2VEC2 ON CREMA-D)
# ==================================================

epochs_speech = list(range(1, 11))

wa =  [0.3474, 0.6594, 0.6920, 0.6811, 0.6947,
       0.7341, 0.7286, 0.7639, 0.7341, 0.7490]

ua =  [0.3457, 0.6565, 0.6983, 0.6870, 0.7014,
       0.7339, 0.7324, 0.7675, 0.7375, 0.7522]

macro_f1_speech = [
    0.2968, 0.6606, 0.6849, 0.6675, 0.6851,
    0.7344, 0.7242, 0.7614, 0.7334, 0.7476
]

# ---- Plot: speech WA / UA / Macro-F1 over epochs ----
plt.figure(figsize=(7, 4))
plt.plot(epochs_speech, wa, marker="o", label="WA (Weighted Acc)")
plt.plot(epochs_speech, ua, marker="o", linestyle="--", label="UA (Unweighted Acc)")
plt.plot(epochs_speech, macro_f1_speech, marker="o", linestyle="-.", label="Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Speech Emotion Model – Validation Metrics vs Epoch")
plt.xticks(epochs_speech)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("speech_erc_metrics.png", dpi=300)
plt.close()

# ==================================================
# 3) COMPARISON BAR CHART – BEST MACRO-F1
# ==================================================

best_macro_text_base = max(base_macro_f1)          # 0.4456
best_macro_text_large = max(large_macro_f1)        # 0.5222
best_macro_speech = max(macro_f1_speech)           # 0.7614

models = ["RoBERTa-base (text)", "RoBERTa-large (text)", "Wav2Vec2 (speech)"]
best_macro = [best_macro_text_base, best_macro_text_large, best_macro_speech]

plt.figure(figsize=(6, 4))
plt.bar(models, best_macro)
plt.ylabel("Best Validation Macro-F1")
plt.title("Best Macro-F1 Across Models")
plt.ylim(0, 1.0)
for i, v in enumerate(best_macro):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("best_macro_f1_comparison.png", dpi=300)
plt.close()

# ==================================================
# 4) OPTIONAL: TRAINING LOSS CURVES (SKELETON)
# ==================================================
# If you later extract average training loss per epoch from your logs,
# you can just fill these lists and re-run the script.

# Example skeleton (values here are placeholders – replace with real averages):
# base_train_loss = [0.30, 0.20, 0.18]
# large_train_loss = [0.27, 0.19, 0.17]

# plt.figure(figsize=(6, 4))
# plt.plot(epochs_text, base_train_loss, marker="o", label="RoBERTa-base train loss")
# plt.plot(epochs_text, large_train_loss, marker="o", linestyle="--", label="RoBERTa-large train loss")
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.title("Text Models – Training Loss vs Epoch")
# plt.xticks(epochs_text)
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend()
# plt.tight_layout()
# plt.savefig("text_roberta_train_loss.png", dpi=300)
# plt.close()

print("Saved figures:")
print("  - text_roberta_macro_f1.png")
print("  - text_roberta_sample_acc.png")
print("  - speech_erc_metrics.png")
print("  - best_macro_f1_comparison.png")
# plus text_roberta_train_loss.png if you fill in losses
