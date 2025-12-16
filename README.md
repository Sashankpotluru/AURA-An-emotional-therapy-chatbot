# AURA – Emotion-Aware Multimodal Therapy Chatbot

AURA is a multimodal chatbot that combines:

- **Text emotion recognition** using a RoBERTa encoder fine-tuned on GoEmotions.
- **Speech emotion recognition** using a Wav2Vec2 encoder fine-tuned on CREMA-D.
- **Late fusion** of text and speech emotion distributions.
- A **policy layer** that maps fused emotions to response style.
- An **LLM backend** that generates the final response.
- A simple **UI** layered on top of an HTTP API server.

This README describes exactly how to:

1. Set up the environment.
2. Run the text model.
3. Run the speech model.
4. Run the fusion model.
5. Run the policy + LLM logic.
6. Start the API server and UI.


## 0. Quick Start in the project: 

From the project root (`AURA/`):

1) Create and activate virtual env (only once):

    python -m venv .venv
    source .venv/bin/activate        # macOS / Linux
    # .venv\Scripts\activate         # Windows PowerShell

3) Ensure checkpoints are in `./checkpoints/` (text + speech models).

4) Run text emotion demo:

    python run_infer.py

5) Run speech emotion demo:

    python run_speech_infer.py

6) Run fusion demo:

    python run_fusion_infer.py

7) Start LLM chatbot CLI:

    python run_llm_chat_cli.py

8) Start API server (for UI):

    python api_server.py
    # Then open the URL printed in the terminal (e.g., http://localhost:8000)

------------------------------------------------------------
## 1. Repository Overview

High-level structure:

    AURA/
    ├── __pycache__/                # Python cache (ignored)
    ├── .venv/                      # Local virtual environment (ignored)
    ├── .vscode/                    # VS Code settings
    ├── checkpoints/                # Fine-tuned model weights (text & speech)
    ├── data/                       # Data and demo audio (not in git)
    ├── fusion/                     # Fusion utilities / helper code
    ├── speech_erc/                 # Speech emotion recognition (Wav2Vec2)
    ├── static/                     # Front-end assets for the UI (HTML/JS/CSS)
    ├── text_erc/                   # Text emotion recognition (RoBERTa)
    ├── .gitignore
    ├── api_server.py               # HTTP API server (backend for UI)
    ├── asr_client.py               # ASR client (Whisper-style speech-to-text)
    ├── best_macro_f1_comparison.png
    ├── emotion_router.py           # Routes emotions to policy + LLM
    ├── llm_client.py               # LLM client wrapper
    ├── plot_metrics.py             # Script to generate evaluation plots
    ├── policy.py                   # Core policy (maps emotions → style)
    ├── response_policy.py          # Response templates / style control
    ├── run_asr_policy_demo.py      # Demo: ASR + policy
    ├── run_fusion_infer.py         # Demo: text+speech fusion
    ├── run_infer.py                # Demo: text emotion inference
    ├── run_llm_chat_cli.py         # CLI chatbot
    ├── run_llm_prompt_demo.py      # Demo: policy + LLM on fixed prompts
    ├── run_policy_demo.py          # Demo: policy logic only
    ├── run_speech_infer.py         # Demo: speech emotion inference
    ├── speech_erc_metrics.png
    ├── text_roberta_macro_f1.png
    └── text_roberta_sample_acc.png

Main entry points for running:

- Text-only emotion demo: `run_infer.py`
- Speech-only emotion demo: `run_speech_infer.py`
- Fusion demo: `run_fusion_infer.py`
- Policy-only + ASR demos: `run_policy_demo.py`, `run_asr_policy_demo.py`
- Full chatbot (CLI): `run_llm_chat_cli.py`
- HTTP API + UI backend: `api_server.py`

------------------------------------------------------------
## 2. Environment Setup

All commands assume you are in the project root (`AURA/`).

### 2.1. Create and activate virtual environment

    python -m venv .venv

Activate it:

- macOS / Linux:

    source .venv/bin/activate

- Windows (PowerShell):

    .venv\Scripts\activate

Upgrade pip:

    pip install --upgrade pip

### 2.2. Install dependencies

If a `requirements.txt` file is present:

    pip install -r requirements.txt

If not, install core packages manually, for example:

    pip install torch torchaudio transformers datasets soundfile librosa \
                scikit-learn numpy matplotlib fastapi uvicorn

You may also need `ffmpeg` at the system level (e.g., `brew install ffmpeg` on macOS) for audio.

------------------------------------------------------------
## 3. Checkpoints and Data

### 3.1. How checkpoints are created and used

1. **Text model training** (inside `text_erc/`) fine-tunes RoBERTa on GoEmotions and saves a model in:

       checkpoints/text_roberta/

2. **Speech model training** (inside `speech_erc/`) fine-tunes Wav2Vec2 on CREMA-D and saves a model in:

       checkpoints/speech_wav2vec2/

3. All inference and demo scripts in the project load these checkpoints.  
   You do **not** need to retrain models to run the demos as long as these folders exist and contain the fine-tuned weights.

If your checkpoint folder names differ, adjust the relevant constants at the top of the scripts in:

- `text_erc/`
- `speech_erc/`
- `fusion/`

### 3.2. Data folder

The `data/` directory is used for:

- Raw / preprocessed datasets (for training or analysis).
- Small demo audio files, e.g., `data/demo/example.wav`.
- Metric dumps used by `plot_metrics.py`.

For grading/demo purposes, it is sufficient to include:

- A few demo audio clips.
- Any small precomputed metric files needed by `plot_metrics.py`.

The original GoEmotions and CREMA-D datasets are not required in this repo.

------------------------------------------------------------
## 4. Step-by-Step: Text Model (RoBERTa)

We first executed the **text model** to confirm text-based emotion recognition.

### 4.1. (Optional) Training the text model

Training code is under `text_erc/`. If you want to retrain:

    cd text_erc
    python train_text_erc.py      # adjust filename if needed
    cd ..

This script:

- Loads GoEmotions or its preprocessed version from `data/`.
- Fine-tunes RoBERTa (base or large).
- Saves the trained model under `checkpoints/text_roberta/`.

### 4.2. Running text emotion inference (what we did first)

From the project root:

    python run_infer.py

This script:

1. Loads the RoBERTa checkpoint from `checkpoints/text_roberta/`.
2. Either prompts you in the terminal for a text input or uses a fixed example defined in the code.
3. Outputs:
   - The 7-way emotion probabilities for: neutral, happy, sad, angry, fear, disgust, surprise.
   - The top predicted emotion label.

You can see available options (if any) with:

    python run_infer.py --help

------------------------------------------------------------
## 5. Step-by-Step: Speech Model (Wav2Vec2)

After validating the text model, we executed the **speech model**.

### 5.1. (Optional) Training the speech model

Training code is under `speech_erc/`. To retrain the Wav2Vec2 model:

    cd speech_erc
    python train_speech_erc.py    # adjust filename if needed
    cd ..

This script:

- Loads CREMA-D (paths configured to use `data/`).
- Fine-tunes a Wav2Vec2-based classifier.
- Saves the trained model under `checkpoints/speech_wav2vec2/`.

### 5.2. Running speech emotion inference (what we did second)

From the project root:

    python run_speech_infer.py

This script:

1. Loads the Wav2Vec2 checkpoint from `checkpoints/speech_wav2vec2/`.
2. Uses a demo WAV file (e.g., under `data/demo/`) or lets you specify a path.
3. Produces:
   - The 7-way speech emotion distribution.
   - The top predicted speech emotion label.

To see options:

    python run_speech_infer.py --help

------------------------------------------------------------
## 6. Step-by-Step: Fusion Model (Text + Speech)

After both modalities were working separately, we executed the **fusion** script.

From the project root:

    python run_fusion_infer.py

This script:

1. Loads:
   - Text model from `checkpoints/text_roberta/`.
   - Speech model from `checkpoints/speech_wav2vec2/`.
2. Obtains:
   - Text emotion distribution from text input or transcript.
   - Speech emotion distribution from audio.
3. Applies late fusion (for example):

       p_fused = α * p_text + (1 - α) * p_speech

   where `α` is a weight (e.g., 0.5) configured in the script or in `fusion/`.
4. Prints:
   - Text-only emotions.
   - Speech-only emotions.
   - Fused emotion vector and final fused label.

This is exactly the stage that connects both modalities for downstream use.

------------------------------------------------------------
## 7. Policy + LLM Layer

Once fusion worked, we used the **policy** and **LLM** components to turn emotion signals into responses.

### 7.1. Policy-only demo

    python run_policy_demo.py

This script:

- Uses synthetic or pre-saved emotion vectors (e.g., “very sad”, “high anger”, “neutral”).
- Runs them through `policy.py` and `response_policy.py`.
- Displays:
  - The inferred conversation style (e.g., gentle, concise, or more elaborate).
  - Example policy-driven response templates.

This does not require the LLM or the full models and helps explain the routing logic.

### 7.2. ASR + policy demo

    python run_asr_policy_demo.py

This script:

1. Uses `asr_client.py` to transcribe speech from an audio file (Whisper-style ASR).
2. Runs:
   - Transcript → text emotion model (RoBERTa).
   - Raw audio → speech emotion model (Wav2Vec2).
3. Applies fusion as in Section 6.
4. Sends the fused emotions into the policy.
5. Prints:
   - Transcribed text.
   - Text and speech emotion distributions.
   - Final fused label and a policy-shaped response.

### 7.3. Full LLM chatbot (CLI)

This is the main “end-to-end” demo we used.

Before running, configure `llm_client.py` to point to your LLM provider and set the necessary API key, for example:

    export OPENAI_API_KEY="your_api_key_here"

Then run:

    python run_llm_chat_cli.py

What happens:

1. The script initializes:
   - Text emotion model.
   - (Optionally) speech emotion model, depending on configuration.
   - Policy and response policy.
   - LLM client.
2. It starts an interactive loop in the terminal:

       User: <type your message here>
       AURA: <responds with emotion-aware reply>

3. For each input, AURA:
   - Predicts emotions.
   - Applies policy to determine tone, directness, and verbosity.
   - Calls the LLM to produce the final response.

------------------------------------------------------------
## 8. UI and API Server

Finally, we started the **API server** and used it to back a simple UI.

### 8.1. Start the API server

From the project root:

    python api_server.py

This:

- Starts a web server (commonly at `http://localhost:8000`; see console output for exact host/port).
- Exposes endpoints such as:
  - `POST /chat/text` – accepts JSON with a text field and returns emotions + LLM response.
  - `POST /chat/audio` – accepts audio input, performs ASR + emotion recognition + fusion + policy + LLM.

Internally, the server calls:

- `emotion_router.py` to route from input → emotions → policy.
- `llm_client.py` to obtain the final LLM-generated reply.

### 8.2. Front-end UI

UI assets reside under the `static/` directory (e.g., `static/index.html`, JavaScript, CSS).

Depending on how `api_server.py` is configured, there are two typical ways to use it:

1. **Served by the API server**

   - `api_server.py` serves `static/`.
   - After starting the server, open the URL printed in the terminal, such as:

         http://localhost:8000
         # or http://localhost:8000/index.html

   - The page exposes:
     - A text input box for user messages.
     - Optional upload/record controls for audio.
     - A conversation area showing detected emotions and AURA’s replies.

2. **Open HTML directly**

   - Open `static/index.html` in a browser.
   - Ensure the JavaScript in that file points to the correct API base URL
     (e.g., `http://localhost:8000`) so it can call the backend.

------------------------------------------------------------
## 9. Metrics and Plots

To regenerate the figures used in the project report:

    python plot_metrics.py

This script:

- Reads metric files (e.g., JSON/CSV) from `data/`.
- Produces PNG figures, including:

  - `best_macro_f1_comparison.png`
  - `speech_erc_metrics.png`
  - `text_roberta_macro_f1.png`
  - `text_roberta_sample_acc.png`

These plots summarize:

- Macro-F1 comparisons for different settings.
- Per-emotion performance of the speech model.
- Overall performance of the text model.

------------------------------------------------------------
## 10. Troubleshooting

- **Missing checkpoints**

  Ensure:

  - `checkpoints/text_roberta/` exists with a valid RoBERTa checkpoint.
  - `checkpoints/speech_wav2vec2/` exists with a valid Wav2Vec2 checkpoint.
  - Paths in `text_erc/`, `speech_erc/`, and `fusion/` match your actual directories.

- **Import or dependency errors**

  - Confirm the virtual environment is activated.
  - Re-run:

        pip install -r requirements.txt

- **ASR issues**

  - Verify configuration inside `asr_client.py` (model name, device).
  - Check that `ffmpeg` is installed if audio loading fails.

- **LLM errors**

  - Confirm environment variable (e.g., `OPENAI_API_KEY`) is set.
  - Ensure the model name in `llm_client.py` is valid for your provider.

- **UI not loading**

  - Confirm `api_server.py` is running without errors.
  - Use the exact URL printed to the terminal.
  - If opening `static/index.html` directly, make sure the JS configuration points to your API base URL.

With these steps, a new user or grader can reproduce exactly what we did:

1. Run text model.
2. Run speech model.
3. Run fusion model.
4. Run policy and ASR demos.
5. Start the CLI chatbot.
6. Launch the HTTP API and UI on top of it.
