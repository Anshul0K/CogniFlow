# CogniFlow

### A Closed-Loop Framework for Cognitive-Load-Aware Human–LLM Interaction

CogniFlow is a research prototype that estimates a user's cognitive load from **multimodal interaction signals** and dynamically adapts the way an LLM responds.

Instead of changing or fine-tuning the underlying language model, CogniFlow introduces an **adaptive interaction layer** between the user and the LLM.

The system continuously observes:
- Webcam-based facial signals
- Keyboard interaction patterns
- Human–LLM interaction behavior

These signals are combined to estimate cognitive load as:
- **LOW**
- **MEDIUM**
- **HIGH**

The estimated state is then used to control the response style of the LLM.

---

## System Overview

```text
                    ┌─────────────────────┐
                    │       Webcam        │
                    │ Facial Features     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Feature Fusion    │
                    │                     │
                    │ 21 Multimodal       │
                    │ Features            │
                    └──────────┬──────────┘
                               ▲
                               │
                    ┌──────────┴──────────┐
                    │                     │
             ┌──────┴──────┐     ┌───────┴────────┐
             │  Keyboard   │     │  Interaction   │
             │  Features   │     │    Features   │
             └─────────────┘     └────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      XGBoost        │
                    │ Cognitive Load      │
                    │ Classification      │
                    └──────────┬──────────┘
                               │
                         LOW / MEDIUM / HIGH
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Adaptation Policy   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Gemini        │
                    │       LLM           │
                    └──────────┬──────────┘
                               │
                               ▼
                         Adaptive Response
                               │
                               ▼
                             User
                               │
                               └──────→ Next interaction
```

### Core Idea
CogniFlow follows a closed-loop interaction model:

`Lₜ → π(Lₜ, Cₜ) → Rₜ → LLM → User → Lₜ₊₁`

Where:
* **Lₜ** = estimated cognitive load at time t
* **Cₜ** = current interaction context
* **π** = response adaptation policy
* **Rₜ** = selected response style

The system therefore adapts the interaction strategy, rather than modifying the underlying LLM.

---

## Response Adaptation
CogniFlow currently uses three response modes:

### LOW Cognitive Load
The system allows more detailed responses.
* Detailed explanations
* Technical reasoning
* Examples
* Implementation details

### MEDIUM Cognitive Load
The system provides moderately detailed responses.
* Structured explanations
* Short sections
* Useful examples
* Reduced unnecessary detail

### HIGH Cognitive Load
The system reduces information density.
* Concise responses
* Short sentences
* Step-by-step explanations
* Only essential information
* Progressive disclosure

---

## Multimodal Features
CogniFlow currently uses 21 features.

**Webcam Features** (Extracted using MediaPipe Face Mesh):
* `blink_rate_per_min`
* `mean_eye_openness`
* `eye_openness_std`
* `gaze_movement_rate`
* `head_movement_rate`
* `head_pose_variance`

**Keyboard Features** (Captured using pynput):
* `typing_speed_wpm`
* `mean_key_hold_ms`
* `mean_interkey_interval_ms`
* `interkey_interval_std`
* `typing_pause_count`
* `mean_pause_duration_ms`
* `backspace_rate`
* `typing_variability`

**Interaction Features** (Collected by the FastAPI backend):
* `response_time_sec`
* `prompt_length_words`
* `edit_count`
* `revision_rate`
* `interaction_frequency`
* `time_since_last_interaction_sec`
* `AI_suggestion_interaction_rate`

---

## Machine Learning Model
CogniFlow uses an XGBoost classifier for cognitive-load estimation.

**Model Configuration:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
```
The train/test split is performed at the user level using `GroupShuffleSplit` to avoid placing data from the same user in both training and testing sets.

### Prototype Results
The current prototype model achieved approximately:

| Metric | Result |
| :--- | :--- |
| **Accuracy** | 78.7% |
| **Balanced Accuracy** | 78.2% |
| **Macro F1** | 0.78 |

**Confusion Matrix:**
```text
                Predicted
              LOW  MED  HIGH
Actual LOW   1431  243    4
Actual MED    198  985  200
Actual HIGH     2  224  800
```
*Note: The MEDIUM class is the most difficult class to distinguish, which is expected because it represents an intermediate state between lower and higher cognitive load.*

---

## ⚠️ Important Dataset Note
> **The current multimodal training dataset is synthetically generated using distributions and feature structures informed by public datasets.**

Public datasets were used for modality-specific feature design because a directly suitable publicly available dataset containing synchronized **webcam signals**, **keystroke dynamics**, **human–LLM interaction signals**, and **cognitive-load annotations** was not available for this prototype.

**Therefore, the reported model performance should be interpreted as prototype/model-pipeline validation, not as evidence of real-world cognitive-load detection accuracy.** Real-user validation is planned as future work.

---

## Technology Stack
* **Backend:** Python, FastAPI, Uvicorn, Pydantic
* **Machine Learning:** XGBoost, Scikit-learn, NumPy, Pandas, Joblib
* **Computer Vision:** OpenCV, MediaPipe Face Mesh
* **Interaction Sensing:** pynput
* **LLM:** Google Gemini, LangChain
* **Frontend:** HTML, CSS, JavaScript

---

## Project Structure
```text
CogniFlow/
│
├── data/
│   ├── cognitive_load/
│   ├── keystroke/
│   └── coauthor/
│
├── models/
│   └── cognitive_load_model.pkl
│
├── notebooks/
│   └── 01_cognitive_load_model.ipynb
│
├── src/
│   ├── backend.py
│   ├── webcam_features.py
│   ├── keyboard_features.py
│   ├── interaction_features.py
│   └── feature_collector.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── results/
│
├── .env
├── .gitignore
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Anshul0K/CogniFlow.git
cd CogniFlow
```

**2. Create the Conda environment**
```bash
conda create -n cogniflow python=3.11
conda activate cogniflow
```

**3. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter opencv-python mediapipe==0.10.21 pynput fastapi uvicorn python-dotenv langchain-google-genai
```

**macOS Users:** If using Apple Silicon and OpenCV requires OpenMP:
```bash
brew install libomp
```

### Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```
*Never commit `.env` or expose your API key publicly.*

---

## Running CogniFlow
CogniFlow currently runs using two processes.

### Terminal 1 — Backend
From the project root:
```bash
conda activate cogniflow
uvicorn src.backend:app
```
The backend runs at: `http://127.0.0.1:8000`

The backend starts:
* Webcam feature collection
* Keyboard feature collection
* Interaction tracking
* XGBoost inference
* Gemini interaction

### Terminal 2 — Frontend
Open another terminal:
```bash
cd frontend
conda activate cogniflow
python -m http.server 5500
```
Open: `http://127.0.0.1:5500`

---

## API

### `POST /chat`

**Request:**
```json
{
    "prompt": "Explain binary lifting simply"
}
```

**Response:**
```json
{
    "response": "Binary lifting is...",
    "cognitive_load": "LOW",
    "features": {
        "blink_rate_per_min": 5.9,
        "mean_eye_openness": 0.44,
        "typing_speed_wpm": 17.2,
        "typing_pause_count": 1,
        "response_time_sec": 4.33,
        "interaction_frequency": 2.24
    }
}
```

---

## Adaptation Pipeline
A typical interaction follows:

1. User opens CogniFlow
2. Webcam starts collecting facial features
3. Keyboard listener collects keystroke features
4. User enters a prompt
5. Backend collects interaction features
6. 21 features are combined
7. XGBoost predicts cognitive load
8. Adaptation policy selects response style
9. Policy instruction is passed to Gemini
10. Gemini generates the response
11. Response + cognitive state are shown in UI
12. Next interaction begins

### Example
**User:** *"Explain binary lifting simply."*

* If the estimated state is **LOW**, CogniFlow may request a detailed explanation with examples.
* If the estimated state is **HIGH**, the adaptation layer instead requests:
  * concise explanation
  * short sentences
  * step-by-step structure
  * minimal unnecessary information

*The underlying Gemini model remains unchanged.*

---

## Research Motivation
Traditional LLM interfaces generally treat every interaction similarly regardless of the user's current cognitive state. However, the same amount of information can have different effects depending on the user's current level of cognitive load. 

CogniFlow explores a closed-loop approach where the interaction itself becomes part of the adaptive system:

`User State` → `Sensing` → `Cognitive Load Estimation` → `Response Adaptation` → `LLM Response` → `New User State`

The goal is to investigate whether multimodal cognitive-state estimation can be used to make human–AI interaction more context-sensitive and information-efficient.

---

## Current Limitations
* The current training dataset is synthetic.
* Real-world cognitive-load validation has not yet been performed.
* Webcam features are currently limited to facial landmarks and basic movement/eye metrics.
* Keyboard features depend on OS-level input permissions.
* Some interaction features such as edit/revision tracking are currently simplified.
* Cognitive load is estimated in discrete LOW/MEDIUM/HIGH classes.
* The adaptation policy is currently rule-based rather than learned.
* The system is currently designed as a research prototype rather than a production application.

---

## Future Work

1. **Real User Study:** Collect synchronized Webcam + Keyboard + Interaction logs + Self-reported cognitive load from real participants.
2. **Learned Adaptation Policy:** Replace the rule-based policy with a learned policy or contextual bandit.
3. **Temporal Modeling:** Instead of treating each interaction independently, model cognitive-load trajectories over time (e.g., LSTM, Temporal CNN, Transformer, Hidden Markov Models).
4. **Personalization:** Adapt the model to individual typing and interaction baselines.
5. **More Adaptive Dimensions:** The system could adapt response length, explanation depth, number of examples, terminology complexity, step-by-step guidance, progressive disclosure, and visual structure.

---

## Research Contribution
CogniFlow focuses on the interaction layer between a human and an LLM. 
The key idea is: **Estimate cognitive state from multimodal interaction signals and use that state to dynamically adapt the interaction strategy of an LLM.**

Rather than modifying the language model itself, CogniFlow creates a cognitive-state-conditioned response policy around the model.

---

## License
This project is intended for research and educational purposes.