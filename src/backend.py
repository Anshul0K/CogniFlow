import os
import joblib
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI

from src.feature_collector import FeatureCollector
from src.interaction_features import InteractionTracker


load_dotenv()

app = FastAPI()


# -------------------------
# Gemini
# -------------------------

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# -------------------------
# ML model
# -------------------------

model = joblib.load(
    "models/cognitive_load_model.pkl"
)


labels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


# -------------------------
# Feature collector
# -------------------------

collector = FeatureCollector()


# -------------------------
# Interaction tracker
# -------------------------

tracker = InteractionTracker()


# -------------------------
# Request
# -------------------------

class ChatRequest(BaseModel):
    prompt: str


# -------------------------
# Startup
# -------------------------

@app.on_event("startup")
def startup():

    import threading

    thread = threading.Thread(
        target=collector.start,
        daemon=True
    )

    thread.start()

    print("CogniFlow backend started.")


# -------------------------
# Chat
# -------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    # Get webcam + keyboard features

    features = collector.get_features()


    # Get interaction features

    interaction = tracker.record_interaction(
        prompt=request.prompt
    )


    # Combine everything

    features.update(interaction)


    # Make sure all 21 features exist

    feature_order = [

        "blink_rate_per_min",
        "mean_eye_openness",
        "eye_openness_std",
        "gaze_movement_rate",
        "head_movement_rate",
        "head_pose_variance",

        "typing_speed_wpm",
        "mean_key_hold_ms",
        "mean_interkey_interval_ms",
        "interkey_interval_std",
        "typing_pause_count",
        "mean_pause_duration_ms",
        "backspace_rate",
        "typing_variability",

        "response_time_sec",
        "prompt_length_words",
        "edit_count",
        "revision_rate",
        "interaction_frequency",
        "time_since_last_interaction_sec",
        "AI_suggestion_interaction_rate"
    ]


    values = [[
        features.get(name, 0)
        for name in feature_order
    ]]


    # Predict cognitive load

    prediction = model.predict(values)[0]

    cognitive_load = labels[prediction]


    # -------------------------
    # Adaptation policy
    # -------------------------

    if cognitive_load == "HIGH":

        instruction = """
The user currently has HIGH cognitive load.

Give a concise and simple response.
Use short sentences.
Use step-by-step explanations.
Avoid unnecessary details.
Focus only on what is needed.
"""

    elif cognitive_load == "MEDIUM":

        instruction = """
The user currently has MEDIUM cognitive load.

Give a moderately detailed response.
Use clear structure and short sections.
Include useful examples when appropriate.
"""

    else:

        instruction = """
The user currently has LOW cognitive load.

Give a detailed technical response.
Explain the reasoning clearly.
Include examples and implementation details when useful.
"""


    # -------------------------
    # Gemini
    # -------------------------

    start_time = time.time()

    response = llm.invoke(
        instruction +
        "\n\nUser question:\n" +
        request.prompt
    )

    response_time = time.time() - start_time

    tracker.update_response_time(response_time)


    return {
        "response": response.content,
        "cognitive_load": cognitive_load,
        "features": features
    }