import os
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY")
)

model = joblib.load("models/cognitive_load_model.pkl")


class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat")
def chat(request: ChatRequest):

    # temporary test features
    features = {
        "blink_rate_per_min": 15,
        "mean_eye_openness": 0.25,
        "eye_openness_std": 0.04,
        "gaze_movement_rate": 0.01,
        "head_movement_rate": 0.01,
        "head_pose_variance": 0.001,
        "typing_speed_wpm": 45,
        "mean_key_hold_ms": 70,
        "mean_interkey_interval_ms": 200,
        "interkey_interval_std": 100,
        "typing_pause_count": 2,
        "mean_pause_duration_ms": 1500,
        "backspace_rate": 0.05,
        "typing_variability": 1.0,
        "response_time_sec": 5,
        "prompt_length_words": 10,
        "edit_count": 2,
        "revision_rate": 0.5,
        "interaction_frequency": 5,
        "time_since_last_interaction_sec": 20,
        "AI_suggestion_interaction_rate": 0.5
    }

    load = model.predict([list(features.values())])[0]

    labels = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }

    cognitive_load = labels[load]

    if cognitive_load == "HIGH":
        instruction = """
        The user has high cognitive load.
        Give a concise and simple answer.
        Use short sentences and step-by-step explanations.
        Avoid unnecessary details.
        """
    elif cognitive_load == "MEDIUM":
        instruction = """
        The user has medium cognitive load.
        Give a moderately detailed and structured answer.
        """
    else:
        instruction = """
        The user has low cognitive load.
        Give a detailed technical explanation with examples.
        """

    response = llm.invoke(
        instruction + "\n\nUser question:\n" + request.prompt
    )

    return {
        "cognitive_load": cognitive_load,
        "response": response.content
    }