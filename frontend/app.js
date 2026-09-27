const chatBox = document.getElementById("chat-box");
const promptInput = document.getElementById("prompt");
const sendButton = document.getElementById("send-btn");

const loadText = document.getElementById("load-text");
const loadValue = document.getElementById("load-value");
const loadDot = document.getElementById("load-dot");


function addMessage(sender, text) {

    const message = document.createElement("div");

    message.className =
        sender === "You"
            ? "message user"
            : "message assistant";

    message.innerHTML = `
        <div class="message-label">
            ${sender}
        </div>

        <div class="message-content">
            ${text}
        </div>
    `;

    chatBox.appendChild(message);

    chatBox.scrollTop = chatBox.scrollHeight;
}


function updateLoad(load) {

    loadText.textContent = load;
    loadValue.textContent = load;

    if (load === "LOW") {
        loadDot.style.background = "#22c55e";
    }

    else if (load === "MEDIUM") {
        loadDot.style.background = "#f59e0b";
    }

    else if (load === "HIGH") {
        loadDot.style.background = "#ef4444";
    }
}


function updateFeatures(features) {

    document.getElementById("blink-rate").textContent =
        features.blink_rate_per_min.toFixed(1);

    document.getElementById("eye-openness").textContent =
        features.mean_eye_openness.toFixed(3);

    document.getElementById("gaze-movement").textContent =
        features.gaze_movement_rate.toFixed(4);

    document.getElementById("typing-speed").textContent =
        features.typing_speed_wpm.toFixed(1) + " WPM";

    document.getElementById("pause-count").textContent =
        features.typing_pause_count;

    document.getElementById("backspace-rate").textContent =
        features.backspace_rate.toFixed(2);

    document.getElementById("response-time").textContent =
        features.response_time_sec.toFixed(2) + " s";

    document.getElementById("interaction-rate").textContent =
        features.interaction_frequency.toFixed(2) + " / min";
}


async function sendMessage() {

    const prompt = promptInput.value.trim();

    if (!prompt) {
        return;
    }

    addMessage("You", prompt);

    promptInput.value = "";

    sendButton.disabled = true;
    sendButton.textContent = "Thinking...";

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/chat",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    prompt: prompt
                })
            }
        );

        if (!response.ok) {
            throw new Error("Server error");
        }

        const data = await response.json();

        addMessage(
            "CogniFlow",
            data.response
        );

        updateLoad(
            data.cognitive_load
        );

        updateFeatures(
            data.features
        );

    }

    catch (error) {

        console.error(error);

        addMessage(
            "CogniFlow",
            "Sorry, something went wrong."
        );

    }

    finally {

        sendButton.disabled = false;
        sendButton.textContent = "Send";

        promptInput.focus();
    }
}


sendButton.addEventListener(
    "click",
    sendMessage
);


promptInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            sendMessage();
        }

    }
);