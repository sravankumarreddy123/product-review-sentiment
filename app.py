
import os
import re
import html
import urllib.request

import torch
import gradio as gr

from torch import nn
from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

MODEL_PATH = "multitask_model.pt"

MODEL_URL = (
    "https://github.com/sravankumarreddy123/"
    "product-review-sentiment/releases/download/"
    "v1.0.0/multitask_model.pt"
)

DEVICE = torch.device("cpu")

SENT_LABELS = [
    "Negative",
    "Neutral",
    "Positive"
]

EMO_LABELS = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Disgust",
    "Surprise"
]

# Replace these with the exact aspect labels
# from your notebook if they are different.
ASP_LABELS = [
    "Aspect class 0",
    "Aspect class 1",
    "Aspect class 2"
]

tokenizer = None
model = None


# ==========================================
# 2. MULTI-TASK DISTILBERT MODEL
# ==========================================

class MultiTaskModel(nn.Module):

    def __init__(self, name):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(name)

        h = self.encoder.config.hidden_size

        self.drop = nn.Dropout(0.1)

        self.heads = nn.ModuleDict({

            "sent": nn.Linear(h, 3),

            "emo": nn.Linear(h, 6),

            "asp": nn.Linear(h, 3)

        })

    def forward(self, input_ids, attention_mask, task="sent"):

        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        return self.heads[task](self.drop(cls))


# ==========================================
# 3. DOWNLOAD TRAINED MODEL
# ==========================================

def download_model():

    if os.path.exists(MODEL_PATH):

        print("Trained model already exists.")

        return

    print("Downloading trained model from GitHub...")

    try:

        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "Product-Review-App"}
        )

        with urllib.request.urlopen(request, timeout=300) as response:

            with open(MODEL_PATH, "wb") as file:

                while True:

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)

        print("Model downloaded successfully.")

    except Exception:

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        raise RuntimeError(
            "Unable to download multitask_model.pt. "
            "Check the GitHub Release URL."
        )


# ==========================================
# 4. LOAD TOKENIZER AND MODEL
# ==========================================

def load_model():

    global tokenizer, model

    if model is not None:
        return

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    download_model()

    print("Loading trained model...")

    model = MultiTaskModel(MODEL_NAME)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully!")


# ==========================================
# 5. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = html.unescape(str(text))

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 6. PREDICTION FUNCTION
# ==========================================

def predict_review(review, task):

    if not review or not review.strip():

        return "Please enter a product review."

    try:

        load_model()

        review = clean_text(review)

        inputs = tokenizer(
            review,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        task_map = {

            "Sentiment": ("sent", SENT_LABELS),

            "Emotion": ("emo", EMO_LABELS),

            "Aspect": ("asp", ASP_LABELS)

        }

        task_name, labels = task_map[task]

        with torch.no_grad():

            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task=task_name
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            predicted_index = torch.argmax(
                probabilities
            ).item()

            confidence = (
                probabilities[predicted_index].item() * 100
            )

        predicted_label = labels[predicted_index]

        return (
            f"Prediction: {predicted_label}\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as error:

        return f"Error: {str(error)}"


# ==========================================
# 7. GRADIO USER INTERFACE
# ==========================================

with gr.Blocks(
    title="Product Review Sentiment Analysis",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Product Review Sentiment Analysis

        ### Multi-Task DistilBERT Model

        Analyze product reviews using Artificial Intelligence.

        Select a task to predict sentiment, emotion, or aspect class.
        """
    )

    review_input = gr.Textbox(
        label="Enter Product Review",
        placeholder=(
            "Example: This phone has excellent battery "
            "life and a very good camera."
        ),
        lines=5
    )

    task_input = gr.Dropdown(
        choices=[
            "Sentiment",
            "Emotion",
            "Aspect"
        ],
        value="Sentiment",
        label="Select Prediction Task"
    )

    predict_button = gr.Button(
        "Analyze Review",
        variant="primary"
    )

    output = gr.Textbox(
        label="Analysis Result",
        lines=3
    )

    predict_button.click(
        fn=predict_review,
        inputs=[
            review_input,
            task_input
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            [
                "This phone has excellent battery life.",
                "Sentiment"
            ],
            [
                "I am very happy with this product!",
                "Emotion"
            ],
            [
                "The camera quality is poor.",
                "Aspect"
            ]
        ],
        inputs=[
            review_input,
            task_input
        ]
    )


# ==========================================
# 8. START APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
