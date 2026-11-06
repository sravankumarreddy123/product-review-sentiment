
import os
import re
import html
import torch
import gradio as gr

from torch import nn
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Configuration
# -----------------------------

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

MODEL_PATH = "multitask_model.pt"

SENT_LABELS = ["Negative", "Neutral", "Positive"]

EMO_LABELS = [
    "Joy", "Sadness", "Anger",
    "Fear", "Disgust", "Surprise"
]

# The aspect head has 3 classes.
# Replace these names if your notebook defines different aspect labels.
ASP_LABELS = ["Aspect class 0", "Aspect class 1", "Aspect class 2"]

DEVICE = torch.device("cpu")

tokenizer = None
model = None


# -----------------------------
# 2. Model architecture
# -----------------------------

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


# -----------------------------
# 3. Load tokenizer and model
# -----------------------------

def load_model():

    global tokenizer, model

    if model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file multitask_model.pt was not found. "
            "Please make the trained model file available "
            "in the application's root directory."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = MultiTaskModel(MODEL_NAME)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()


# -----------------------------
# 4. Text cleaning
# -----------------------------

def clean_text(text):

    text = html.unescape(str(text))

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"http\S+|www\.\S+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 5. Prediction function
# -----------------------------

def predict_review(review, task):

    if not review or not review.strip():
        return "Please enter a review."

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

            probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = torch.argmax(probabilities).item()

            confidence = probabilities[predicted_index].item() * 100

        predicted_label = labels[predicted_index]

        return (
            f"Prediction: {predicted_label}\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    except FileNotFoundError as error:
        return f"Model file error: {error}"

    except Exception as error:
        return f"Prediction error: {str(error)}"


# -----------------------------
# 6. Gradio interface
# -----------------------------

with gr.Blocks(title="Product Review Sentiment Analysis") as demo:

    gr.Markdown(
        """
        # Product Review Sentiment Analysis

        Analyze product reviews using a Multi-Task DistilBERT model.

        Enter a product review and select the prediction task.
        """
    )

    review_input = gr.Textbox(
        label="Enter Product Review",
        placeholder="Example: This phone has excellent battery life...",
        lines=5
    )

    task_input = gr.Dropdown(
        choices=["Sentiment", "Emotion", "Aspect"],
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
        inputs=[review_input, task_input],
        outputs=output
    )


# -----------------------------
# 7. Start application
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
