### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["pizza", "steak", "sushi"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=3, # len(class_names) would also work
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
import gradio as gr

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0f0d0b;
    --surface: #1a1714;
    --card: #221f1b;
    --border: #2e2a25;
    --accent: #e8602c;
    --accent2: #f5a623;
    --text: #f0ebe4;
    --muted: #8a8278;
    --radius: 16px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

footer { display: none !important; }

#header {
    background: linear-gradient(135deg, #1a1714 0%, #2a1f14 50%, #1a1714 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(232,96,44,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
#header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: var(--text) !important;
    margin: 0 0 8px !important;
    line-height: 1.1 !important;
}
#header p {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    background: rgba(232,96,44,0.15);
    color: var(--accent);
    border: 1px solid rgba(232,96,44,0.3);
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    margin-bottom: 16px;
}

.section-label {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 10px !important;
}

.panel {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.upload-box .wrap {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    min-height: 260px !important;
    transition: border-color 0.2s ease;
}
.upload-box .wrap:hover { border-color: var(--accent) !important; }

#predict-btn {
    background: linear-gradient(135deg, var(--accent) 0%, #c94e20 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(232,96,44,0.3) !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
#predict-btn:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

#clear-btn {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    padding: 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s !important;
}
#clear-btn:hover { border-color: var(--muted) !important; color: var(--text) !important; }

.label-output .bar {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border-radius: 4px !important;
    height: 8px !important;
}

.timing-box input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--accent2) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
    text-align: center !important;
    padding: 14px !important;
}

#footer-strip {
    margin-top: 20px;
    padding: 16px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 0.82rem;
    color: var(--muted);
}
#footer-strip a { color: var(--accent2) !important; text-decoration: none; }
"""

with gr.Blocks(css=custom_css, title="FoodVision Mini") as demo:

    gr.HTML("""
    <div id="header">
        <span style="font-size:2rem;display:block;margin-bottom:12px">🍕 🥩 🍣</span>
        <div class="badge">EfficientNetB2 · Computer Vision</div>
        <h1>FoodVision Mini</h1>
        <p>Drop any food photo and the model will classify it as pizza, steak, or sushi — with confidence scores and inference timing.</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<p class="section-label">📸 Upload Image</p>')
            image_input = gr.Image(type="pil", label="", show_label=False, elem_classes="upload-box")
            gr.HTML('<p class="section-label" style="margin-top:16px">⚡ Quick Examples</p>')
            gr.Examples(examples=example_list, inputs=image_input)

        with gr.Column(scale=1):
            with gr.Row():
                clear_btn   = gr.Button("✕  Clear",   elem_id="clear-btn")
                predict_btn = gr.Button("Classify →", elem_id="predict-btn")

            gr.HTML('<p class="section-label" style="margin-top:20px">📊 Predictions</p>')
            label_output = gr.Label(num_top_classes=3, label="", show_label=False, elem_classes="label-output panel")

            gr.HTML('<p class="section-label" style="margin-top:16px">⏱ Inference Time</p>')
            time_output = gr.Number(label="", show_label=False, elem_classes="timing-box panel", precision=4)

    gr.HTML("""
    <div id="footer-strip">
        📖 &nbsp;Tutorial: <a href="https://www.learnpytorch.io/09_pytorch_model_deployment/" target="_blank">
        08. PyTorch Model Deployment — learnpytorch.io</a>
    </div>
    """)

    predict_btn.click(fn=predict, inputs=image_input, outputs=[label_output, time_output])
    clear_btn.click(fn=lambda: (None, None, None), inputs=[], outputs=[image_input, label_output, time_output])

demo.launch()
