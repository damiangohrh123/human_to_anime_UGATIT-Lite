# app.py
"""
Human -> Anime (UGATIT-Lite) Gradio app for Hugging Face Spaces.
Assumes:
 - model.py exists and defines Generator
 - checkpoints/latest.pth exists and contains "G_A2B" (state_dict)
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
from model import Generator

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = "checkpoints/latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 128   # your trained generator produces 128x128 outputs

# -----------------------------
# Utilities
# -----------------------------
def safe_load_state_dict(model, state_dict):
    """Remove 'module.' prefix if present (from DataParallel) and load."""
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v
    model.load_state_dict(new_state)

def tensor_to_pil(tensor_img):
    """Convert a [-1,1] torch tensor (C,H,W) to PIL Image."""
    t = tensor_img.detach().cpu().clamp(-1, 1)
    t = (t * 0.5 + 0.5)  # to [0,1]
    t = (t * 255).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(t)

# -----------------------------
# Load Generator
# -----------------------------
G_A2B = None
try:
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    G_A2B = Generator().to(DEVICE)
    # ckpt could be the entire checkpoint dict or a state_dict
    if "G_A2B" in ckpt:
        safe_load_state_dict(G_A2B, ckpt["G_A2B"])
    else:
        safe_load_state_dict(G_A2B, ckpt)
    G_A2B.eval()
    print("Loaded generator from", CHECKPOINT_PATH)
except Exception as e:
    print("Failed to load generator:", e)
    G_A2B = None

# -----------------------------
# Preprocess / Postprocess
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def prepare_image(img_pil):
    """PIL -> normalized tensor on device"""
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    t = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    return t

# -----------------------------
# Inference
# -----------------------------
def run_inference(img: Image.Image, quality_mode: bool):
    """
    img: PIL image from Gradio
    quality_mode: bool (EMA toggle, currently uses same weights unless you supply EMA checkpoint)
    Returns: PIL image (anime)
    """
    if G_A2B is None:
        return None, "Model not loaded. Check logs."

    # convert and forward
    x = prepare_image(img)
    with torch.no_grad():
        out = G_A2B(x)
        # many UGATIT/Generator impls return (img, cam_logit); keep first element
        if isinstance(out, tuple) or isinstance(out, list):
            out = out[0]
        out = out.squeeze(0).cpu()
    pil = tensor_to_pil(out)
    return pil, "OK"

# -----------------------------
# Example images (if present)
# -----------------------------
examples = []
if os.path.isdir("examples"):
    for fname in sorted(os.listdir("examples")):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            examples.append([os.path.join("examples", fname), False])
# limit examples for UI
examples = examples[:6]

# -----------------------------
# Build Gradio Blocks UI
# -----------------------------
title = "Human → Anime (UGATIT-Lite)"
description = (
    "Upload a human face and get an anime-stylized version. "
    "Quality Mode uses EMA weights if available (currently uses base weights)."
)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload Photo (or take a webcam shot)", type="pil")
            quality = gr.Checkbox(label="Quality Mode (EMA)", value=False)
            btn = gr.Button("Generate")
            gr.Examples(examples=examples, inputs=[inp, quality]) if examples else None

            gr.Markdown("**Tips:** Best results with frontal faces. Use a clean, well-lit photo.")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Anime Output", type="pil")
            status = gr.Textbox(value="Ready", label="Status", interactive=False)
            download_btn = gr.DownloadButton(label="Download Result", file=None)

    def generate_and_return(img, q):
        pil, msg = run_inference(img, q)
        if pil is None:
            status_txt = f"Error: {msg}"
            return None, status_txt, None
        # prepare a bytes IO for download (gradio DownloadButton will accept bytes or file path)
        import io
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return pil, "OK", buf

    btn.click(fn=generate_and_return, inputs=[inp, quality], outputs=[out_img, status, download_btn])

    gr.Markdown("---")
    gr.Markdown("### About\nThis demo uses a UGATIT-Lite generator to convert real human faces into anime-style portraits. The model was trained locally and loaded from `checkpoints/latest.pth`.")

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)