import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# Import the EXACT generator used during training
from train_ugatit import Generator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
G = Generator().to(device)

state_dict = torch.load("model/G_A2B.pth", map_location=device)
G.load_state_dict(state_dict, strict=True)

G.eval()
torch.set_grad_enabled(False)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Inference function
def infer(image: Image.Image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_B, _ = G(x)  # UGATIT generator returns (image, cam_logit)

    # [-1,1] → [0,1]
    fake_B = (fake_B * 0.5 + 0.5).clamp(0, 1)

    fake_B = fake_B.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return fake_B

# Gradio UI
demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Human Face"),
    outputs=gr.Image(type="numpy", label="Anime Output"),
    title="UGATIT-Lite: Human → Anime",
    description=(
        "UGATIT-Lite human-to-anime face translation model trained at 512×512 resolution.\n\n"
        "Note: Identity is partially preserved, while anime stylization is still improving."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()