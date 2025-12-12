import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import Generator  # your UGATIT Generator class

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
checkpoint = torch.load("checkpoints/latest.pth", map_location=device)
G_A2B = Generator().to(device)
G_A2B.load_state_dict(checkpoint["G_A2B"])
G_A2B.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def tensor_to_pil(tensor_img):
    img = tensor_img.detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0,1)
    return transforms.ToPILImage()(img)

def convert_to_anime(image, use_ema=False):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = G_A2B(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        output = output.squeeze(0)
    return tensor_to_pil(output)

interface = gr.Interface(
    fn=convert_to_anime,
    inputs=[gr.Image(type="pil"), gr.Checkbox(label="Quality Mode (EMA)")],
    outputs=gr.Image(type="pil"),
    title="Human → Anime (UGATIT-Lite)",
    description="Upload a real face and get an anime-style version."
)

if __name__ == "__main__":
    interface.launch()
