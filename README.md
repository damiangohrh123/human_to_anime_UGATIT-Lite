# Human → Anime Face Translator (UGATIT)
Convert real human faces into anime-style faces using a lightweight CycleGAN (UGATIT-Lite) model.

## 🚀 Demo
Try the live demo here: [Hugging Face Space Link](https://huggingface.co/spaces/damiangohrh123/human_to_anime_UGATIT-Lite)  
Upload a face, and the model generates an anime-style version instantly.

## 💻 Features
Human → Anime conversion
CycleGAN (UGATIT) architecture for lightweight, fast inference
Download the generated anime face

## 📦 Installation (for local use)
### Clone the repository
`git clone https://github.com/YOUR_GITHUB_USERNAME/human-to-anime.git`
`cd human-to-anime`

### Install dependencies
`pip install -r requirements.txt`

### Launch the Gradio app
`python app.py`

## ⚙️ Usage
Upload a human face image.
(Optional) Toggle “Quality Mode” for EMA-based output.
View or download the anime-styled image.

## 🧠 Model
Generator: UGATIT
Discriminator: PatchGAN
Input/Output: 256×256 RGB images
Training: Human and anime face datasets
