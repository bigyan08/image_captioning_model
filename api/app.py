import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.utils import Vocabulary
from PIL import Image
import torch
from torchvision import transforms
from src.model import CNNtoRNN
import pickle
import src.config as cfg

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# load vocab once
vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebook/models/flickr30k_vocab.pkl')

class FixPickle(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            from src.utils import Vocabulary
            return Vocabulary
        return super().find_class(module, name)

with open(vocab_path, 'rb') as f:
    vocab = FixPickle(f).load()

# load model once
model = CNNtoRNN(cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, len(vocab), cfg.NUM_LAYERS).to(cfg.DEVICE)
model.load_state_dict(torch.load("../notebook/models/final_model_30k.pth", map_location="cpu"))
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(356),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(cfg.DEVICE)
    #generating captions
    with torch.no_grad():
        caption_tokens = model.caption_image(image_tensor,vocab)
    caption = ' '.join(caption_tokens)
    return {"caption": caption}
