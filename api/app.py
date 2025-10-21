import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils import Vocabulary
from PIL import Image
import torch
from torchvision import transforms
from src.model import CNNtoRNN
import pickle
import src.config as cfg
import os
from azure.storage.blob import BlobServiceClient
from fastapi.responses import FileResponse

def download_model_if_needed():
    model_dir = "notebook/models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "final_model_30k.pth")
    vocab_path = os.path.join(model_dir, "flickr30k_vocab.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        print("Downloading model from Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
        container_name = "models"

        blobs = {
            "final_model_30k.pth": model_path,
            "flickr30k_vocab.pkl": vocab_path,
        }

        for blob_name, file_path in blobs.items():
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(file_path, "wb") as f:
                data = blob_client.download_blob()
                f.write(data.readall())

        print("✅ Model files downloaded successfully")

# Call before loading model

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://image-captioning.azurewebsites.net"], 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

download_model_if_needed()
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
model.load_state_dict(torch.load("notebook/models/final_model_30k.pth", map_location="cpu"))
model.eval()

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    origin = request.headers.get("origin") or request.headers.get("Origin")
    if origin !="https://image-captioning.azurewebsites.net":
        raise HTTPException(status_code=403, detail="Forbidden origin")
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
