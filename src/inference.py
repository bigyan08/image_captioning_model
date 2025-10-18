from torchvision.transforms import transforms
import torch
from PIL import Image
import os 
from src.utils import get_loader
from src.model import CNNtoRNN
import src.config as cfg
import pickle
from src.utils import Vocabulary

def inference():
    # need to load the vocab file which was saved during training
    vocab_path = './notebook/models/flickr30k_vocab.pkl'
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f" Vocabulary loaded successfully with {len(vocab)} words")
    except FileNotFoundError:
        print(f" Vocabulary file not found: {vocab_path}")
        return
    except Exception as e:
        print(f" Error loading vocabulary: {e}")
        return

    inference_transform = transforms.Compose(
        [
            transforms.Resize(356),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ]
    )
        
    model = CNNtoRNN(cfg.EMBED_SIZE,cfg.HIDDEN_SIZE,len(vocab),cfg.NUM_LAYERS).to(cfg.DEVICE)
    model.load_state_dict(torch.load("./notebook/models/final_model_30k.pth",map_location=cfg.DEVICE))
    model.eval()

    image_path = input("enter image path: ").strip()
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = inference_transform(image).unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            caption_tokens = model.caption_image(image_tensor,vocab)
        
        caption = " ".join(caption_tokens)
        print(f"Caption:{caption}")
    else:
        print(f'Error loading image')


if __name__=='__main__':
    inference() 