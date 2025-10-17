from torchvision.transforms import transforms
import torch
from PIL import Image
import os 
from src.utils import get_loader
from src.model import CNNtoRNN
import src.config as cfg

def inference():
    inference_transform = transforms.Compose(
        [
            transforms.Resize(356),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ]
    )
    
    _,_,dataset = get_loader(
        root_folder=cfg.Image_DIR,
        annotation_file=cfg.Annotation_File,
        transform = inference_transform,
        batch_size=1,
        num_workers=1,
        split_ratio=cfg.SPLIT_RATIO,
        )
    
    model = CNNtoRNN(cfg.EMBED_SIZE,cfg.HIDDEN_SIZE,len(dataset.vocab),cfg.NUM_LAYERS).to(cfg.DEVICE)
    model.load_state_dict(torch.load("./notebook/model/final_model.pth",map_location=cfg.DEVICE))
    model.eval()

    image_path = input("enter image path: ").strip()
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = inference_transform(image).unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            caption_tokens = model.caption_image(image_tensor,dataset.vocab)
        
        caption = " ".join(caption_tokens)
        print(f"Caption:{caption}")
    else:
        print(f'Error loading image')


if __name__=='__main__':
    inference() 