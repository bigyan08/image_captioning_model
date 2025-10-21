import src.config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms
from src.utils import get_loader,load_checkpoint,save_checkpoint
from src.model import CNNtoRNN
import os

transform = transforms.Compose(
    [
        transforms.Resize(356),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #from pytorch docs(resnet)
    ]
    )

train_loader ,val_loader, dataset = get_loader(
    root_folder=cfg.Image_DIR,
    annotation_file=cfg.Annotation_File,
    transform = transform,
    num_workers=2
    )


def train():
    torch.backends.cudnn.benchmark = True 
    writer = SummaryWriter("../notebook/runs/flickr") 
    step = 0 
    model = CNNtoRNN(cfg.EMBED_SIZE,cfg.HIDDEN_SIZE,cfg.VOCAB_SIZE,cfg.NUM_LAYERS).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg.LEARNING_RATE) 
    if cfg.LOAD_MODEL:
        checkpoint_path="../notebook/checkpoints/my_checkpoint.pth.tar"
        if os.path.exists(checkpoint_path):
            try:
                step = load_checkpoint(torch.load(checkpoint_path),model,optimizer)
                print("Checkpoint Loaded.")
            except Exception as e:
                print("Error loading checkpoint:{e}")
                print("Training from scratch...")
                step=0

    model.train()
    for epoch in range(cfg.NUM_EPOCHS):
        total_train_loss = 0
        loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}]",
            unit="batch",
            leave=True
        )
        for batch_idx, (imgs, captions) in loop:
            imgs, captions = imgs.to(cfg.DEVICE), captions.to(cfg.DEVICE)
            captions_in = captions[:-1, :]      # remove last token
            captions_target = captions[1:, :]   # remove first token
        
            outputs = model(imgs, captions_in)
            outputs = outputs[49:]  # remove 49 image-region outputs
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(batch_loss=loss.item(),
            avg_loss=total_train_loss/(batch_idx+1))
    
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
        if cfg.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
                        }
            save_checkpoint(checkpoint)
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}")
    torch.save(model.state_dict(),"final_model.pth")
    print("Model saved successfully!")

if __name__=='__main__':
    train()