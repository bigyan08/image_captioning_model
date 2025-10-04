import spacy
import os
import torch
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset,random_split


spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod 
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self,sentence_list):
        '''
        sentence_list: input to the function of list of sentences

        Flow: Loop over sentencelist -> for each word in sentence update its freq
              -> if freq reaches threshold then add it to the vocabulary, else not(<UNK>).

        Note: The counter starts from 4, because 0-3 are reserved.
        '''
        frequencies = {}
        idx = 4 # 0-3 are reserved for pad,sos,bos,unk
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word,0) + 1 # new word gets 1, if word already exists then updates the counter
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        '''
        Takes in text and returns the id of the tokenized text if in stoi else <UNK>
        '''
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # get image and caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        '''
        Takes in particular index, maps to the index-th row of dataframe with 
        image,caption pair. Then wraps the numericalized caption with SOS and EOS.
        Returns the tensor of the final list of ids.
        '''
        #pull a row(image,caption pair) from dataframe
        caption = self.captions[index]
        img_id = self.imgs[index]
        # load the image with PIL
        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # wrapping up the numericalized token with SOS and EOS at first and last respectively.
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    

class MyCollate:
    '''
    This class aims to match the sequence of every text sequence using <PAD> tokens.
    For Images, This class introduces a Batch dimension making it [B,C,H,W] from [C,H,W]
    '''
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx

    def __call__(self,batch):
        '''
        Input Example: batch -> [(img1_tensor,[1,4,8,4]),.... ], The image tensor size remains same, but 
        text tensor sequence length may differ so to fix we add pad tokens.

        For images: Take each 1st value of each item from batch and unsqueeze at dimension 0 to add batch dim in all the image tensor
        then concatenate along the 0th(batch) dimension.

        For captions (as targets): Take the 2nd value of each item from batch and then pad it with pad value
        '''
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first = False,padding_value=self.pad_idx)

        return imgs,targets



def get_loader( root_folder, annotation_file, transform, batch_size, num_workers, split_ratio, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder,annotation_file,transform=transform)
    #train val split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        drop_last = False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        drop_last = False
    )
    return train_loader,val_loader,dataset



def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print("Saving Checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step