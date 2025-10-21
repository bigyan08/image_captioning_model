import torch
import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # keep conv features, drop avgpool + fc
        self.train_CNN = train_CNN
        self.resnet = nn.Sequential(*modules)

        self.conv2embed = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        for name, param in self.resnet.named_parameters():
            param.requires_grad = train_CNN

    def forward(self, images):
        # Extract conv feature map: [B, 2048, 7, 7]
        if not self.train_CNN:
            with torch.no_grad():
                features = self.resnet(images)
        else:
            features = self.resnet(images)
        # Project to embed size and flatten spatially
        features = self.conv2embed(features)        # [B, embed_size, 7, 7]
        features = features.flatten(2)              # [B, embed_size, 49]
        features = features.permute(0, 2, 1)        # [B, 49, embed_size]
        features = self.dropout(self.relu(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features, captions):
        # features: [B, 49, embed_size]
        # captions: [seq_len, B]
        features = features.permute(1, 0, 2)  # [49, B, embed_size]
        embeddings = self.dropout(self.embed(captions))  # [seq_len, B, embed_size]

        lstm_input = torch.cat((features, embeddings), dim=0)  # [49+seq_len, B, embed_size]
        hiddens, _ = self.lstm(lstm_input)
        outputs = self.linear(hiddens)
        return outputs
    

                
class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.CNN = EncoderCNN(embed_size)
        self.RNN = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features = self.CNN(images)
        outputs = self.RNN(features,captions)
        return outputs
        
    def caption_image(self, image, vocabulary, max_len=50):
        '''
        During inference or evaluation we wont have target captions(which is going to be predicted,duh!).
        image: inference image
        vocabulary: whole mapping dictionary
        max_len: max length for captions to be predicted.
        '''
        caption_result = []
        with torch.no_grad():
            # Get CNN features
            features = self.CNN(image)  # [B, 49, embed_size]
            
            # Start with <SOS> token
            caption_so_far = [vocabulary.stoi['<SOS>']]
            
            for _ in range(max_len):
                # Convert current caption to tensor
                captions_tensor = torch.tensor([caption_so_far]).to(image.device)  # [1, len(caption_so_far)]
                captions_tensor = captions_tensor.permute(1, 0)  # [len(caption_so_far), 1] - match training format
                
                # Use the model's forward pass - same as training
                outputs = self.RNN(features, captions_tensor)  # [49+len(caption_so_far), 1, vocab_size]
                
                # Skip the first 49 outputs (image features) - same as training
                text_outputs = outputs[49:, :, :]  # [len(caption_so_far), 1, vocab_size]
                
                # Get prediction for the last text token
                predicted = text_outputs[-1, 0, :].argmax(0)  # Get last prediction
                predicted_idx = predicted.item()
                
                caption_result.append(predicted_idx)
                caption_so_far.append(predicted_idx)
                
                if vocabulary.itos[predicted_idx] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in caption_result]