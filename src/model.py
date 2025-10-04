import torch
import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self,embed_size,train_CNN=False):
        super(EncoderCNN,self).__init__()
        self.train_CNN = train_CNN
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,embed_size) #removing the classifier layer, then preserving the features in embed_size size.
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        for name,param in self.resnet.named_parameters():
            param.requires_grad = (train_CNN or 'fc' in name)

    def forward(self,images):
        features = self.resnet(images)
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self,features,captions):
        '''
        Adds a sequence dimension to the features (dim0)[1,batch_size,embed_size] and then concatenates along dim0.
        First timestep gives image feature, Next timestep gives caption embeddings.
        Final shape becomes:[1+seq_len,batch_size,embed_size]
        '''
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)
        hiddens,_ = self.lstm(embeddings)
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
        
    def caption_image(self,image,vocabulary,max_len=50):
        '''
        During inference or evaluation we wont have target captions(which is going to be predicted,duh!).
        image: inference image
        vocabulary: whole mapping dictionary
        max_len: max length for captions to be predicted.
        '''
        caption_result = []
        with torch.no_grad():
            x = self.CNN(image).unsqueeze(0) # add batch dimension at dim0
            states = None
            for _ in range(max_len):
                hiddens,states = self.RNN.lstm(x,states)
                output = self.RNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                caption_result.append(predicted.item())
                x = self.RNN.embed(predicted).unsqueeze(0)
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in caption_result]
                