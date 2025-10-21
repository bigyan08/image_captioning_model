import torch

#dataset
Image_DIR='./data/images' #'path/to/image/'
Annotation_File='./data/captions.txt' #'path/to/annotationfile' #in csv


#model
VOCAB_SIZE = None   #set after loading dataset
LOAD_MODEL = False
SAVE_MODEL = True
TRAIN_CNN = False #set this true to train all the params of the ResNet model

EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS =10
BATCH_SIZE=16
NUM_WORKERS=8
SPLIT_RATIO=0.8 #traindata percentage

FREQUENCY_THRESHOLD=5 #min no. of times a word appears to be counted as non-rare. if less than this then <UNK>.

#device setup
# DEVICE='cuda' if torch.cuda.is_available else 'cpu'
DEVICE = 'cpu'