
# Image Captioning Model

Generate captions based on the image provided.

## Overview

This project combines Computer Vision and Natural Language Processing (NLP) to generate natural language captions for images.
The model extracts visual features using a CNN as encoder and generates textual descriptions using an RNN(LSTM) as decoder.

- Used RESNET50 default weights to capture the essential image features.
- Those features passed onto the LSTM for text generation. 
- Used Flickr30k dataset with 30k images(each with 5 different captions for variety.)
- Trained on Kaggle's T4 gpu, for 15 epochs(~3 hours).

## Installation

### Option 1: Local Installation
Clone the repository:

```bash
git clone https://github.com/bigyan08/image_captioning_model
cd image_captioning_model
```

Install dependencies:

```bash
pip install -r requirements.txt 
```

### Option 2: Docker Installation (Recommended)

Clone the repository:

```bash
git clone https://github.com/bigyan08/image_captioning_model
cd image_captioning_model
```
Build the Docker image:

```bash
docker build -t image-captioning .
```

## Usage

### Local Usage

```bash
uvicorn api.app:app --host <host address> --port <port address>
```
Try: 0.0.0.0 for host address, 8000 for port address.
Then follow the local host link provided.

### Docker Usage
```bash
docker run -p 8000:8000 image-captioning
```

## Project Structure

- `src/` - Model training and inference code
- `api/` - Wrapped the model using FastApi.
- `data/` - some test images
- `README.md` - Project documentation
- `notebook/` - Jupyter notebooks for experimentation 
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration file

## Deployment

- Acess the website here: [https://image-captioning.azurewebsites.net/](https://image-captioning.azurewebsites.net/)
- Deployed on cloud (Azure container registry.)
- Trained model weights uploaded on Azure blob storage.
- Deployed as container and then accessed through web services.


## Note
- I have not provided my trained weights, feel free to train your own model with the full architecture and training codes provided.
- Ensure you have a compatible GPU for optimal performance and enable it in `config.py`. In my case, I used cpu as I don't have a GPU. The requirements file also explicitly downloads the 'cpu' only version so not I recommend to install full pytorch version with cuda to train.
- Train your own model and give a name, it will be saved in models directory.
- Vibecoded the frontend with just html,css and js using Claude sonnet.(One shot ui done.)

## Known Issues and Future Fixes
- Model is trained for just 15 epochs for now, I have saved checkpoints and I will keep on updating it along the way.
- The input given to the model should be more of a natural images with humans and all, it gives random captions to the images that are off-context.
- 30K images is still not enough for state of the art model, will be training on MS_COCO and other datsets in future for more resources.
- Optimization of the core architecture is due, I reviewed it throughly myself and also with claude sonnet, some fixes can be made to ensure faster inference. Will be updating in future.
- Feel free to contact me or raise an issue if any problem found.


## Contributing

Contributions are welcome! Please open issues or submit pull requests.


