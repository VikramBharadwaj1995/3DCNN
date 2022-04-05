from PIL import Image
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from skimage.color import rgb2gray
from basic_model import Net
import sys, os
import numpy as np

if __name__ == '__main__':
    # Read the two input parameters, which is the model checkpoint and grayscale image
    model_checkpoint, image = sys.argv[1], sys.argv[2]
    # Load model from basic_model.py by calling the class constructor
    model = Net()
    # If GPU available, set current execution to the CUDA instance, else, use CPU 
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    
    # Open the grayscale image
    img = Image.open(image)

    # Reshape it to the same size as the one that is accepted by the network
    image_l = img.resize((256, 256))
    image_l = rgb2gray(image_l)
    # Use the transform to convert to a float tensor 
    image_l = transforms.ToTensor()(image_l).float()
    
    # Evaluate the image from the model
    model.eval()
    
    with torch.no_grad():
        preds = model(image_l.unsqueeze(0).to(device))

    print(torch.exp(preds[0].cpu()))