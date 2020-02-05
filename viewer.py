import argparse
import os

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from model import FastSCNN
from utils import transform, palette

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
    parser.add_argument('--model_weight', type=str, default='1024_2048_model.pth', help='pretrained model weight')
    parser.add_argument('--input_pic', type=str,
                        default='/home/data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                        help='path to the input picture')
    # args parse
    args = parser.parse_args()
    model_weight, input_pic = args.model_weight, args.input_pic

    image = Image.open(input_pic).convert('RGB')
    image = transform(image).unsqueeze(dim=0).cuda()

    # model load
    model = FastSCNN(in_channels=3, num_classes=19)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))
    model = model.cuda()
    model.eval()

    # predict
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        mask = ToPILImage()(pred.byte().cpu())
        mask.putpalette(palette)
        mask.save(os.path.split(input_pic)[-1].replace('leftImg8bit', 'pred'))
