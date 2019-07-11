import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils
from models.C3D import C3D
from models.R2Plus1D import R2Plus1D
from models.STTS import STTS


def center_crop(image):
    height_index = math.floor((image.shape[0] - crop_size) / 2)
    width_index = math.floor((image.shape[1] - crop_size) / 2)
    image = image[height_index:height_index + crop_size, width_index:width_index + crop_size, :]
    return np.array(image).astype(np.uint8)


def crop_frames(buffer):
    if buffer.shape[0] > clip_len:
        frame_groups = []
        # select the middle and center frames
        for frames in np.array_split(buffer, clip_len, axis=0):
            frame_groups.append(frames[math.ceil(frames.shape[0] / 2 - 1), :, :, :])
        buffer = np.stack(frame_groups, axis=0)

    # padding repeated frames to make sure the shape as same
    if buffer.shape[0] < clip_len:
        repeated = clip_len // buffer.shape[0] - 1
        remainder = clip_len % buffer.shape[0]
        buffered, reverse = buffer, True
        if repeated > 0:
            padded = []
            for i in range(repeated):
                if reverse:
                    pad = buffer[::-1, :, :, :]
                    reverse = False
                else:
                    pad = buffer
                    reverse = True
                padded.append(pad)
            padded = np.concatenate(padded, axis=0)
            buffer = np.concatenate((buffer, padded), axis=0)
        if reverse:
            pad = buffered[::-1, :, :, :][:remainder, :, :, :]
        else:
            pad = buffered[:remainder, :, :, :]
        buffer = np.concatenate((buffer, pad), axis=0)
    return buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Activity Recognition')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51'], help='dataset type')
    parser.add_argument('--model_type', default='stts-a', type=str, choices=['stts-a', 'stts', 'r2plus1d', 'c3d'],
                        help='model type')
    parser.add_argument('--video_name', type=str, help='test video name')
    parser.add_argument('--model_name', default='ucf101_stts-a.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE, MODEL_TYPE, VIDEO_NAME, MODEL_NAME = opt.data_type, opt.model_type, opt.video_name, opt.model_name

    clip_len, resize_height, crop_size, = utils.CLIP_LEN, utils.RESIZE_HEIGHT, utils.CROP_SIZE
    class_names = utils.get_labels(DATA_TYPE)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if '{}_{}.pth'.format(DATA_TYPE, MODEL_TYPE) != MODEL_NAME:
        raise NotImplementedError('the model name must be the same model type and same data type')

    if MODEL_TYPE == 'stts-a' or MODEL_TYPE == 'stts':
        model = STTS(len(class_names), (2, 2, 2, 2), MODEL_TYPE)
    elif MODEL_TYPE == 'r2plus1d':
        model = R2Plus1D(len(class_names), (2, 2, 2, 2))
    else:
        model = C3D(len(class_names))

    checkpoint = torch.load('epochs/{}'.format(MODEL_NAME), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE).eval()

    # read video
    cap, retaining, clips = cv2.VideoCapture(VIDEO_NAME), True, []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        resize_width = math.floor(frame.shape[1] / frame.shape[0] * resize_height)
        # make sure it can be cropped correctly
        if resize_width < crop_size:
            resize_width = resize_height
            resize_height = math.floor(frame.shape[0] / frame.shape[1] * resize_width)
        tmp_ = center_crop(cv2.resize(frame, (resize_width, resize_height)))
        tmp = tmp_.astype(np.float32) / 255.0
        clips.append(tmp)
    cap.release()

    clips = np.stack(clips, axis=0)
    inputs = crop_frames(clips)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
    inputs = torch.from_numpy(inputs).to(DEVICE)
    with torch.no_grad():
        outputs = model.forward(inputs)

    prob = F.softmax(outputs, dim=-1)
    label = torch.argmax(prob, dim=-1).detach().cpu().numpy()[0]
    print('{} is {}({.4f})'.format(VIDEO_NAME, class_names[label].split(' ')[-1].strip(), prob[0][label].item()))
