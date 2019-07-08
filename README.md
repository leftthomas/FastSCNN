# STTS
A PyTorch implementation of Spatio-Temporal and Temporal-Spatio Convolutional Network based on the paper 
[Spatio-Temporal and Temporal-Spatio Convolutional Network for Activity Recognition]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```
- rarfile
```
pip install rarfile
```
- rar
```
sudo apt install rar
```
- unrar
```
sudo apt install unrar
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```

## Datasets
The datasets are coming from [UCF101](http://crcv.ucf.edu/data/UCF101.php) and 
[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
Download `UCF101` and `HMDB51` datasets with `train/val/test` split files into `data` directory.
We use the `split1` to split files. Run `misc.py` to preprocess these datasets.

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 20
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51'])
--gpu_ids                     selected gpu [default value is '0,1']
--model_type                  model type [default value is 'stts-a'](choices=['stts-a', 'stts', 'r2plus1d', 'c3d'])
--batch_size                  training batch size [default value is 16]
--num_epochs                  training epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

### Inference Video
```
python inference.py --video_name data/ucf101/ApplyLipstick/v_ApplyLipstick_g04_c02.avi
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51', 'kinetics600'])
--model_type                  model type [default value is 'stts-a'](choices=['stts-a', 'stts', 'r2plus1d', 'c3d'])
--video_name                  test video name
--model_name                  model epoch name [default value is 'ucf101_st-ts-a.pth']
```
The inferences will show in a pop up window.

## Results
The train/val/test loss, accuracy and confusion matrix are showed on visdom. 
### UCF101
![result](results/ucf101_c3d.png)
![result](results/ucf101_r2plus1d.png)
### HMDB51
![result](results/hmdb51_c3d.png)
![result](results/hmdb51_r2plus1d.png)

