# Fast-SCNN
A PyTorch implementation of Fast-SCNN based on the paper [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502).

![Network Architecture image from the paper](structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- opencv
```
pip install opencv-python
```
- cityscapesScripts
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

## Expected dataset structure for Cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
  leftImg8bit/
    train/
    val/
```
run [createTrainIdLabelImgs.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to creat `labelTrainIds.png`.

## Usage
```
python train.py --crop_h 512 --crop_w 1024
optional arguments:
--data_path                   Data path for cityscapes dataset [default value is '/home/data/cityscapes']
--crop_h                      Crop height for training images [default value is 1024]
--crop_w                      Crop width for training images [default value is 2048]
--batch_size                  Number of data for each batch to train [default value is 12]
--epochs                      Number of sweeps over the dataset to train [default value is 1000]
```

## Results
There are some difference between this implementation and official implementation:
1. No `L2 regularization` used;
2. No `Dropout` used in the last layer;
3. The scales of `Multi-Scale Training` are `(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)`;
4. No `color channels noise and brightness` used;
5. No `auxiliary losses` at the end of `learning to downsample` and the `global feature extraction modules` used;

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Backbone</th>
		<th>feature dim</th>
		<th>batch size</th>
		<th>epoch num</th>
		<th>temperature</th>
		<th>momentum</th>
		<th>k</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r18 -->
		<tr>
			<td align="center">ResNet18</td>
			<td align="center">128</td>
			<td align="center">128</td>
			<td align="center">200</td>
			<td align="center">0.1</td>
			<td align="center">0.5</td>
			<td align="center">200</td>
			<td align="center">80.64</td>
			<td align="center">98.56</td>
			<td align="center"><a href="https://pan.baidu.com/s/1akdeCaWiKQ03MeTD_MeapA">model</a>&nbsp;|&nbsp;v7qm</td>
		</tr>
	</tbody>
</table>