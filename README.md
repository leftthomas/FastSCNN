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
python train.py --epochs 50 --feature_dim 256
optional arguments:
--feature_dim                 Feature dim for each image [default value is 128]
--m                           Negative sample number [default value is 4096]
--temperature                 Temperature used in softmax [default value is 0.1]
--momentum                    Momentum used for the update of memory bank [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 128]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
```

## Results

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