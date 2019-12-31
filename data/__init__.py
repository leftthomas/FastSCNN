"""
This module provides data loaders and transformers for popular vision datasets.
"""
from data.cityscapes import CitySegmentation
from data.mscoco import COCOSegmentation
from data.pascal_aug import VOCAugSegmentation
from data.pascal_voc import VOCSegmentation

datasets = {
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
