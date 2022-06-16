# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from .base import BaseFewShotDataset


@DATASETS.register_module()
class TieredImageNetDataset(BaseFewShotDataset):
    """TieredImageNet dataset for few shot classification.

    Args:
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
    """

    resource = 'https://github.com/renmengye/few-shot-ssl-public'
    ALL_CLASSES = []
    TRAIN_CLASSES = []
    VAL_CLASSES = []
    TEST_CLASSES = []

    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 *args,
                 **kwargs):
        #### init CLASSES
        dataset_file = osp.join(kwargs['data_prefix'], 'data_config.py')
        dataset_config = mmcv.Config.fromfile(dataset_file)
        self.ALL_CLASSES = dataset_config.ALL_CLASSES
        self.TRAIN_CLASSES = dataset_config.TRAIN_CLASSES
        self.VAL_CLASSES = dataset_config.VAL_CLASSES
        self.TEST_CLASSES = dataset_config.TEST_CLASSES
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        self.GENERAL_CLASSES = self.get_general_classes()
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += [i[0] for i in self.TRAIN_CLASSES]
                elif subset_ == 'val':
                    class_names += [i[0] for i in self.VAL_CLASSES]
                elif subset_ == 'test':
                    class_names += [i[0] for i in self.TEST_CLASSES]
                else:
                    raise ValueError(f'invalid subset {subset_} only '
                                     f'support train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def get_general_classes(self) -> List[str]:
        """Get general classes of each classes."""
        general_classes = []
        for subset_ in self.subset:
            if subset_ == 'train':
                general_classes += [i[1] for i in self.TRAIN_CLASSES]
            elif subset_ == 'val':
                general_classes += [i[1] for i in self.VAL_CLASSES]
            elif subset_ == 'test':
                general_classes += [i[1] for i in self.TEST_CLASSES]
            else:
                raise ValueError(f'invalid subset {subset_} only '
                                 f'support train, val or test.')
        return general_classes

    def load_annotations(self) -> List[Dict]:
        """Load annotation according to the classes subset."""
        data_infos = []
        for subset_ in self.subset:
            labels_file = osp.join(self.data_prefix, f'{subset_}_labels.pkl')
            assert osp.exists(labels_file), \
                f'Please download ann_file through {self.resource}.'
            data_infos = []
            with open(labels_file, 'rb') as labels:
                labels = pickle.load(labels)
                label_specific = labels['label_specific']
                label_general = labels['label_general']
                class_specific = labels['label_specific_str']
                class_general = labels['label_general_str']
                unzip_file_path = osp.join(self.data_prefix, subset_)
                is_unzip_file = osp.exists(unzip_file_path)
                if not is_unzip_file:
                    msg = ('Please use the provided script '
                           'tools/classification/data/unzip_tiered_imagenet.py'
                           'to unzip pickle file. Otherwise the whole pickle '
                           'file may cost heavy memory usage when the model '
                           'is trained with distributed parallel.')
                    warnings.warn(msg)
                for i in range(len(label_specific)):
                    class_specific_name = class_specific[label_specific[i]]
                    class_general_name = class_general[label_general[i]]
                    gt_label = self.class_to_idx[class_specific_name]
                    assert class_general_name == self.GENERAL_CLASSES[gt_label]
                    filename = osp.join(subset_, f'{subset_}_image_{i}.byte')
                    info = {
                        'img_prefix': self.data_prefix,
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label': np.array(gt_label, dtype=np.int64),
                    }
                    # if the whole pickle file isn't unzipped,
                    # image bytes of will be put into data_info
                    if not is_unzip_file:
                        # info['img_bytes'] = img_bytes[i]
                        pass
                    data_infos.append(info)
        return data_infos
