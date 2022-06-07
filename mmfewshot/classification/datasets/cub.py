# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from mmfewshot.utils import local_numpy_seed
from .base import BaseFewShotDataset

@DATASETS.register_module()
class CUBDataset(BaseFewShotDataset):
    """CUB dataset for few shot classification.

    Args:
        classes_id_seed (int | None): A random seed to shuffle order
            of classes. If seed is None, the classes will be arranged in
            alphabetical order. Default: None.
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
    """

    resource = 'http://www.vision.caltech.edu/visipedia/CUB-200-2011.html'
    ALL_CLASSES = []
    TRAIN_CLASSES = []
    TEST_CLASSES = []
    VAL_CLASSES = []

    def __init__(self,
                 classes_id_seed: int = None,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 *args,
                 **kwargs) -> None:
        dataset_file = osp.join(kwargs['data_prefix'], 'data_config.py')
        dataset_config = mmcv.Config.fromfile(dataset_file)
        self.ALL_CLASSES = dataset_config.ALL_CLASSES
        
        self.classes_id_seed = classes_id_seed
        self.num_all_classes = len(self.ALL_CLASSES)

        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
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
            classes_ids = list(range(self.num_all_classes))
            if self.classes_id_seed is not None:
                with local_numpy_seed(self.classes_id_seed):
                    np.random.shuffle(classes_ids)
            # 100 train classes / 50 val classes / 50 test classes.
            # we follow the class splits used in Baseline++.
            # More details please refer to,
            # https://github.com/wyharveychen/CloserLookFewShot/blob/master/filelists/CUB/write_CUB_filelist.py
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if i % 2 == 0
                    ]
                elif subset_ == 'val':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if i % 4 == 1
                    ]
                elif subset_ == 'test':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if i % 4 == 3
                    ]
                else:
                    raise ValueError(f'invalid subset {subset_} only support '
                                     f'train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self) -> List[Dict]:
        """Load annotation according to the classes subset."""
        image_root_path = osp.join(self.data_prefix, 'images')

        assert osp.exists(image_root_path), \
            f'Please download dataset through {self.resource}.'

        image_dirs = [
            image_dir for image_dir in os.listdir(image_root_path)
            if osp.isdir(osp.join(image_root_path, image_dir))
        ]

        data_infos = []
        for i, image_dir in enumerate(image_dirs):
            if image_dir not in self.CLASSES:
                continue
            img_dir = osp.join(image_root_path, image_dir)

            for filename in os.listdir(img_dir):
                if osp.isfile(osp.join(img_dir, filename)):
                    gt_label = self.class_to_idx[image_dir]
                    info = {
                        'img_prefix': img_dir,
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label': np.array(gt_label, dtype=np.int64)
                    }
                    data_infos.append(info)
        return data_infos
