import pathlib
from typing import List, Union

import torch
import yacs.config
from torch.utils.data import Dataset

from ..transforms import create_transform
from ..types import GazeEstimationMethod
from .mpiifacegaze import OnePersonDataset


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:
    
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]

    transform = create_transform(config)

    if config.model.name == 'face_res50':
        load_mode = 'load_single_face'
    elif config.model.name == 'multi_region_res50':
        load_mode = 'load_multi_region'
    elif config.model.name == 'multi_region_res50_share_eyenet':
        load_mode = 'load_multi_region'
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")



    if is_train:
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform, load_mode)
                for person_id in person_ids
            ])
            assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform, load_mode)
                for person_id in person_ids if person_id != test_person_id
            ])
            assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_person_id = person_ids[config.test.test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform, load_mode)
        assert len(test_dataset) == 3000
        return test_dataset
