import pathlib
from typing import Callable, Tuple

import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable, load_model: str):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.load_model = load_model

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
            if self.load_model == 'load_multi_region':
                left_eye = f.get(f'{self.person_id_str}/left/{index:04}')[()]
                right_eye = f.get(f'{self.person_id_str}/right/{index:04}')[()]

        image = self.transform(image)
        gaze = torch.from_numpy(gaze)
        if self.load_model == 'load_single_face':
            images = {"face": image}
        if self.load_model == 'load_multi_region':
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)
            images = {"face": image, "left_eye": left_eye, "right_eye": right_eye}
        return images, gaze#, left_eye, right_eye

    def __len__(self) -> int:
        return 3000
