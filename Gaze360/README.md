## Codes for Gaze Estimation on Gaze360 dataset

This folder contains the codes for gaze estimation on [Gaze360](http://gaze360.csail.mit.edu/) dataset. This framework serves to create baselines. It allows loading single face image or multi-region (face and eye) images for different models flexibly.

### Requirements

The models in the technical report are trained with Python 3.8.8, PyTorch 1.12.0, and CUDA 11.3.

To install the required packages, run:\
`pip install -r requirements.txt`

### Dataset

Please refer to our [data normalization repository](https://github.com/X-Shi/Data-Normalization-Gaze-Estimation) for pre-processing the multi-region Gaze360 dataset. If you would like to run the codes on Gaze360 dataset, please download through the link above. We do not provide the access to the dataset.

### Training and Evaluation

We use [Weights and Biases](https://wandb.ai/) (wandb) to view the learning curves and validataion performance. The final models are selected based on the performance on the validation subset.

To run the codes for training, please run:\
`python main.py --model_name=face_res50 --in_stride=2 --batch_size=100`

To run the codes for evaluation, please run:\
`python main.py --is_train=False --pre_trained_model_path='your_local_path/ckpt/epoch_xx_0.0001_ckpt.pth.tar' --model_name='face_res50' --in_stride=2 --batch_size=32`

You can find more hyperparameters and their descriptions in **config.py** file.

### Trained models

If you want to evaluate our trained models, you can download them through [this link](https://drive.google.com/drive/folders/18VI3Uh5h_4BDO8vB_YWeawyLDjHwBzvG?usp=sharing).

### Citation

If you find this framework useful, please consider citing the paper of Gaze360 and our paper.
```
@inproceedings{gaze360_2019,
    author = {Petr Kellnhofer and Adria Recasens and Simon Stent and Wojciech Matusik and Antonio Torralba},
    title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

TODO, OUR CITATION
