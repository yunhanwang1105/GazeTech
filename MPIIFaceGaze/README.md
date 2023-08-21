## Codes for Gaze Estimation on MPIIFaceGaze dataset

This folder contains the codes for gaze estimation on [MPIIFaceGaze]([http://gaze360.csail.mit.edu/](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)) dataset. This framework serves to create baselines. It allows loading single face image or multi-region (face and eye) images for different models flexibly.

### Requirements

The models in the technical report are trained with Python 3.8.8, PyTorch 1.12.0, and CUDA 11.3.

To install the required packages, run:\
`pip install -r requirements.txt`

### Dataset

Please refer to our [data normalization repository](https://github.com/X-Shi/Data-Normalization-Gaze-Estimation) for pre-processing the multi-region MPIIFaceGaze dataset. If you would like to run the codes on MPIIFaceGaze dataset, please download through the link above. We do not provide the access to the dataset.\
For the single face training, we use the provided normalized data from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation).

#### **Notice**
**The coordinate system of the gaze labels of the provided normalized data is different from the one from our data normalization repository. The coordinate system in our data normalization repository is aligned with the one in ETH-XGaze dataset.
The difference between two coordinate systems is that the X, Y, and Z directions are opposite from the corresponding ones.**

### Training and Evaluation

To run the codes for training, please run:\
`python train.py --config configs/mpiifacegaze/resnet_50_train_x.yaml`\
Please change the 'x' above. In the `.yaml` file, you may need to change some details, including the path to your dataset, the name of model (choose from 'face_res50', 'multi_region_res50', and 'multi_region_res50_share_eyenet'), in_stride, resolution and etc.

To run the codes for evaluation, please run:\
`python evaluate.py --config configs/mpiifacegaze/resnet_50_eval_x.yaml`\
Please change the 'x' above. You may change the path to your saved models.

### Trained models

If you want to evaluate our trained models, you can download them through [this link]().

### Citation

If you find this framework useful, please consider citing the paper of MPIIFaceGaze and our paper.
```
    @inproceedings{zhang2017s,
    title={It’s written all over your face: Full-face appearance-based gaze estimation},
    author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
    booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
    pages={2299--2308},
    year={2017},
    organization={IEEE}
    }

```
```
@misc{wang2023investigation,
      title={Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation}, 
      author={Yunhan Wang and Xiangwei Shi and Shalini De Mello and Hyung Jin Chang and Xucong Zhang},
      year={2023},
      eprint={2308.09593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
