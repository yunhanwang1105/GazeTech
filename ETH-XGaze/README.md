# ETH-XGaze-tech
This repository is an extended version of the [ETH-XGaze official implementation](https://github.com/xucong-zhang/ETH-XGaze) to train gaze estimation models on ETH-XGaze. This framework serves to create baselines and [benchmark new full face or multi-region gaze estimation models](#benchmark-your-own-model). It allows loading single face image or multi-region (face and eye) images for different models flexibly.

<br/>

# Set-up

## 1. Dataset
Please refer to our [data normalization repository](https://github.com/X-Shi/Data-Normalization-Gaze-Estimation) for downloading and pre-processing the multi-region ETH-XGaze dataset. After that, sepcify the path to the dataset at *--data_dir* in **config.py**. This directory should contain **/train** and **/test** where the h5 files are placed. 

## 2. Environment
This code base is tested on the Linux system. We recommond the paradigm of creating a Conda environment first:
`conda create --name gaze_env` and then install the dependencies in the Conda environment: `pip install -r requirements.txt`. You could also install the dependencies directly via the previous pip command. 

<br/>

# Train

## 1. Weights and Biases
We recommend the use of [Weights and Biases](https://wandb.ai/) (wandb) to view the learning curves and more. To use it, please first login and remove `mode="disabled"` at L22 in **main.py**. You can then access the learning curves and other training results on the web while you model is being trained.

## 2. Start training example
`python main.py --model_name multi_region_res50 --epochs 30`

You can find more hyperparameters and their descriptions in **config.py**. 

<br/>

# Evaluation

## 1. Predict
To evaluate a model's performance, first specify the model path at *--pre_trained_model_path*, please make sure the other training and model hyperparameters are consistent with the hyperparameters your model was trained with. Then run `python main.py --is_train False`. 

## 2. Obtain gaze error
You are supposed to see a txt file called **within_eva_results.txt** after finish testing. Put this file in a folder and zip it, the upload it to the [Codalab page](https://codalab.lisn.upsaclay.fr/competitions/7423) for testing results on ETH-XGaze. 

<br/>

# Benchmark your own model
First give your model a name, then sepcify whether your model needs a single face image and multi-region images in *get_load_mode* in **main.py**. Initialize your model at L60-77 in **trainer.py**. The forward pass is performed from L160 and L230 in **trainer.py**. 

<br/>

# Pre-trained models
Please find the pre-trained models at this [link](https://drive.google.com/drive/folders/15XvsRPorAYqyMyBxVmxeog9bZB64OKhy?usp=sharing).

<br/>

# Miscellaneous

If you would like to draw a gaze arrow, please refer to the demo of the [ETH-XGaze official implementation](https://github.com/xucong-zhang/ETH-XGaze). 

<br/>

# Citation
If you find this framework useful, please consider citing our paper and ETH-XGaze. 

    @misc{wang2023investigation,
          title={Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation}, 
          author={Yunhan Wang and Xiangwei Shi and Shalini De Mello and Hyung Jin Chang and Xucong Zhang},
          year={2023},
          eprint={2308.09593},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
<br/>

    @inproceedings{Zhang2020ETHXGaze,
      author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and Derek Bradley and Siyu Tang and Otmar Hilliges},
      title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation},
      year      = {2020},
      booktitle = {European Conference on Computer Vision (ECCV)}
    }

<br/>


