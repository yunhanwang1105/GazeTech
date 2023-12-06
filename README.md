
<div align="center">

<h2>Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation</h2>

<div>
    <a href='https://yunhanwang1105.github.io/' target='_blank'>Yunhan Wang</a><sup>1</sup>&emsp;
    <a href='https://x-shi.github.io/' target='_blank'>Xiangwei Shi</a><sup>1</sup>&emsp;
    <a href='https://research.nvidia.com/person/shalini-de-mello' target='_blank'>Shalini De  Mello</a><sup>2</sup>&emsp;
    <a href='https://hyungjinchang.wordpress.com/' target='_blank'>Hyung Jin Chang</a><sup>3</sup>&emsp;
    <a href='https://www.ccmitss.com/zhang' target='_blank'>Xucong Zhang</a><sup>1</sup>&emsp;
</div>

<div>
    <sup>1</sup>Computer Vision Lab, Delft University of Technology&emsp;
    <sup>2</sup>NVIDIA&emsp; <br>
    <sup>3</sup>School of Computer Science, University of Birmingham
</div>

</div>

<br/>

## Description
This repository contains frameworks for pre-processing, training, and evaluating full face or multi-region (face, left and right eyes) gaze estimation models on the following three datasets: ETH-XGaze, MPIIFaceGaze, and Gaze360. These frameworks allow for flexible load single-face input or multi-region input and serve to reproduce our results and benchmark new models.

The readme files in the submodules of different datasets here contain step-by-step tutorials to help you set up, train and evaluate our existing and your new models. 

## Links
- [Paper arXiv page](https://arxiv.org/abs/2308.09593)
- To prepare the normalized data for ETH-XGaze, MPIIFaceGaze and Gaze360 datasets, please refer to our [data normalization repository](https://github.com/X-Shi/Data-Normalization-Gaze-Estimation).

## Pre-trained models
The pre-trained models on different datasets can be found in the sub-folders (ETHX-Gaze, MPIIFaceGaze, Gaze360). 

## Citation
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
