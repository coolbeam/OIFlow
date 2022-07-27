# [TIP2021] OIFlow: Occlusion-Inpainting Optical Flow Estimation by Unsupervised Learning

<h3 align="center"> Shuaicheng Liu$^1$, Kunming Luo$^2$, Nianjin Ye$^2$, Chuan Wang$^2$, Jue Wang$^2$, Bing Zeng$^1$</h3>
<h3 align="center"> 1. University of Electronic Science and Technology of China</h3>
<h3 align="center"> 2. Megvii Research </h3>

This is the official implementation of paper [***OIFlow: Occlusion-Inpainting Optical Flow Estimation by Unsupervised Learning***](), IEEE Transactions on Image Processing, 2021



    @article{liu2021oiflow,
      title={OIFlow: occlusion-inpainting optical flow estimation by unsupervised learning},
      author={Liu, Shuaicheng and Luo, Kunming and Ye, Nianjin and Wang, Chuan and Wang, Jue and Zeng, Bing},
      journal={IEEE Transactions on Image Processing},
      volume={30},
      pages={6420--6433},
      year={2021},
      publisher={IEEE}
    }

## Introduction
![inpainting_visualization](./images/gif_v2_1.gif)
Occlusion is an inevitable and critical problem in unsupervised optical flow learning. 
Existing methods either treat occlusions equally as non-occluded regions or simply remove them to avoid incorrectness. However, the occlusion regions can provide 
effective information for optical flow learning. In this paper, we present OIFlow, an occlusion-inpainting framework to make full use of occlusion regions. 
Specifically, a new appearance-flow network is proposed to inpaint occluded flows based on the image content. 
Moreover, a boundary dilated warp is proposed to deal with occlusions caused by displacement beyond the image border. 
We conduct experiments on multiple leading flow benchmark datasets such as Flying Chairs, KITTI and MPISintel, which demonstrate that the performance is significantly improved by our proposed occlusion handling framework.

## Acknowledgement
Part of our codes are adapted from [IRR-PWC](https://github.com/visinf/irr) and [UnFlow](https://github.com/simonmeister/UnFlow), we thank the authors for their contributions.
