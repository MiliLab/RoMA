
<p align="center">

  <h2 align="center"><strong>RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing</strong></h2>

<div align="center">
<h5>
<em>Fengxiang Wang<sup>1</sup>, Hongzhen Wang<sup>2 †</sup>, Yulin Wang<sup>2</sup>, Di Wang<sup>3</sup>, Mingshuo Chen<sup>4</sup>, Haiyan Zhao<sup>2</sup>,<br/> Yangang Sun<sup>2</sup>, Shuo Wang<sup>2</sup>, Long Lan<sup>1</sup>, Wenjing Yang<sup>1 †</sup>, Jing Zhang<sup>3 †</sup> </em>
    <br><br>
       	<sup>1</sup> National University of Defense Technology, China, <sup>2</sup> Tsinghua University, China, <br/> <sup>3</sup> Wuhan University, China, <sup>4</sup> Beijing University of Posts and Telecommunications, China
    </h5>
</div>

<h5 align="center">
<a href="https://huggingface.co/initiacms/RoMA"> <img src="https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg"></a> <a href="https://arxiv.org/abs/2503.10392"> <img src="https://img.shields.io/badge/Arxiv-2503.10392-b31b1b.svg?logo=arXiv"></a>
</h5>

# 📚 Contents

- [News](#news)
- [Abstract](#abstract)
- [Overview](#overview)
- [Evaluation Results](#evaluation-results)
- [Scaling Behavior](#scaling-behavior)
- [Pretraining](#pretraining)
- [Checkpoints](#checkpoints)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

# 🔥News
* **[2025.09.19]** : **RoMA has been accepted by NeurlPS 2025.**
* **[2025.03.13]**  The paper is available on [arXiv](http://arxiv.org/abs/2503.10392).

# 📄Abstract

Recent advances in self-supervised learning for Vision Transformers (ViTs) have fueled breakthroughs in remote sensing (RS) foundation models. However, the quadratic complexity of self-attention poses a significant barrier to scalability, particularly for large models and high-resolution images. While the linear-complexity Mamba architecture offers a promising alternative, existing RS applications of Mamba remain limited to supervised tasks on small, domain-specific datasets. To address these challenges, we propose RoMA, a framework that enables scalable self-supervised pretraining of Mamba-based RS foundation models using large-scale, diverse, unlabeled data. 

# 🔍Overview

<figure>
<img src="assets/image-20250311170540530.png">
<figcaption align = "center"><b>Figure 1: Overview of the RoMA Pretraining Pipeline. 
 </b></figcaption>
</figure>

The input image is first divided into patches, and high-value patches are selected for random rotation using the Adaptive Rotation Encoding Strategy. These patches are then tokenized and processed by the Mamba encoder. The encoded features undergo autoregressive next-token prediction, followed by a multi-scale strategy that computes losses at different scales for gradient updates. RoMA optimally adapts the Mamba architecture for remote sensing, making its encoder a robust feature extractor for diverse downstream tasks.

# ✅Evaluation Results

<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <caption style="caption-side: top; text-align: center; font-weight: bold; margin-bottom: 10px;">
    Results for scene classification, change detection, and semantic segmentation.<br>
    “TR” represents the ratio of training data.<br>
   <sup>★</sup> indicates results from MA3E and MTP.<br>
      <sup>†</sup> denotes our reproduction with their official code.
  </caption>
  <colgroup>
    <col style="width: 25%;"> <!-- Set first column width -->
    <col> <!-- Auto-width for the other columns -->
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
  </colgroup>
<thead>
  <tr>
    <th rowspan="3">Methods</th>
    <th rowspan="3">Publication</th>
    <th rowspan="3">Backbone</th>
    <th rowspan="3">Params</th>
    <th colspan="2">Scene Classification</th>
    <th colspan="1">Change Detection</th>
    <th colspan="1">Semantic Segmentation</th>
  </tr>
  <tr>
    <th>AID</th>
    <th>UCM</th>
    <th>OSCD</th>
    <th>SpaceNetv1</th>
  </tr>
  <tr>
    <th>OA(TR=50%)</th>
    <th>OA(TR=50%)</th>
    <th>F1</th>
    <th>mF1</th>
  </tr>
</thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left;"><em style="color: gray;">Natural Image pretraining</em></td>
    </tr>
    <tr>
      <td>MoCo v3<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>78.72</td>
      <td>38.34</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DINO<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>78.51</td>
      <td>40.04</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>MAE<sup>★</sup></td>
      <td>CVPR'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>84.21</td>
      <td>52.75</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SimMIM<sup>★</sup></td>
      <td>CVPR'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>83.19</td>
      <td>51.48</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LoMaR<sup>★</sup></td>
      <td>Arxiv'22</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>82.26</td>
      <td>51.89</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>MixMAE<sup>★</sup></td>
      <td>CVPR'23</td>
      <td>Swin-B/W14</td>
      <td>88M</td>
      <td>81.53</td>
      <td>50.63</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARM<sup>†</sup></td>
      <td>ICLR'25</td>
      <td>Mamba-B</td>
      <td>85M</td>
      <td>81.14</td>
      <td>50.41</td>
      <td>47.28</td>
      <td>77.89</td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left;"><em style="color: gray;">RS Image pretraining</em></td>
    </tr>
    <tr>
      <td>SeCo<sup>★</sup></td>
      <td>ICCV'21</td>
      <td>ResNet-50</td>
      <td>25.6M</td>
      <td>78.26</td>
      <td>47.45</td>
      <td>47.67</td>
      <td>77.09</td>
    </tr>
    <tr>
      <td>CACo<sup>★</sup></td>
      <td>CVPR'23</td>
      <td>ResNet-50</td>
      <td>25.6M</td>
      <td>77.81</td>
      <td>40.53</td>
      <td>52.11</td>
      <td>77.94</td>
    </tr>
    <tr>
      <td>SatMAE<sup>★</sup></td>
      <td>NIPS'22</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>55.10</td>
      <td>34.28</td>
      <td>52.76</td>
      <td>78.07</td>
    </tr>
    <tr>
      <td>ScaleMAE<sup>★</sup></td>
      <td>ICCV'23</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>48.46</td>
      <td>28.19</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GFM<sup>★</sup></td>
      <td>ICCV'23</td>
      <td>Swin-B</td>
      <td>88M–</td>
      <td>80.58</td>
      <td>49.73</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>RVSA<sup>★</sup></td>
      <td>TGRS'23</td>
      <td>ViT-B+RVSA</td>
      <td>86M</td>
      <td>84.06</td>
      <td>50.86</td>
      <td>50.28</td>
      <td><strong>79.56</strong></td>
    </tr>
    <tr>
      <td>SatMAE++<sup>†</sup></td>
      <td>CVPR'24</td>
      <td>ViT-L</td>
      <td>307M</td>
      <td>85.98</td>
      <td>55.72</td>
      <td>53.10</td>
      <td>79.21</td>
    </tr>
    <tr>
      <td>MA3E<sup>★</sup></td>
      <td>ECCV'24</td>
      <td>ViT-B</td>
      <td>86M</td>
      <td>85.86</td>
      <td>55.69</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>RoMA</td>
      <td><a href="https://pan.baidu.com/s/1e7VOvca7894hugM-f2UitQ?pwd=e1up">Baidu</a> & <a href="https://huggingface.co/initiacms/RoMA">Hugging Face</a>
</td>
      <td>Mamba-B</td>
      <td>85M</td>
      <td><strong>87.36</strong></td>
      <td><strong>59.45</strong></td>
      <td><strong>55.63</strong></td>
      <td>79.50</td>
    </tr>
  </tbody>
</table>

For implementation of each task, please check the corresponding folder for more details.

* [Scene Classification](https://github.com/MiliLab/RoMA/tree/main/Scene%20Classification) 
* [Change Detection](https://github.com/MiliLab/RoMA/tree/main/Change%20Detection)
* [Semantic Segmentation](https://github.com/MiliLab/RoMA/tree/main/Semantic%20Segmentation)

# 📈Scaling Behavior

<figure>
<img src="assets/image-20250312111728161.png">
<figcaption align = "center"><b>Figure 2: Scaling with Data Volume. 
 </b></figcaption>
</figure>

Mamba shows a clear performance boost on downstream tasks as the pretraining data volume grows. We pretrain the Mamba-Base model with RoMA across various data scales and evaluate its performance in the downstream tasks. As illustrated in Figure 2, larger datasets lead to significant improvements. Mamba-based RSFMs exhibit no significant performance bottlenecks across a broad pretraining data scale from 62.5K to 4M, achieving data learning capabilities on par with ViT-based RSFMs. 

<figure>
<img src="assets/image-20250312112103330.png">
<figcaption align = "center"><b>Figure 3: Scaling with Model Size. 
 </b></figcaption>
</figure>

Mamba’s performance also improves with increasing model size. We conduct extensive pretraining on four model variants—Tiny, Small, Base, and Large—following the configurations in our code. As shown in Figure 3, larger models consistently achieve superior results on downstream tasks. Although Mamba-Large surpasses Mamba-Base in AID dataset, its performance gain remains limited, likely due to insufficient pretraining. With only 300 epochs on 4 million samples, the training may not be adequate for a 297M-parameter model. Due to experimental constraints, we did not extend pretraining to 800 epochs as in MAE. The OSCD and SpaceNet experiments are ongoing, with updates to follow. However, these results do not alter our key findings: Mamba-based RSFMs pretrained with RoMA demonstrate performance gains as model parameters scale. 

# 🚀Pretraining

For environment setup and pretraining instructions, please refer to [RoMA/requirements.txt](https://github.com/MiliLab/RoMA/blob/main/RoMA/requirements.txt)  and [RoMA/train.sh](https://github.com/MiliLab/RoMA/blob/main/RoMA/train.sh).

# 🎯Checkpoints

We provide our pretrained weights in <a href="https://pan.baidu.com/s/1e7VOvca7894hugM-f2UitQ?pwd=e1up">Baidu</a> & <a href="https://huggingface.co/initiacms/RoMA">Hugging Face</a>.

# 🔗Citation

If you find RoMA helpful, please consider citing:

```latex
@article{wang2025roma,
  title={RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing},
  author={Fengxiang Wang and Hongzhen Wang and Yulin Wang and Di Wang and Mingshuo Chen and Haiyan Zhao and Yangang Sun and Shuo Wang and Long Lan and Wenjing Yang and Jing Zhang},
  journal={arXiv preprint arXiv:2503.10392},
  year={2025}
}
```

# 🤝Acknowledgement

* [ARM](https://github.com/OliverRensu/ARM/tree/main): Autoregressive Pretraining with Mamba in Vision.
* [MTP](https://github.com/ViTAE-Transformer/MTP): MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining.
* [RSP](https://github.com/ViTAE-Transformer/RSP): An Empirical Study of Remote Sensing Pretraining.
* [open-cd](https://github.com/likyoo/open-cd): An open source change detection toolbox based on a series of open source general vision task tools.
* [mmcv](https://github.com/open-mmlab/mmcv), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Toolbox.
