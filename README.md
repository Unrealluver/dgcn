<div align="center">   
  
# Deep Graph 🔪 Network
</div>

**TL;DR:**  We study the **Graph Cut** algorithm on deep neural network with **Weakly Supervised Semantic Segmentation** 
task, and achieved SOTA performence at that time.

:man_technologist: This project is under active development :woman_technologist: :


* **`Jun 29, 2021`:**  We release [this](https://github.com/Unrealluver/dgcn) repo that contains more results, check it out!

* **`Jan 26, 2021`:**  We firstly release our [DGCN](https://github.com/hustvl/DGCN) repo, check it out!

#

> [**Deep Graph Cut Network for Weakly-supervised Semantic Segmentation**](http://xinggangw.info/pubs/scis-dgcn.pdf)
>
> by [Jiapei Feng] \*, [Xinggang Wang](https://xinggangw.info/)<sup>* </sup>,  [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/).
> 
> <sup>1</sup> [School of EIC, HUST](http://eic.hust.edu.cn/English/Home.htm), <sup>2</sup> Institute of AI, HUST, <sup>3</sup> [Horizon Robotics](https://en.horizon.ai).
> 
> (<sup>* </sup>) corresponding author.
> 
> *Science China Information Sciences volume  ([Published: 07 February 2021](https://link.springer.com/article/10.1007/s11432-020-3065-4))*

<br>

## Deep Graph Cut Network for Weakly-supervised Semantic Segmentation (DGCN)

### The Illustration of DGCN
![DGCN](./docs/DGCN_overview.png)

### Highlights

Directly inherited from [ViT](https://arxiv.org/abs/2010.11929) ([DeiT](https://arxiv.org/abs/2012.12877)), YOLOS is not designed to be yet another high-performance object detector, but to unveil the versatility and transferability of Transformer from image recognition to object detection.
Concretely, our main contributions are summarized as follows:

* We use the mid-sized `ImageNet-1k` as the sole pre-training dataset, and show that a vanilla [ViT](https://arxiv.org/abs/2010.11929) ([DeiT](https://arxiv.org/abs/2012.12877)) can be successfully transferred to perform the challenging object detection task and produce competitive `COCO` results with the fewest possible modifications, _i.e._, by only looking at one sequence (YOLOS).

* We demonstrate that 2D object detection can be accomplished in a pure sequence-to-sequence manner by taking a sequence of fixed-sized non-overlapping image patches as input. Among existing object detectors, YOLOS utilizes minimal 2D inductive biases. Moreover, it is feasible for YOLOS to perform object detection in any dimensional space unaware the exact spatial structure or geometry.

* For [ViT](https://arxiv.org/abs/2010.11929) ([DeiT](https://arxiv.org/abs/2012.12877)), we find the object detection results are quite sensitive to the pre-train scheme and the detection performance is far from saturating. Therefore the proposed YOLOS can be used as a challenging benchmark task to evaluate different pre-training strategies for [ViT](https://arxiv.org/abs/2010.11929) ([DeiT](https://arxiv.org/abs/2012.12877)).

* We also discuss the impacts as wel as the limitations of prevalent pre-train schemes and model scaling strategies for Transformer in vision through transferring to object detection.

### Results

| Backbone  | ResizeLong | Crop | ResizeCropped | ColorJitter | RandomAug | Cutout | Pad0 | Batchsize | LR       | Decay | Freeze-1 | Freeze01 | 10xLr LastStage | SyncBn | GPUNum | Optim | mIOU   |
| --------- | ---------- | ---- | ------------- | ----------- | --------- | ------ | ---- | --------- | -------- | ----- | -------- | -------- | --------------- | ------ | ------ | ----- | ------ |
| ResNet101 | ❌        | 321  | ❌           | ❌         | ❌       | ❌    | ❌  | 16        | 5.00E-04 | 3     | ❌      | ❌      | ✅             | ❌    | 1      | Step  | 0.615  |
| HRNetw48  | ❌        | 321  | ❌           | ❌         | ❌       | ❌    | ❌  | 16        | 5.00E-04 | 3     | ❌      | ❌      | ❌             | ❌    | 1      | Step  | 0.6239 |
| HRNetw48  | ❌        | 422  | 321           | ❌         | ❌       | ❌    | ❌  | 16        | 5.00E-04 | 3     | ❌      | ❌      | ❌             | ❌    | 1      | Step  | 0.6296 |
| HRNetw48  | [320, 640] | 512  | ❌           | ❌         | ❌       | ❌    | ❌  | 8         | 5.00E-04 | 3     | ❌      | ❌      | ❌             | ❌    | 2      | Step  | 0.6173 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 8         | 5.00E-04 | 3     | ❌      | ❌      | ❌             | ❌    | 2      | Step  | 0.6312 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 12        | 5.00E-04 | 3     | ❌      | ❌      | ❌             | ❌    | 2      | Step  | 0.6257 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 24        | 5.00E-04 | 3     | ❌      | ❌      | ✅             | ✅    | 4      | Step  | 0.6328 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ✅       | ✅    | ❌  | 24        | 5.00E-04 | 3     | ❌      | ❌      | ✅             | ✅    | 4      | Step  | 0.6241 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ✅       | ✅    | ❌  | 6         | 5.00E-04 | 3     | ❌      | ❌      | ✅             | ❌    | 1      | Step  | 0.6164 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ✅    | ❌  | 12        | 5.00E-04 | 6     | ❌      | ❌      | ✅             | ✅    | 2      | Step  | 0.6183 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ✅    | ❌  | 12        | 1.00E-03 | 3     | ❌      | ❌      | ✅             | ✅    | 2      | Step  | 0.6121 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ✅    | ❌  | 12        | 7.50E-04 | 3     | ❌      | ❌      | ✅             | ✅    | 2      | Step  | 0.6204 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 12        | 2.50E-04 | 3     | ❌      | ❌      | ✅             | ✅    | 2      | Step  | 0.6343 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 12        | 2.50E-04 | 3     | ❌      | ✅      | ✅             | ✅    | 2      | Step  | 0.6315 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ✅       | ❌    | ❌  | 12        | 2.50E-04 | 3     | ❌      | ✅      | ✅             | ✅    | 2      | Poly  | 0.6321 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ✅  | 12        | 2.50E-04 | 3     | ❌      | ✅      | ✅             | ✅    | 2      | Step  | 0.6327 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ✅  | 12        | 2.50E-04 | 3     | ❌      | ✅      | ✅             | ✅    | 2      | Poly  | 0.6348 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 12        | 2.50E-04 | 3     | ✅      | ❌      | ✅             | ✅    | 2      | Step  | 0.6312 |
| HRNetw48  | [320, 640] | 512  | ❌           | ✅         | ❌       | ❌    | ❌  | 12        | 2.50E-04 | 3     | ✅      | ❌      | ✅             | ✅    | 2      | Poly  | 0.6296 |

**Notes**: 

- The access code for `Baidu Drive` is `yolo`. 
- The `FB` stands for model weights provided by DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.
- We will update other models in the future, please stay tuned :) 

### Requirement
This codebase has been developed with python version 3.6, PyTorch 1.5+ and torchvision 0.6+:
```setup
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```setup
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### Data preparation
Download and extract COCO 2017 train and val images with annotations from http://cocodataset.org. We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
### Training
Before finetuning on COCO, you need download the ImageNet pretrained model to the `/path/to/YOLOS/` directory
<details>
<summary>To train the <code>YOLOS-Ti</code> model in the paper, run this command:</summary>
<pre><code>
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 2 \
    --lr 5e-5 \
    --epochs 300 \
    --backbone_name tiny \
    --pre_trained /path/to/deit-tiny.pth\
    --eval_size 512 \
    --init_pe_size 800 1333 \
    --output_dir /output/path/box_model
</code></pre>
</details>

<details>
<summary>To train the <code>YOLOS-S</code> model with 200 epoch pretrained Deit-S in the paper, run this command:</summary>
<pre><code>

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 1 \
    --lr 2.5e-5 \
    --epochs 150 \
    --backbone_name small \
    --pre_trained /path/to/deit-small-200epoch.pth\
    --eval_size 800 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --output_dir /output/path/box_model

</code></pre>
</details>

<details>
<summary>To train the <code>YOLOS-S</code> model with 300 epoch pretrained Deit-S in the paper, run this command:</summary>
<pre><code>
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 1 \
    --lr 2.5e-5 \
    --epochs 150 \
    --backbone_name small \
    --pre_trained /path/to/deit-small-300epoch.pth\
    --eval_size 800 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --output_dir /output/path/box_model

</code></pre>
</details>

<details>
<summary>To train the <code>YOLOS-S (dWr)</code> model in the paper, run this command:</summary>
<pre><code>
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 1 \
    --lr 2.5e-5 \
    --epochs 150 \
    --backbone_name small_dWr \
    --pre_trained /path/to/deit-small-dWr-scale.pth\
    --eval_size 800 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --output_dir /output/path/box_model
</code></pre>
</details>

<details>
<summary>To train the <code>YOLOS-B</code> model in the paper, run this command:</summary>
<pre><code>
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 1 \
    --lr 2.5e-5 \
    --epochs 150 \
    --backbone_name base \
    --pre_trained /path/to/deit-base.pth\
    --eval_size 800 \
    --init_pe_size 800 1344 \
    --mid_pe_size 800 1344 \
    --output_dir /output/path/box_model
</code></pre>
</details>


### Evaluation

To evaluate `YOLOS-Ti` model on COCO, run:

```eval
python main.py --coco_path /path/to/coco --batch_size 2 --backbone_name tiny --eval --eval_size 512 --init_pe_size 800 1333 --resume /path/to/YOLOS-Ti
```
To evaluate `YOLOS-S` model on COCO, run:
```eval
python main.py --coco_path /path/to/coco --batch_size 1 --backbone_name small --eval --eval_size 800 --init_pe_size 512 864 --mid_pe_size 512 864 --resume /path/to/YOLOS-S
```
To evaluate `YOLOS-S (dWr)` model on COCO, run:
```eval
python main.py --coco_path /path/to/coco --batch_size 1 --backbone_name small_dWr --eval --eval_size 800 --init_pe_size 512 864 --mid_pe_size 512 864 --resume /path/to/YOLOS-S(dWr)
```

To evaluate `YOLOS-B` model on COCO, run:
```eval
python main.py --coco_path /path/to/coco --batch_size 1 --backbone_name small --eval --eval_size 800 --init_pe_size 800 1344 --mid_pe_size 800 1344 --resume /path/to/YOLOS-B
```

### Visualization

* **Visualize box prediction and object categories distribution:**


1. To Get visualization in the paper, you need the finetuned YOLOS models on COCO, run following command to get 100 Det-Toks prediction on COCO val split, then it will generate `/path/to/YOLOS/visualization/modelname-eval-800-eval-pred.json`
```
python cocoval_predjson_generation.py --coco_path /path/to/coco --batch_size 1 --backbone_name small --eval --eval_size 800 --init_pe_size 512 864 --mid_pe_size 512 864 --resume /path/to/yolos-s-model.pth --output_dir ./visualization
```
2. To get all ground truth object categories on all images from COCO val split, run following command to generate `/path/to/YOLOS/visualization/coco-valsplit-cls-dist.json`
```
python cocoval_gtclsjson_generation.py --coco_path /path/to/coco --batch_size 1 --output_dir ./visualization
```
3. To visualize the distribution of Det-Toks' bboxs and categories, run following command to generate `.png` files in `/path/to/YOLOS/visualization/`
```
 python visualize_dettoken_dist.py --visjson /path/to/YOLOS/visualization/modelname-eval-800-eval-pred.json --cococlsjson /path/to/YOLOS/visualization/coco-valsplit-cls-dist.json
```
![cls](visualization/yolos_s_300_pre.pth-eval-800eval-pred-bbox.png)
![cls](./visualization/yolos_s_300_pre.pth-eval-800eval-pred-all-tokens-cls.png)


* **Use [VisualizeAttention.ipynb](VisualizeAttention.ipynb) to visualize self-attention of `[Det]` tokens on different heads of the last layer:**

![Det-Tok-41](visualization/exp/Det-Tok-41/Det-Tok-41_attn.png)
![Det-Tok-96](visualization/exp/Det-Tok-96/Det-Tok-96_attn.png)

## Acknowledgement :heart:
This project is based on DETR ([paper](https://arxiv.org/abs/2005.12872), [code](https://github.com/facebookresearch/detr)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)), DINO ([paper](https://arxiv.org/abs/2104.14294), [code](https://github.com/facebookresearch/dino)) and [timm](https://github.com/rwightman/pytorch-image-models). Thanks for their wonderful works.



## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :

```BibTeX
@article{YOLOS,
  title={You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
  author={Fang, Yuxin and Liao, Bencheng and Wang, Xinggang and Fang, Jiemin and Qi, Jiyang and Wu, Rui and Niu, Jianwei and Liu, Wenyu},
  journal={arXiv preprint arXiv:2106.00666},
  year={2021}
}
```
