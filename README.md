
<h2 align="center">
  LINEA: Fast and accurate line detection using scalable transformers
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/LINEA/master/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="https://colab.research.google.com/github/SebastianJanampa/LINEA/blob/master/LINEA_tutorial.ipynb">
        <img alt="colab" src="https://img.shields.io/badge/-colab-blue?style=for-the-badge&logo=googlecolab&logoColor=white&labelColor=%23daa204&color=yellow">
  </a>

  <a href='https://huggingface.co/spaces/SebasJanampa/LINEA'>
      <img src='https://img.shields.io/badge/-SPACE-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF5500&color=orange'>
   </a>
   
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
   LINEA: Fast and accurate line detection using scalable transformers
</p>


<p align="center">
Sebastian Janampa and Marios Pattichis
</p>

<p align="center">
The University of New Mexico
  <br>
Department of Electrical and Computer Engineering
</p>

LINEA is a powerful real-time line detector that introduces Line Attention mechanism,
achieving outstanding performance without being pretrained on COCO or Object365 datasets. 

<details open>
<summary> Attention Mechanishm </summary>

We compare line attention with traditional attention and deformable attention.
We highlight two advantages of our proposed mechanism:

- Line attention is a sparse mechanism like deformable attention. This significantly reduces memory complexity.
- Line attention pays attention to the line endpoints like traditional attention but also attends locations between the endpoints.

</details>

## 🚀 Updates
- [x] **\[2025.02.22\]** LINEA has been accepted in ICIP 2025
- [x] **\[2025.02.22\]** Fix bug and reduce FLOPs
- [x] **\[2025.02.19\]** Release LINEA series.
- [x] **\[2025.02.20\]** Release LINEA weights.
- [x] **\[2025.02.20\]** Release [Google Colab Notebook](https://colab.research.google.com/github/SebastianJanampa/LINEA/blob/master/LINEA_tutorial.ipynb).
- [x] **\[2025.02.20\]** Release [HuggingFace 🤗 Space](https://huggingface.co/spaces/SebasJanampa/LINEA).

## 📝 TODO
- [ ] Upload paper (accepted in ICIP 2025)
- [x] Upload requirements
- [x] Upload LINEA weigths.
- [x] Create HuggingFace 🤗 space.
- [x] Create Collab demo.

## Model Zoo

### Wireframe
| Model | Dataset | AP<sup>5</sup> | AP<sup>10</sup> | AP<sup>15</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**LINEA&#8209;N** | Wireframe | **58.7** | **65.0** | **67.9** | 3.9M | 2.54ms / 2.50ms | 12.1 / 11.5 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_n.py) | [65.0](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_n.pth) | 
**LINEA&#8209;S** | Wireframe | **58.4** | **64.7** | **67.6** | 8.6M | 3.08ms / 3.07ms | 31.7 / 29.4 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_s.py) | [64.7](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_m.pth) | 
**LINEA&#8209;M** | Wireframe | **59.5** | **66.3** | **69.1** | 13.5M | 3.87ms / 3.79ms | 45.6 / 43.4 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_m.py) | [66.3](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_m.pth) | 
**LINEA&#8209;L** | Wireframe | **61.0** | **67.9** | **70.8** | 25.2M | 5.78ms /5.42ms | 83.8 / 81.5 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_l.py) | [67.9](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_l.pth) |

### YorkUrban
| Model | Dataset | AP<sup>5</sup> | AP<sup>10</sup> | AP<sup>15</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**LINEA&#8209;N** | YorkUrban | **27.3** | **30.5** | **32.5** | 3.9M | 2.54ms / 2.50ms | 12.1 / 11.5 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_n.py) | [65.0](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_n.pth) | 
**LINEA&#8209;S** | YorkUrban | **28.9** | **32.6** | **34.8** | 8.6M | 3.08ms / 3.07ms | 31.7 / 29.4 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_s.py) | [64.7](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_s.pth) | 
**LINEA&#8209;M** | YorkUrban | **30.3** | **34.5** | **36.7** | 13.5M | 3.87ms / 3.79ms | 45.6 / 43.4 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_m.py) | [66.3](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_m.pth) | 
**LINEA&#8209;L** | YorkUrban | **30.9** | **34.9** | **37.3** | 25.2M | 5.78ms /5.42ms | 83.8 / 81.5 | [py](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_l.py) | [67.9](https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_l.pth) |

**Notes:**
- **Latency** is evaluated on a single NVIDIA RTX A5500 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==10.5.0$.



## Quick start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lqxe4ruXR_ly9YPxmOUwY4CA4sFEmL43#scrollTo=nthoulkvpQn3)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/SebasJanampa/LINEA)

### Setup

```shell
conda create -n linea python=3.11.9
conda activate linea
pip install -r requirements.txt
```


### Data Preparation

To reproduce our results, you need to process two datasets, [ShanghaiTech](https://github.com/huangkuns/wireframe) and [YorkUrban](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/). 

```shell
mkdir data
cd data
wget https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/wireframe_processed.zip
wget https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/york_processed.zip

unzip wireframe_processed.zip
unzip york_processed.zip

rm *zip
cd ..
```


## Usage
<details open>
<summary> Wireframe </summary>

<!-- <summary>1. Training </summary> -->
1. Set Model
```shell
export model=l  # n s m l
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 main.py -c configs/linea/linea_hgnetv2_${model}.py --coco_path data/wireframe_processed --amp 
```

<!-- <summary>2. Testing </summary> -->
3. Testing
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_${model}.py --coco_path data/wireframe_processed --amp  --eval --resume <checkpoit.pth>
```

4. Replicate results (optional)
```shell
# First, download the official weights
wget https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_${model}.pth

# Second, run test
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_${model}.py --coco_path data/wireframe_processed --amp  --eval --resume linea_hgnetv2_${model}.pth
```

</details>

<details open>
<summary> YorkUrban </summary>

<!-- <summary>1. Training </summary> -->
1. Set Model
```shell
export model=l  # n s m l
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_${model}.py --coco_path data/york_processed --amp  --eval --resume linea_hgnetv2_${model}.pth
```

3. Replicate results (optional)
```shell
# First, download the official weights
wget https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_${model}.pth

# Second, run test
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_${model}.py --coco_path data/york_processed --amp  --eval --resume linea_hgnetv2_${model}.pth
```

</details>

<details>
<summary> Customizing Batch Size </summary>

For example, if you want to train with a total batch size of 16 when training **LINEA-L** on Wireframe:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 main.py -c configs/linea/linea_hgnetv2_l.py --coco_path data/wireframe_processed --amp --options batch_size_train=16
```
</details>

<details>
<summary> Customizing Input Size </summary>

If you'd like to train **LINEA-L** on Wireframe with an input size of 320x320 (we only support square shapes):
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_l.py --coco_path data/wireframe_processed --amp --options eval_spatial_size=320,320
```
or 
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_l.py --coco_path data/wireframe_processed --amp --options eval_spatial_size=320
```

</details>

<details>
<summary> Multiple Costumizations </summary>

If you'd like to train **LINEA-L** on Wireframe with an input size of 480x480 and a total batch size of 4:

```shell
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 main.py -c configs/linea/linea_hgnetv2_l.py --coco_path data/wireframe_processed --amp --options eval_spatial_size=320 batch_size_train=4
```

</details>

## Tools
<details>
<summary> Deployment </summary>

<!-- <summary>4. Export onnx </summary> -->
1. Setup
```shell
pip install onnx onnxsim
export model=l  # n s m l
```

2. Export onnx
```shell
python tools/deployment/export_onnx.py --check -c configs/linea/linea_hgnetv2_${model}.py -r linea_hgnetv2_${model}.pth
```

3. Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
For a specific file
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

or, for all files inside a folder
```shell
python tools/deployment/export_tensorrt.py
```

</details>

<details>
<summary> Inference (Visualization) </summary>


1. Setup
```shell
pip install -r tools/inference/requirements.txt
export model=l  # n s m l
```


<!-- <summary>5. Inference </summary> -->
2. Inference (onnxruntime / tensorrt / torch)

Inference on images and videos is supported.

For a single file
```shell
python tools/inference/onnx_inf.py --onnx linea_hgnetv2_${model}.onnx --input example/example1.jpg  
python tools/inference/trt_inf.py --trt linea_hgnetv2_${model}.engine --input example/example1.jpg
python tools/inference/torch_inf.py -c configs/linea/linea_hgnetv2_${model}.py -r <checkpoint.pth> --input example/example1.jpg --device cuda:0
```

For a folder
```shell
python tools/inference/onnx_inf.py --onnx linea_hgnetv2_${model}.onnx --input example  
python tools/inference/trt_inf.py --trt linea_hgnetv2_${model}.engine --input example
python tools/inference/torch_inf.py -c configs/linea/linea_hgnetv2_${model}.py -r linea_hgnetv2_${model}t.pth --input example --device cuda:0
```
</details>

<details>
<summary> Benchmark </summary>

1. Setup
```shell
pip install -r tools/benchmark/requirements.txt
export model=l  # n s m l
```

<!-- <summary>6. Benchmark </summary> -->
2. Model FLOPs, MACs, and Params
```shell
python tools/benchmark/get_info.py --config configs/linea/linea_hgnetv2_${model}.py 
```

3. TensorRT Latency
```shell
python tools/benchmark/trt_benchmark.py --infer_dir ./data/wireframe_processed/val2017 --engine_dir trt_engines
```

4. Pytorch Latency
```shell
python tools/benchmark/torch_benchmark.py -c ./configs/linea/linea_hgnetv2_${model}.py --resume linea_hgnetv2_${model}.pth --infer_dir ./data/wireframe_processed/val2017
```
</details>


<details>
<summary> Visualization: Line attention </summary>
  
```shell
python tools/visualization/line_attention.py -c ./configs/linea/linea_hgnetv2_${model}.py --resume linea_hgnetv2_${model}.pth --data-path ./data/wireframe_processed -d cuda --num_images 10
```
</details>

<details>
<summary> Visualization: Feature maps from the backbone and encoder </summary>
  
``` shell
python tools/visualization/backbone_encoder.py -c ./configs/linea/linea_hgnetv2_${model}.py --resume linea_hgnetv2_${model}.pth --data-path ./data/wireframe_processed -d cuda --num_images 10
```
</details>



## Citation
If you use `LINEA` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```latex
TODO
```
</details>

## Acknowledgement
Our work is built upon [DT-LSD](https://github.com/SebastianJanampa/DT-LSD) and [D-FINE](https://github.com/Peterande/D-FINE).
Thanks to the inspirations from [DT-LSD](https://github.com/SebastianJanampa/DT-LSD), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), and [LETR](https://github.com/mlpc-ucsd/LETR).

✨ Feel free to contribute and reach out if you have any questions! ✨


