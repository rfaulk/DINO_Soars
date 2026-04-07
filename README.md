<div align="center">

<h1>DINO Soars: DINOv3 for Open-Vocabulary Semantic Segmentation of Remote Sensing Imagery</h1>

<h3>💖CVPRW: Morse 2026💖</h3>

<div>
    Ryan Faukenberry&emsp;
    Saurabh Prasad&emsp;
</div>
<div>
    University of Houston&emsp;
</div>

<div>
    <h4 align="center">
        • todo arXiv link go here •
    </h4>
</div>

<img src="https://github.com/rfaulk/DINO_Soars/blob/main/dinosoars.png" width="100%"/>
Open-vocabulary segmentation maps from our model, CAFe-DINO. CAFe-DINO can accurately segment remote sensing scenes
with arbitrary semantic classes, despite zero training on remote sensing imagery.

</div>

## Abstract
> *The remote sensing (RS) domain suffers from a lack of
densely labeled datasets, which are costly to obtain. Thus,
models that can segment RS imagery well without supervised fine-tuning are valuable, but existing solutions fall
behind supervised methods. Recently, DINOv3 surpassed SOTA RS foundation models on the GEO-bench segmentation benchmark without pre-training on RS data. Additionally, DINO.txt has enabled open vocabulary semantic segmentation (OVSS) with the DINOv3 backbone.
We leverage these developments to form an OVSS model for RS imagery, free of RS-domain fine-tuning. Our model,
CAFe-DINO (Cost Aggregation + Feature Upsampling with
DINO) exploits the strong OVSS performance of DINOv3
for RS imagery via cost aggregation and training-free upsampling of text-image similarity scores. The robust latent
of the DINOv3 backbone eliminates the need for fine-tuning
on RS imagery; we instead fine-tune our model on a RS-targeted subset of COCO-Stuff. CAFe-DINO achieves state-of-the-art performance on key RS segmentation datasets,
outperforming OVSS methods fine-tuned on RS data*

## Dependencies and Installation


```
# 1. git clone this repository
git clone https://github.com/rfaulk/DINO_Soars.git
cd DINO_Soars

# 2. create new anaconda env
conda create -n cafedino python=3.11
conda activate cafedino

# 3. install torch and dependencies
Install torch 2.11 first: (https://pytorch.org/get-started/locally/) or (https://pytorch.org/get-started/previous-versions/)
pip install -r requirements.txt
```

## Usage

### Training
You will need to reset some paths in train.py to match your filetree. To train our best performing model:

```python train.py --config configs/config_cocostuff_subset_frz_text.yaml```

### Validation
You will again need to set your own paths in these python files. We offer validation with and without background classes. To reproduce our model's result:

```
python validate_strided.py --config configs/config_cocostuff_subset_frz_text.yaml --model ../best.pth
python validate_strided_with_bg.py --config configs/config_cocostuff_subset_frz_text.yaml --model ../best.pth
```

### Inference and analysis
We provide a tool that shows both the segmentation prediction and each cost map for a given image. Run like so:

```python analysis.py --weights ../best.pth --image ./sample_images/top_potsdam_4_13_RGB_y00_x00.tif```

## Citation
todo

## Acknowledgement
todo
