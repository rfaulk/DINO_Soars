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
# 2. git clone this repository
git clone https://github.com/rfaulk/DINO_Soars.git
cd todo

# 3. create new anaconda env
conda create -n cafedino python=3.9
conda activate cafedino

# install torch and dependencies
pip install -r requirements.txt
```



## Citation

```

```

## Acknowledgement
This implementation is based on [ClearCLIP](https://github.com/mc-lan/ClearCLIP) and [FeatUp](https://github.com/mhamilton723/FeatUp). Thanks for the awesome work.
