# ThemeStation: Generating Theme-Aware 3D Assets from Few Exemplars

## NEWS: 
- [Sep. 2024] Welcome to follow [Phidias](https://rag-3d.github.io/), our new 3D generative model that supports fast theme-aware 3D-to-3D. Code will be released soon at [here](https://github.com/3DTopia/Phidias-Diffusion).

- [Apr. 2024] ThemeStation has been accepted to SIGGRAPH 2024!


https://github.com/3DThemeStation/ThemeStation/assets/158151171/cbd2b81b-b224-4bac-92fd-4f612df77172


### [Project page](https://3dthemestation.github.io/) |   [Paper](https://arxiv.org/abs/2403.15383) | [Video](https://www.youtube.com/watch?v=q6afxQXRl_o)

<!-- <br> -->
[Zhenwei Wang](https://zhenwwang.github.io/), [Tengfei Wang](https://tengfei-wang.github.io/), [Gerhard Hancke](https://rfidblog.org.uk/), [Ziwei Liu](https://liuziwei7.github.io/) and [Rynson W.H. Lau](https://www.cs.cityu.edu.hk/~rynson/).
<!-- <br> -->

## Abstract
>Real-world applications often require a large gallery of 3D assets that share a consistent theme. While remarkable advances have been made in general 3D content creation from text or image, synthesizing customized 3D assets following the shared theme of input 3D exemplars remains an open and challenging problem. In this work, we present ThemeStation, a novel approach for theme-aware 3D-to-3D generation. ThemeStation synthesizes customized 3D assets based on given few exemplars with two goals: 1) unity for generating 3D assets that thematically align with the given exemplars and 2) diversity for generating 3D assets with a high degree of variations. To this end, we design a two-stage framework that draws a concept image first, followed by a reference-informed 3D modeling stage. We propose a novel dual score distillation (DSD) loss to jointly leverage priors from both the input exemplars and the synthesized concept image. Extensive experiments and user studies confirm that ThemeStation surpasses prior works in producing diverse theme-aware 3D models with impressive quality. ThemeStation also enables various applications such as controllable 3D-to-3D generation.

## Overview
<div class="half">
    <img src="figures/overview.png" width="1080">
</div>



## Todo (Latest update: 2024/04/26)
- [x] **Release rendering code**
- [x] **Release StageI code for concept image generation**
- [x] **Release StageII code for reference-informed 3D assets modeling**
- [ ] **Update our code to support more powerful image-to-3d models, such as InstantMesh/CRM to obtain the initial 3D**



## Installation
- Setup:
```bash
    conda create -n themestation python==3.10
    conda activate themestation
    pip install -r requirements.txt
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


- We use [Wonder3D](https://github.com/xxlong0/Wonder3D?tab=readme-ov-file) to obtain the initial 3D model.
    - Download the [checkpoints](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EgSHPyJAtaJFpV_BjXM3zXwB-UMIrT4v-sQwGgw-coPtIA) into folder `Wonder3D/ckpts`.
    - Make sure you have the following models.
    ```bash
    Wonder3D
    |-- ckpts
        |-- unet
        |-- scheduler.bin
        ...
    ```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:
```bash
    pip install ninja
```


- (Optional) tiny-cuda-nn installation might require downgrading pip to 23.0.1

- [Stable Diffusion](https://huggingface.co/models?other=stable-diffusion). We use diffusion prior from a pretrained 2D Stable Diffusion 2.1 model. To start with, you may need a huggingface [token](https://huggingface.co/settings/tokens) to access the model, or use `huggingface-cli login` command.


## Preparation 
- Place reference 3D models under `data/reference_models/models` and install [Blender3.2.2](https://download.blender.org/release/Blender3.2/)

- Render images for generating concept images:
```bash
    cd data/reference_models
    sh render_script/render_single_concept.sh obj_file_name blender_path elevation
    # For example:
    # sh render_script/render_single_concept.sh 20 owl.glb ~/blender-3.2.2-linux-x64/blender
```

- Render images for learning reference prior:
```bash
    # RGB
    cd data/reference_models
    sh render_script/render_single_ref_rgb.sh obj_file_name blender_path

    # Normal (optional)
    cd data/reference_models
    sh render_script/render_single_ref_normal.sh obj_file_name blender_path
    
    # For example:
    # sh render_script/render_single_ref_rgb.sh owl.glb ~/blender-3.2.2-linux-x64/blender
```

- Make sure you have the following files.
```bash
ThemeStation
| -- data
    | -- reference_models
        | -- renderings
            | -- elevation_n (used to generate concept images with elevation = n)
                | -- object_name
                    | -- 000.png
                    ...
            | -- elevation_rand (used to learn reference prior)
                | -- object_name
                    | -- 000.png
                    ...
                    | -- normals_000_0001.png (optional)
                    ...
```

## Inference 
### Stage I: Concept image generation

> For now, you can simply use [Freepik-reimagine](https://www.freepik.com/pikaso/reimagine) for concept image generation given the rendered front view of a given 3D exemplar.

We also show the steps as introduced in our paper below:
- to generate concept images, run:
```bash
    cd dreambooth
    sh scripts/generate_concept.sh 20 # 20 is elevation, edit this template for you own data

    # for controllable-3D-to-3D using text run:
    sh scripts/generate_concept_control.sh 20
    # results are saved to `dreambooth/outputs/...`
```

- select the concept images you like and remove their background using [Clipdrop](https://clipdrop.co/remove-background) or Photoshop and put the final concept images (with transparant background) under `data/concept_images`

### Stage II: Reference-informed 3D asset modeling

Run ThemeStation to generate a final 3D model given the concept image and reference model.

- run a specific concept image
```bash
    sh scrips/run.sh # edit this template script for your own data
```

- run a batch of concept images
```bash
    sh scrips/run_batch.sh # edit this template script for your own data
```

- export mesh
```bash
    sh scripts/export_mesh.sh # edit this template script for your own data
```




## Citation
If you find this code helpful for your research, please cite:
```
@article{wang2024themestation,
        title={ThemeStation: Generating Theme-Aware 3D Assets from Few Exemplars}, 
        author={Zhenwei Wang and Tengfei Wang and Gerhard Hancke and Ziwei Liu and Rynson W.H. Lau},
        booktitle={ACM SIGGRAPH},
        year={2024}
  }
```

## Acknowledgments
We have intensively borrowed codes from the following repositories. Many thanks to the authors for sharing their codes.
- [threestudio](https://github.com/threestudio-project/threestudio)
- [Wonder3D](https://github.com/xxlong0/Wonder3D)
- [Diffusers-DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
- [Objaverse Rendering](https://github.com/allenai/objaverse-rendering)
