[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![arXiv](https://img.shields.io/badge/cs.CV-arxiv%3A2402.00627-red)](https://arxiv.org/abs/2402.00627)

<p align="center">
<img src="assets/logo.png" width="100">
</p>

# CapHuman: Capture Your Moments in Parallel Universes

[[Paper]](https://arxiv.org/abs/2402.00627) [[Project Page]](https://caphuman.github.io/)

This is the repository for the paper *CapHuman: Capture Your Moments in Parallel Universes*.

Chao Liang<sup></sup>,&nbsp;[Fan Ma](https://flowerfan.site/),&nbsp;[Linchao Zhu](https://ffmpbgrnn.github.io/)<sup></sup>,&nbsp;[Yingying Deng](https://diyiiyiii.github.io/),&nbsp;[Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en).&nbsp;

<img src='./assets/teaser.png' width=850>

We concentrate on a novel human-centric image synthesis task, that is, given only one reference facial photograph, it is expected to generate specific individual images with diverse head positions, poses, and facial expressions in different contexts. To accomplish this goal, we argue that our generative model should be capable of the following favorable characteristics: (1) a strong visual and semantic understanding of our world and human society for basic object and human image generation. (2) generalizable identity preservation ability. (3) flexible and fine-grained head control. Recently, large pre-trained text-to-image diffusion models have shown remarkable results, serving as a powerful generative foundation. As a basis, we aim to unleash the above two capabilities of the pre-trained model. In this work, we present a new framework named CapHuman. We embrace the ``encode then learn to align" paradigm, which enables generalizable identity preservation for new individuals without cumbersome tuning at inference. CapHuman encodes identity features and then learns to align them into the latent space. Moreover, we introduce the 3D facial prior to equip our model with control over the human head in a flexible and 3D-consistent manner. Extensive qualitative and quantitative analyses demonstrate our CapHuman can produce well-identity-preserved, photo-realistic, and high-fidelity portraits with content-rich representations and various head renditions, superior to established baselines.

## :flags: News
- [2024/04/26] We release the code and checkpoint.
- [2024/02/27] Our paper is accepted by CVPR2024.
- [2024/02/01] We release the [Project Page](https://caphuman.github.io/).

## :hammer: Installation

### Dependency

```bash
conda create -n caphuman python=3.7
pip install -r requirements.txt
```
Following [INSTALL](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install pytorch3d (e.g. 0.7.4, 0.7.6)

Following [adobe-research/diffusion-rig](https://github.com/adobe-research/diffusion-rig?tab=readme-ov-file#deca-setup) for DECA setup.


```
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```

Download our checkpoint [caphuman.ckpt](https://huggingface.co/VamosC/CapHuman/tree/main), [vae-ft-mse-840000-ema-pruned.ckpt](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt), [Realistic_Vision_V3.0.ckpt](https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE/resolve/main/Realistic_Vision_V3.0.ckpt), [79999_iter.pth](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) and put them into ckpts.

```
ckpts/
  face-parsing/
    79999_iter.pth
  caphuman.ckpt
  Realistic_Vision_V3.0.ckpt
  vae-ft-mse-840000-ema-pruned.ckpt
```

Note: you can download [comic-babes](https://civitai.com/models/20294/comic-babes), [disney-pixar-cartoon-type-a](https://civitai.com/models/65203/disney-pixar-cartoon-type-a), [toonyou](https://civitai.com/models/30240/toonyou) for different styles.

## :camera_flash: Inference

```bash
python inference.py --ckpt ckpts/caphuman.ckpt --vae_ckpt ckpts/vae-ft-mse-840000-ema-pruned.ckpt --model models/cldm_v15.yaml --sd_ckpt ckpts/Realistic_Vision_V3.0.ckpt --input_image examples/input_images/196251.png --pose_image examples/pose_images/pose1.png --prompt "a photo of a man wearing a suit in front of Space Needle"
```

Note: you can replace the sd backbone for different styles, e.g. `--sd_ckpt disneyPixarCartoon_v10.safetensors`.

If you prefer gradio, you can try the following command:

```bash
python -m gradios.gradio_visualization --ckpt ckpts/caphuman.ckpt --vae_ckpt ckpts/vae-ft-mse-840000-ema-pruned.ckpt --model models/cldm_v15.yaml --sd_ckpt ckpts/Realistic_Vision_V3.0.ckpt
```

If you are familiar with [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), please refer to the extension [sd-webui-controlnet](https://github.com/VamosC/sd-webui-controlnet). Note: we make some modifications to support CapHuman.

## :paperclip: Citation
```
@inproceedings{liang2024caphuman,
  author={Liang, Chao and Ma, Fan and Zhu, Linchao and Deng, Yingying and Yang, Yi},
  title={CapHuman: Capture Your Moments in Parallel Universes}, 
  booktitle={CVPR},
  year={2024}
}
```

## :warning: License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## :pray: Acknowledgements

- [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- [adobe-research/diffusion-rig](https://github.com/adobe-research/diffusion-rig)
- [yfeng95/DECA](https://github.com/yfeng95/DECA)
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [openai/CLIP](https://github.com/openai/CLIP)
- [VamosC/CLIP4STR](https://github.com/VamosC/CLIP4STR)
- [mzhaoshuai/RLCF](https://github.com/mzhaoshuai/RLCF)
- [mzhaoshuai/CenterCLIP](https://github.com/mzhaoshuai/CenterCLIP)
- [FreeformRobotics/Divide-and-Co-training](https://github.com/FreeformRobotics/Divide-and-Co-training)
- [Realistic_Vision_V3.0.ckpt](https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE/resolve/main/Realistic_Vision_V3.0.ckpt)
- [comic-babes](https://civitai.com/models/20294/comic-babes)
- [disney-pixar-cartoon-type-a](https://civitai.com/models/65203/disney-pixar-cartoon-type-a)
- [toonyou](https://civitai.com/models/30240/toonyou)
