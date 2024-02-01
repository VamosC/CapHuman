[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)

<p align="center">
<img src="assets/logo.png" width="100">
</p>

# CapHuman: Capture Your Moments in Parallel Universes

[[Project Page]](https://caphuman.github.io/)

This is the repository for the paper *CapHuman: Capture Your Moments in Parallel Universes*.

Chao Liang<sup></sup>,&nbsp;[Fan Ma](https://flowerfan.site/),&nbsp;[Linchao Zhu](https://ffmpbgrnn.github.io/)<sup></sup>,&nbsp;[Yingying Deng](https://diyiiyiii.github.io/),&nbsp;[Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en).&nbsp;

<img src='./assets/teaser.png' width=850>

We concentrate on a novel human-centric image synthesis task, that is, given only one reference facial photograph, it is expected to generate specific individual images with diverse head positions, poses, and facial expressions in different contexts. To accomplish this goal, we argue that our generative model should be capable of the following favorable characteristics: (1) a strong visual and semantic understanding of our world and human society for basic object and human image generation. (2) generalizable identity preservation ability. (3) flexible and fine-grained head control. Recently, large pre-trained text-to-image diffusion models have shown remarkable results, serving as a powerful generative foundation. As a basis, we aim to unleash the above two capabilities of the pre-trained model. In this work, we present a new framework named CapHuman. We embrace the ``encode then learn to align" paradigm, which enables generalizable identity preservation for new individuals without cumbersome tuning at inference. CapHuman encodes identity features and then learns to align them into the latent space. Moreover, we introduce the 3D facial prior to equip our model with control over the human head in a flexible and 3D-consistent manner. Extensive qualitative and quantitative analyses demonstrate our CapHuman can produce well-identity-preserved, photo-realistic, and high-fidelity portraits with content-rich representations and various head renditions, superior to established baselines.

## :flags: News
- [2024/02/01] We release the [Project Page](https://caphuman.github.io/).

## :paperclip: Citation
```
@article{,
  author={Liang, Chao and Ma, Fan and Zhu, Linchao and Deng, Yingying and Yang, Yi},
  journal={}, 
  title={CapHuman: Capture Your Moments in Parallel Universes}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}
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

