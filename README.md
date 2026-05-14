<div align="center">
  <img src="assets/logo.png" alt="EverAnimate logo" width="540">

  <h1>EverAnimate: Minute-Scale Human Animation via Latent Flow Restoration</h1>

  <p>
    <a href="https://wymancv.github.io/wuyang.github.io/"><strong>Wuyang Li</strong></a> &nbsp;
    <a href="https://scholar.google.com/citations?user=rpT0Q6AAAAAJ&hl=en"><strong>Yang Gao</strong></a> &nbsp;
    <a href="https://people.epfl.ch/mariam.hassan?lang=en"><strong>Mariam Hassan</strong></a> &nbsp;
    <a href="https://alan-lanfeng.github.io/"><strong>Lan Feng</strong></a><br>
    <a href="https://scholar.google.com/citations?user=sHKkAToAAAAJ&hl=zh-CN"><strong>Wentao Pan</strong></a> &nbsp;
    <a href="https://scholar.google.com/citations?user=Y2Oth4MAAAAJ&hl=zh-TW"><strong>Po-Chien Luan</strong></a> &nbsp;
    <a href="https://people.epfl.ch/alexandre.alahi/?lang=en"><strong>Alexandre Alahi</strong></a><br>
    <a href="https://www.epfl.ch/labs/vita/"><em>VITA@EPFL</em></a>
  </p>

  <p>
    <a href="https://everanimate.github.io/homepage/">
      <img alt="Project page" src="https://img.shields.io/badge/Project-Page-0f172a?style=for-the-badge">
    </a>
    <a href="assets/paper.pdf">
      <img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-b91c1c?style=for-the-badge">
    </a>
    <img alt="Code status" src="https://img.shields.io/badge/Code-Coming%20Soon-64748b?style=for-the-badge">
    <img alt="Model status" src="https://img.shields.io/badge/Models-Coming%20Soon-64748b?style=for-the-badge">
    <img alt="Data status" src="https://img.shields.io/badge/Data-Coming%20Soon-64748b?style=for-the-badge">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-16a34a?style=for-the-badge">
  </p>
</div>

---

<div align="center">
  <img src="assets/teaser.png" alt="EverAnimate teaser" width="100%">
</div>

## Coming Soon

This repository is being prepared for public release. Code, checkpoints, training data, and evaluation tools will be released once the project materials are ready.

| Resource | Status |
| --- | --- |
| Project page | Available |
| Paper PDF | Available |
| Inference code | Coming soon |
| Training code | Coming soon |
| Model checkpoints | Coming soon |
| Training data | Coming soon |
| Evaluation scripts | Coming soon |

Contact: <a href="https://wymancv.github.io/wuyang.github.io/"><strong>Wuyang Li</strong></a>   
Email: wuyang.li@epfl.ch

## Planned Release

- Environment setup and installation guide
- Single-reference and multi-reference inference scripts
- Long-horizon animation generation pipeline
- Model checkpoints and LoRA weights
- Training code and data preparation instructions
- Evaluation protocol for short- and long-horizon settings

## Citation

The final BibTeX entry will be updated once the paper is public.

```bibtex
@misc{li2026everanimate,
  title  = {EverAnimate: Minute-Scale Human Animation via Latent Flow Restoration},
  author = {Wuyang Li and Yang Gao and Mariam Hassan and Lan Feng and Wentao Pan and Po-Chien Luan and Alexandre Alahi},
  year   = {2026},
  note   = {Coming soon}
}
```

## Abstract

**EverAnimate** is an efficient post-training method for long-horizon animated video generation that preserves visual quality and character identity. Long-form animation remains challenging because highly dynamic human motion must be synthesized against relatively static environments, making chunk-based generation prone to accumulated drift: low-level quality drift, such as progressive degradation of static backgrounds, and high-level semantic drift, such as inconsistent character identity and view-dependent attributes.

EverAnimate restores drifted flow trajectories by anchoring generation to a persistent latent context memory. It consists of two complementary mechanisms: **Persistent Latent Propagation**, which maintains context memory across chunks to propagate identity and motion in latent space while mitigating temporal forgetting, and **Restorative Flow Matching**, which introduces an implicit restoration objective during sampling through velocity adjustment to improve within-chunk fidelity.

With only lightweight LoRA tuning, EverAnimate outperforms state-of-the-art long-animation methods in both short- and long-horizon settings: at 10 seconds, it improves PSNR/SSIM by 8%/7% and reduces LPIPS/FID by 22%/11%; at 90 seconds, the gains increase to 15%/15% and 32%/27%, respectively.

## Method Overview

| Component | Purpose |
| --- | --- |
| Persistent Latent Propagation | Propagates identity and motion through latent memory across chunks. |
| Restorative Flow Matching | Corrects drifted latent trajectories with a bounded restorative velocity target. |
| Lightweight LoRA Adaptation | Enables efficient post-training on top of a video animation backbone. |

## Acknowledgements

This work is developed based on the following projects:

- [Wan-Animate: Unified Character Animation and Replacement with Holistic Replication](https://arxiv.org/abs/2509.14055)
- [Stable Video Infinity: Infinite-Length Video Generation with Error Recycling](https://stable-video-infinity.github.io/homepage/)

