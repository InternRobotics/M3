# M³: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM

[![Project Website](https://img.shields.io/badge/M3-Website-4CAF50?logo=googlechrome&logoColor=white)](https://city-super.github.io/M3/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-Weights-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/cskrren/M3) [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2603.16844)

[Kerui Ren](https://scholar.google.com/citations?user=5kW5apkAAAAJ), [Guanghao Li](https://scholar.google.com/citations?user=6nkKQDIAAAAJ), [Changjian Jiang](https://scholar.google.com/citations?user=V4miywEAAAAJ), [Yingxiang Xu](https://github.com/LeoX0808), [Tao Lu](https://scholar.google.com/citations?user=Ch28NiIAAAAJ), [Linning Xu](https://eveneveno.github.io/lnxu/), [Junting Dong](https://jtdong.com/), [Jiangmiao Pang](https://oceanpang.github.io/), [Mulin Yu<sup>†</sup>](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Bo Dai<sup>†</sup>](https://daibo.info/)

## 📢 News

**[2026.10.29]** 🚀 Pi3 and Pi3X matching checkpoints are released! Code release preparation is in progress.

**[2026.10.29]** 🎉 We are happy to announce that M³ has been accepted to **SIGGRAPH Asia 2026**.

## Overview
![M³ pipeline](assets/M3.png)
**Pipeline of M³.** Our framework consists of joint tracking and global optimization for pose estimation and a mapper for scene reconstruction. For monocular sequences, Pi3X processes retrieved historical keyframes and new frames in one inference to facilitate factor graph construction and keyframe selection. Following the Neural Gaussian and LOD architecture of ARTDECO, Gaussians are initialized via Laplacian norm and optimized jointly with camera poses.
