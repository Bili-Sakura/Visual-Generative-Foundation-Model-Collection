# Visual-Generative-Foundation-Model-Collection

[![🤗 Collection](https://img.shields.io/badge/Collection-Visual%20Generation%20Models-yellow?logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models)

> [!IMPORTANT]
> We only hold the core publicly available ones pre-trained on ImageNet.

A selected collection of CORE visual generative foundation model including code, paper, checkpoint etc. 

## TODO: @src/diffusers Models

> JiT is in progress integrating into diffusers main branch, we will refactor it later on.

- [ ] JiT
- [x] NiT
- [x] PixNerd
- [x] PixelFlow
- [x] SiT
- [x] ADM
- [ ] DDT
- [ ] DeCo
- [ ] DiT
- [ ] EDM2
- [ ] FD-Loss
- [ ] FiT
- [ ] FiTv2
- [ ] LightningDiT
- [ ] MDT
- [ ] MDTv2
- [ ] PAE
- [ ] RAE
- [ ] RAEv2
- [ ] REPA-E

_Update the checklist as new models are added or completed._

## Benchmarks

### ImageNet-256

| Model | #Param | GFLOPs | NFE | FID | IS | Precision | Recall | Code | Paper | Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Pixel modeling** |  |  |  |  |  |  |  |  |  |  |
| ADM-G |  |  | 250 | 4.59 |  | 0.82 | 0.52 | [![Official Code](https://img.shields.io/badge/Official%20Code-ADM-181717?logo=github)](https://github.com/openai/guided-diffusion) | [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-ADM--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/ADM-diffusers) |
| PixelFlow | 677M |  |  | 1.98 | 282.1 | 0.81 | 0.60 | [![Official Code](https://img.shields.io/badge/Official%20Code-PixelFlow-181717?logo=github)](https://github.com/ShoufaChen/PixelFlow) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202025-B31B1B)](http://arxiv.org/abs/2504.07963) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-PixelFlow--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/PixelFlow-diffusers) |
| JiT-H/16 | 953M | 182 | 50 | 1.86 | 303.4 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-JiT-181717?logo=github)](https://github.com/LTH14/JiT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202025-B31B1B)](http://arxiv.org/abs/2511.13720) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-JiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/JiT-diffusers) |
| PixNerd-XL/16 | 700M | 134 | 100 | 1.93 | 297 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-PixNerd-181717?logo=github)](https://github.com/MCG-NJU/PixNerd) | [![Paper](https://img.shields.io/badge/Paper-ICLR%202026-B31B1B)](https://openreview.net/forum?id=BDnOrExHmt) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-PixNerd--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/PixNerd-diffusers) |
| DeCo-XL/16 | 682M |  | 500 | 1.62 | 301 | 0.80 | 0.62 | [![Official Code](https://img.shields.io/badge/Official%20Code-DeCo-181717?logo=github)](https://github.com/Zehong-Ma/DeCo) | [![Paper](https://img.shields.io/badge/Paper-CVPR%202026-B31B1B)](http://arxiv.org/abs/2511.19365) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-DeCo-yellow&logo=huggingface)](https://huggingface.co/zehongma/DeCo) |
| **Latent modeling** |  |  |  |  |  |  |  |  |  |  |
| DiT-XL/2 | 675M | 119 | 250 | 2.27 | 278.24 | 0.83 | 0.57 | [![Official Code](https://img.shields.io/badge/Official%20Code-DiT-181717?logo=github)](https://github.com/facebookresearch/DiT) | [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-B31B1B)](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-DiT--XL--2--256-yellow&logo=huggingface)](https://huggingface.co/facebook/DiT-XL-2-256) |
| MDT-XL/2 | 676M | 119 | 250 | 1.79 | 283.01 | 0.81 | 0.61 | [![Official Code](https://img.shields.io/badge/Official%20Code-MDT-181717?logo=github)](https://github.com/sail-sg/MDT/tree/mdtv1) | [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-B31B1B)](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-MDT--XL2-yellow&logo=huggingface)](https://huggingface.co/shgao/MDT-XL2) |
| MDTv2-XL/2 | 676M | 119 | 250 | 1.58 | 314.73 | 0.79 | 0.65 | [![Official Code](https://img.shields.io/badge/Official%20Code-MDT-181717?logo=github)](https://github.com/sail-sg/MDT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202023-B31B1B)](http://arxiv.org/abs/2303.14389) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-MDT--XL2-yellow&logo=huggingface)](https://huggingface.co/shgao/MDT-XL2) |
| FiT-XL/2 | 824M | 153 | 250 | 4.21 | 254.87 | 0.84 | 0.51 | [![Official Code](https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github)](https://github.com/whlzy/FiT) | [![Paper](https://img.shields.io/badge/Paper-ICML%202024-B31B1B)](https://openreview.net/forum?id=jZVen2JguY) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| SiT-XL/2 | 675M | 119 | 250 | 2.06 | 277.50 | 0.83 | 0.59 | [![Official Code](https://img.shields.io/badge/Official%20Code-SiT-181717?logo=github)](https://github.com/willisma/SiT) | [![Paper](https://img.shields.io/badge/Paper-ECCV%202024-B31B1B)](https://link.springer.com/10.1007/978-3-031-72980-5_2) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-SiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/SiT-diffusers) |
| FiTv2-XL/2 | 671M | 147 | 250 | 2.26 | 260.95 | 0.81 | 0.59 | [![Official Code](https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github)](https://github.com/whlzy/FiT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202024-B31B1B)](http://arxiv.org/abs/2410.13925) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| LightningDiT-XL/2 | 724M | 119 |  | 1.35 | 295.3 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-LightningDiT-181717?logo=github)](https://github.com/hustvl/LightningDiT) | [![Paper](https://img.shields.io/badge/Paper-CVPR%202025-B31B1B)](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_Reconstruction_vs._Generation_Taming_Optimization_Dilemma_in_Latent_Diffusion_Models_CVPR_2025_paper.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| SiT-XL/2 + REG | 677M | 119 | 250 | 1.36 | 299.4 | 0.77 | 0.66 | [![Official Code](https://img.shields.io/badge/Official%20Code-REG-181717?logo=github)](https://github.com/Martinser/REG) | [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B)](https://openreview.net/forum?id=koEALFNBj1) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| DDT-XL/2 | 724M | 119 |  | 1.26 | 310.6 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-DDT-181717?logo=github)](https://github.com/MCG-NJU/DDT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202025-B31B1B)](https://arxiv.org/abs/2504.05741) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| NiT-XL | 675M | 119 | 250 | 2.03 | 265.26 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-NiT-181717?logo=github)](https://github.com/WZDTHU/NiT) | [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B)](https://openreview.net/forum?id=QWQB1ReBkJ) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-NiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/NiT-diffusers) |
| RAE, DiT-DH-XL/2 | 1254M | 146 |  | 1.13 | 262.6 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-RAE-181717?logo=github)](https://github.com/bytetriper/RAE) | [![Paper](https://img.shields.io/badge/Paper-ICLR%202026-B31B1B)](https://openreview.net/forum?id=0u1LigJaab) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |

### ImageNet-512

| Model | #Param | GFLOPs | NFE | FID | IS | Precision | Recall | Code | Paper | Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Pixel modeling** |  |  |  |  |  |  |  |  |  |  |
| ADM-G |  |  | 250 | 7.72 |  | 0.87 | 0.42 | [![Official Code](https://img.shields.io/badge/Official%20Code-ADM-181717?logo=github)](https://github.com/openai/guided-diffusion) | [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-ADM--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/ADM-diffusers) |
| JiT-H/32 | 956M | 183 | 50 | 1.94 | 309.1 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-JiT-181717?logo=github)](https://github.com/LTH14/JiT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202025-B31B1B)](http://arxiv.org/abs/2511.13720) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-JiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/JiT-diffusers) |
| PixNerd-XL/16 | 700M | 583 | 100 | 2.84 | 245.6 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-PixNerd-181717?logo=github)](https://github.com/MCG-NJU/PixNerd) | [![Paper](https://img.shields.io/badge/Paper-ICLR%202026-B31B1B)](https://openreview.net/forum?id=BDnOrExHmt) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-PixNerd--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/PixNerd-diffusers) |
| DeCo-XL/16 | 682M |  | 200 | 2.22 | 290.0 | 0.80 | 0.60 | [![Official Code](https://img.shields.io/badge/Official%20Code-DeCo-181717?logo=github)](https://github.com/Zehong-Ma/DeCo) | [![Paper](https://img.shields.io/badge/Paper-CVPR%202026-B31B1B)](http://arxiv.org/abs/2511.19365) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-DeCo-yellow&logo=huggingface)](https://huggingface.co/zehongma/DeCo) |
| **Latent modeling** |  |  |  |  |  |  |  |  |  |  |
| DiT-XL/2 | 675M | 525 | 250 | 3.04 | 240.82 | 0.84 | 0.54 | [![Official Code](https://img.shields.io/badge/Official%20Code-DiT-181717?logo=github)](https://github.com/facebookresearch/DiT) | [![Paper](https://img.shields.io/badge/Paper-ICCV%202023-B31B1B)](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-DiT--XL--2--512-yellow&logo=huggingface)](https://huggingface.co/facebook/DiT-XL-2-512) |
| EDM2-XXL | 1523M | 552 | 63 | 1.81 |  |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-edm2-181717?logo=github)](https://github.com/NVlabs/edm2) | [![Paper](https://img.shields.io/badge/Paper-CVPR%202024-B31B1B)](http://openaccess.thecvf.com/content/CVPR2024/html/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.html) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| SiT-XL/2 | 675M | 525 | 250 | 2.62 | 252.21 | 0.84 | 0.57 | [![Official Code](https://img.shields.io/badge/Official%20Code-SiT-181717?logo=github)](https://github.com/willisma/SiT) | [![Paper](https://img.shields.io/badge/Paper-ECCV%202024-B31B1B)](https://link.springer.com/10.1007/978-3-031-72980-5_2) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-SiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/SiT-diffusers) |
| FiTv2-XL/2 | 671M | 525 | 250 | 2.90 | 263.11 | 0.83 | 0.53 | [![Official Code](https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github)](https://github.com/whlzy/FiT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202024-B31B1B)](http://arxiv.org/abs/2410.13925) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| DDT-XL/2 | 724M | 525 |  | 1.28 | 305.1 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-DDT-181717?logo=github)](https://github.com/MCG-NJU/DDT) | [![Paper](https://img.shields.io/badge/Paper-arXiv%202025-B31B1B)](https://arxiv.org/abs/2504.05741) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
| NiT-XL | 675M | 525 | 250 | 1.45 | 272.77 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-NiT-181717?logo=github)](https://github.com/WZDTHU/NiT) | [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B)](https://openreview.net/forum?id=QWQB1ReBkJ) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-NiT--diffusers-yellow&logo=huggingface)](https://huggingface.co/BiliSakura/NiT-diffusers) |
| RAE, DiT-DH-XL/2 | 1254M | 642 |  | 1.13 | 259.6 |  |  | [![Official Code](https://img.shields.io/badge/Official%20Code-RAE-181717?logo=github)](https://github.com/bytetriper/RAE) | [![Paper](https://img.shields.io/badge/Paper-ICLR%202026-B31B1B)](https://openreview.net/forum?id=0u1LigJaab) | [![🤗 Model](https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models) |
