# Visual-Generative-Foundation-Model-Collection

[![🤗 Collection](https://img.shields.io/badge/🤗%20Collection-Visual%20Generation%20Models-yellow?logo=huggingface)](https://huggingface.co/collections/BiliSakura/visual-generation-models)

> [!IMPORTANT]
> We only hold the core publicly available ones pre-trained on ImageNet.

A selected collection of CORE visual generative foundation model including code, paper, checkpoint etc. 

## Benchmarks

### ImageNet-256

<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Model</th>
      <th align="center" colspan="3" style="border-bottom:1px solid #ccc;">Compute</th>
      <th align="center" colspan="4" style="border-bottom:1px solid #ccc;">Quality</th>
      <th align="center" colspan="3" style="border-bottom:1px solid #ccc;">Links</th>
    </tr>
    <tr>
      <th align="right">#Param</th>
      <th align="right">GFLOPs</th>
      <th align="right">NFE</th>
      <th align="right">FID</th>
      <th align="right">IS</th>
      <th align="right">Precision</th>
      <th align="right">Recall</th>
      <th align="left">Code</th>
      <th align="left">Paper</th>
      <th align="left">Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="11" align="left" style="background-color:#dbeafe;font-weight:600;border-top:2px solid #3b82f6;">Pixel modeling</td>
    </tr>
    <!--
    <tr>
      <td align="left">IDDPM</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right">250</td>
      <td align="right">12.26</td>
      <td align="right"></td>
      <td align="right">0.70</td>
      <td align="right">0.62</td>
      <td align="left">
        <a href="https://github.com/openai/improved-diffusion"><img src="https://img.shields.io/badge/Official%20Code-improved--diffusion-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://proceedings.mlr.press/v139/nichol21a.html"><img src="https://img.shields.io/badge/Paper-ICML%202021-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">RIN</td>
      <td align="right">410M</td>
      <td align="right">334</td>
      <td align="right"></td>
      <td align="right">4.51</td>
      <td align="right">161.0</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left"></td>
      <td align="left">
        <a href="https://proceedings.mlr.press/v202/jabri23a.html"><img src="https://img.shields.io/badge/Paper-ICML%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    -->
    <tr>
      <td align="left">ADM-G</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right">250</td>
      <td align="right">4.59</td>
      <td align="right"></td>
      <td align="right">0.82</td>
      <td align="right">0.52</td>
      <td align="left">
        <a href="https://github.com/openai/guided-diffusion"><img src="https://img.shields.io/badge/Official%20Code-guided--diffusion-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html"><img src="https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">PixelFlow-G</td>
      <td align="right">677M</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right">1.98</td>
      <td align="right">282.1</td>
      <td align="right">0.81</td>
      <td align="right">0.60</td>
      <td align="left">
        <a href="https://github.com/ShoufaChen/PixelFlow"><img src="https://img.shields.io/badge/Official%20Code-PixelFlow-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2504.07963"><img src="https://img.shields.io/badge/Paper-arXiv%202025-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/ShoufaChen/PixelFlow-Class2Image"><img src="https://img.shields.io/badge/🤗%20Model-PixelFlow--Class2Image-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">JiT-H/16-G</td>
      <td align="right">953M</td>
      <td align="right">182</td>
      <td align="right">50</td>
      <td align="right">1.86</td>
      <td align="right">303.4</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/LTH14/JiT"><img src="https://img.shields.io/badge/Official%20Code-JiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2511.13720"><img src="https://img.shields.io/badge/Paper-arXiv%202025-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/JiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-JiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">PixNerd-XL/16-G</td>
      <td align="right">700M</td>
      <td align="right">134</td>
      <td align="right">100</td>
      <td align="right">1.93</td>
      <td align="right">297</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/MCG-NJU/PixNerd"><img src="https://img.shields.io/badge/Official%20Code-PixNerd-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openreview.net/forum?id=BDnOrExHmt"><img src="https://img.shields.io/badge/Paper-ICLR%202026-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/PixNerd-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-PixNerd--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">DeCo-XL/16-G</td>
      <td align="right">682M</td>
      <td align="right"></td>
      <td align="right">500</td>
      <td align="right">1.62</td>
      <td align="right">301</td>
      <td align="right">0.80</td>
      <td align="right">0.62</td>
      <td align="left">
        <a href="https://github.com/Zehong-Ma/DeCo"><img src="https://img.shields.io/badge/Official%20Code-DeCo-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2511.19365"><img src="https://img.shields.io/badge/Paper-CVPR%202026-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/zehongma/DeCo"><img src="https://img.shields.io/badge/🤗%20Model-DeCo-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td colspan="11" align="left" style="background-color:#ede9fe;font-weight:600;border-top:2px solid #8b5cf6;">Latent modeling</td>
    </tr>
    <tr>
      <td align="left">DiT-XL/2-G</td>
      <td align="right">675M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">2.27</td>
      <td align="right">278.24</td>
      <td align="right">0.83</td>
      <td align="right">0.57</td>
      <td align="left">
        <a href="https://github.com/facebookresearch/DiT"><img src="https://img.shields.io/badge/Official%20Code-DiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/Paper-ICCV%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/facebook/DiT-XL-2-256"><img src="https://img.shields.io/badge/🤗%20Model-DiT--XL--2--256-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">MDT-XL/2-G</td>
      <td align="right">676M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">1.79</td>
      <td align="right">283.01</td>
      <td align="right">0.81</td>
      <td align="right">0.61</td>
      <td align="left">
        <a href="https://github.com/sail-sg/MDT/tree/mdtv1"><img src="https://img.shields.io/badge/Official%20Code-MDT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2303.14389"><img src="https://img.shields.io/badge/Paper-arXiv%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/shgao/MDT-XL2"><img src="https://img.shields.io/badge/🤗%20Model-MDT--XL2-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">MDTv2-XL/2-G</td>
      <td align="right">676M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">1.58</td>
      <td align="right">314.73</td>
      <td align="right">0.79</td>
      <td align="right">0.65</td>
      <td align="left">
        <a href="https://github.com/sail-sg/MDT"><img src="https://img.shields.io/badge/Official%20Code-MDT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2303.14389"><img src="https://img.shields.io/badge/Paper-arXiv%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/shgao/MDT-XL2"><img src="https://img.shields.io/badge/🤗%20Model-MDT--XL2-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">FiT-XL/2-G</td>
      <td align="right">824M</td>
      <td align="right">153</td>
      <td align="right">250</td>
      <td align="right">4.21</td>
      <td align="right">254.87</td>
      <td align="right">0.84</td>
      <td align="right">0.51</td>
      <td align="left">
        <a href="https://github.com/whlzy/FiT"><img src="https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openreview.net/forum?id=jZVen2JguY"><img src="https://img.shields.io/badge/Paper-ICML%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">SiT-XL/2-G</td>
      <td align="right">675M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">2.06</td>
      <td align="right">277.50</td>
      <td align="right">0.83</td>
      <td align="right">0.59</td>
      <td align="left">
        <a href="https://github.com/willisma/SiT"><img src="https://img.shields.io/badge/Official%20Code-SiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://link.springer.com/10.1007/978-3-031-72980-5_2"><img src="https://img.shields.io/badge/Paper-ECCV%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/SiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-SiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">SiT-XL/2 + REG-G</td>
      <td align="right">677M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">1.36</td>
      <td align="right">299.4</td>
      <td align="right">0.77</td>
      <td align="right">0.66</td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">FiTv2-XL/2-G</td>
      <td align="right">671M</td>
      <td align="right">147</td>
      <td align="right">250</td>
      <td align="right">2.26</td>
      <td align="right">260.95</td>
      <td align="right">0.81</td>
      <td align="right">0.59</td>
      <td align="left">
        <a href="https://github.com/whlzy/FiT"><img src="https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2410.13925"><img src="https://img.shields.io/badge/Paper-arXiv%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">LightningDiT-XL/2-G</td>
      <td align="right">724M</td>
      <td align="right">119</td>
      <td align="right"></td>
      <td align="right">1.35</td>
      <td align="right">295.3</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">DDT-XL/2-G</td>
      <td align="right">724M</td>
      <td align="right">119</td>
      <td align="right"></td>
      <td align="right">1.26</td>
      <td align="right">310.6</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">RAE, DiT-DH-XL/2-G</td>
      <td align="right">1254M</td>
      <td align="right">146</td>
      <td align="right"></td>
      <td align="right">1.13</td>
      <td align="right">262.6</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">NiT-XL-G</td>
      <td align="right">675M</td>
      <td align="right">119</td>
      <td align="right">250</td>
      <td align="right">2.03</td>
      <td align="right">265.26</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/WZDTHU/NiT"><img src="https://img.shields.io/badge/Official%20Code-NiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openreview.net/forum?id=QWQB1ReBkJ"><img src="https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/NiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-NiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
  </tbody>
</table>

### ImageNet-512

<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Model</th>
      <th align="center" colspan="3" style="border-bottom:1px solid #ccc;">Compute</th>
      <th align="center" colspan="4" style="border-bottom:1px solid #ccc;">Quality</th>
      <th align="center" colspan="3" style="border-bottom:1px solid #ccc;">Links</th>
    </tr>
    <tr>
      <th align="right">#Param</th>
      <th align="right">GFLOPs</th>
      <th align="right">NFE</th>
      <th align="right">FID</th>
      <th align="right">IS</th>
      <th align="right">Precision</th>
      <th align="right">Recall</th>
      <th align="left">Code</th>
      <th align="left">Paper</th>
      <th align="left">Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="11" align="left" style="background-color:#dbeafe;font-weight:600;border-top:2px solid #3b82f6;">Pixel modeling</td>
    </tr>
    <!--
    <tr>
      <td align="left">RIN + inp. scale</td>
      <td align="right">320M</td>
      <td align="right">415</td>
      <td align="right"></td>
      <td align="right">3.95</td>
      <td align="right">216.0</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left"></td>
      <td align="left">
        <a href="https://proceedings.mlr.press/v202/jabri23a.html"><img src="https://img.shields.io/badge/Paper-ICML%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    -->
    <tr>
      <td align="left">ADM-G</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right">250</td>
      <td align="right">7.72</td>
      <td align="right"></td>
      <td align="right">0.87</td>
      <td align="right">0.42</td>
      <td align="left">
        <a href="https://github.com/openai/guided-diffusion"><img src="https://img.shields.io/badge/Official%20Code-guided--diffusion-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html"><img src="https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">JiT-H/32-G</td>
      <td align="right">956M</td>
      <td align="right">183</td>
      <td align="right">50</td>
      <td align="right">1.94</td>
      <td align="right">309.1</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/LTH14/JiT"><img src="https://img.shields.io/badge/Official%20Code-JiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2511.13720"><img src="https://img.shields.io/badge/Paper-arXiv%202025-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/JiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-JiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">PixNerd-XL/16-G</td>
      <td align="right">700M</td>
      <td align="right">583</td>
      <td align="right">100</td>
      <td align="right">2.84</td>
      <td align="right">245.6</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/MCG-NJU/PixNerd"><img src="https://img.shields.io/badge/Official%20Code-PixNerd-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openreview.net/forum?id=BDnOrExHmt"><img src="https://img.shields.io/badge/Paper-ICLR%202026-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/PixNerd-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-PixNerd--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">DeCo-XL/16-G</td>
      <td align="right">682M</td>
      <td align="right"></td>
      <td align="right">200</td>
      <td align="right">2.22</td>
      <td align="right">290.0</td>
      <td align="right">0.80</td>
      <td align="right">0.60</td>
      <td align="left">
        <a href="https://github.com/Zehong-Ma/DeCo"><img src="https://img.shields.io/badge/Official%20Code-DeCo-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2511.19365"><img src="https://img.shields.io/badge/Paper-CVPR%202026-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/zehongma/DeCo"><img src="https://img.shields.io/badge/🤗%20Model-DeCo-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td colspan="11" align="left" style="background-color:#ede9fe;font-weight:600;border-top:2px solid #8b5cf6;">Latent modeling</td>
    </tr>
    <tr>
      <td align="left">DiT-XL/2-G</td>
      <td align="right">675M</td>
      <td align="right">525</td>
      <td align="right">250</td>
      <td align="right">3.04</td>
      <td align="right">240.82</td>
      <td align="right">0.84</td>
      <td align="right">0.54</td>
      <td align="left">
        <a href="https://github.com/facebookresearch/DiT"><img src="https://img.shields.io/badge/Official%20Code-DiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/Paper-ICCV%202023-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/facebook/DiT-XL-2-512"><img src="https://img.shields.io/badge/🤗%20Model-DiT--XL--2--512-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">EDM2-XXL-G</td>
      <td align="right">1523M</td>
      <td align="right">552</td>
      <td align="right">63</td>
      <td align="right">1.81</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/NVlabs/edm2"><img src="https://img.shields.io/badge/Official%20Code-edm2-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://openaccess.thecvf.com/content/CVPR2024/html/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">SiT-XL/2-G</td>
      <td align="right">675M</td>
      <td align="right">525</td>
      <td align="right">250</td>
      <td align="right">2.62</td>
      <td align="right">252.21</td>
      <td align="right">0.84</td>
      <td align="right">0.57</td>
      <td align="left">
        <a href="https://github.com/willisma/SiT"><img src="https://img.shields.io/badge/Official%20Code-SiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://link.springer.com/10.1007/978-3-031-72980-5_2"><img src="https://img.shields.io/badge/Paper-ECCV%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/SiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-SiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">FiTv2-XL/2-G</td>
      <td align="right">671M</td>
      <td align="right">525</td>
      <td align="right">250</td>
      <td align="right">2.90</td>
      <td align="right">263.11</td>
      <td align="right">0.83</td>
      <td align="right">0.53</td>
      <td align="left">
        <a href="https://github.com/whlzy/FiT"><img src="https://img.shields.io/badge/Official%20Code-FiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="http://arxiv.org/abs/2410.13925"><img src="https://img.shields.io/badge/Paper-arXiv%202024-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">DDT-XL/2-G</td>
      <td align="right">724M</td>
      <td align="right">525</td>
      <td align="right"></td>
      <td align="right">1.28</td>
      <td align="right">305.1</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">RAE, DiT-DH-XL/2-G</td>
      <td align="right">1254M</td>
      <td align="right">642</td>
      <td align="right"></td>
      <td align="right">1.13</td>
      <td align="right">259.6</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Official%20Code-TBD-181717?logo=github" alt="Official Code (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/Paper-TBD-B31B1B" alt="Paper (TBD)"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td align="left">NiT-XL-G</td>
      <td align="right">675M</td>
      <td align="right">525</td>
      <td align="right">250</td>
      <td align="right">1.45</td>
      <td align="right">272.77</td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left">
        <a href="https://github.com/WZDTHU/NiT"><img src="https://img.shields.io/badge/Official%20Code-NiT-181717?logo=github" alt="Official Code"/></a>
      </td>
      <td align="left">
        <a href="https://openreview.net/forum?id=QWQB1ReBkJ"><img src="https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B" alt="Paper"/></a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/BiliSakura/NiT-diffusers"><img src="https://img.shields.io/badge/🤗%20Model-NiT--diffusers-yellow&logo=huggingface" alt="Hugging Face Model"/></a>
      </td>
    </tr>
  </tbody>
</table>
