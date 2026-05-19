# Visual-Generative-Foundation-Model-Collection


> [!IMPORTANT] We only hold the core publicly available ones pre-trained on ImageNet.

A selected collection of CORE visual generative foundation model including code, paper, checkpoint etc. 

## Benchmarks

### ImageNet-256

<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Model</th>
      <th align="center" colspan="3" style="border-bottom:1px solid #ccc;">Compute</th>
      <th align="center" colspan="4" style="border-bottom:1px solid #ccc;">Quality</th>
      <th align="left" rowspan="2">Links</th>
    </tr>
    <tr>
      <th align="right">#Param</th>
      <th align="right">GFLOPs</th>
      <th align="right">NFE</th>
      <th align="right">FID</th>
      <th align="right">IS</th>
      <th align="right">Precision</th>
      <th align="right">Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" align="left" style="background-color:#dbeafe;font-weight:600;border-top:2px solid #3b82f6;">Pixel modeling</td>
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
        <a href="https://proceedings.mlr.press/v139/nichol21a.html"><img src="https://img.shields.io/badge/Paper-ICML%202021-B31B1B" alt="Paper"/></a>
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
      <td align="left">
        <a href="https://proceedings.mlr.press/v202/jabri23a.html"><img src="https://img.shields.io/badge/Paper-ICML%202023-B31B1B" alt="Paper"/></a>
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
        <a href="https://github.com/Bili-Sakura/ADM-diffusers"><img src="https://img.shields.io/badge/Diffusers-ADM--diffusers-181717?logo=github" alt="Diffusers Code"/></a>
        <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html"><img src="https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B" alt="Paper"/></a>
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td colspan="9" align="left" style="background-color:#ede9fe;font-weight:600;border-top:2px solid #8b5cf6;">Latent modeling</td>
    </tr>
    <tr>
      <td align="left"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left"></td>
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
      <th align="left" rowspan="2">Links</th>
    </tr>
    <tr>
      <th align="right">#Param</th>
      <th align="right">GFLOPs</th>
      <th align="right">NFE</th>
      <th align="right">FID</th>
      <th align="right">IS</th>
      <th align="right">Precision</th>
      <th align="right">Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" align="left" style="background-color:#dbeafe;font-weight:600;border-top:2px solid #3b82f6;">Pixel modeling</td>
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
      <td align="left">
        <a href="https://proceedings.mlr.press/v202/jabri23a.html"><img src="https://img.shields.io/badge/Paper-ICML%202023-B31B1B" alt="Paper"/></a>
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
        <a href="https://github.com/Bili-Sakura/ADM-diffusers"><img src="https://img.shields.io/badge/Diffusers-ADM--diffusers-181717?logo=github" alt="Diffusers Code"/></a>
        <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html"><img src="https://img.shields.io/badge/Paper-NeurIPS%202021-B31B1B" alt="Paper"/></a>
        <a href="https://huggingface.co/collections/BiliSakura/visual-generation-models"><img src="https://img.shields.io/badge/🤗%20Model-TBD-yellow&logo=huggingface" alt="Hugging Face Model (TBD)"/></a>
      </td>
    </tr>
    <tr>
      <td colspan="9" align="left" style="background-color:#ede9fe;font-weight:600;border-top:2px solid #8b5cf6;">Latent modeling</td>
    </tr>
    <tr>
      <td align="left"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="right"></td>
      <td align="left"></td>
    </tr>
  </tbody>
</table>
