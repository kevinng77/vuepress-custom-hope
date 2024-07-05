---
title: DIFFUSION 系列笔记 | IP-Adapter
date: 2023-12-20
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- CV
- Diffusion
- AIGC
---

## Fooocus

https://github.com/lllyasviel/Fooocus

```
python entry_with_update.py --language zh --port 8000
```

主要启动函数在 `webui/generate_clicked(*args)`

主要输入为：

```python
ctrls = [
    prompt, negative_prompt, style_selections,
    performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, guidance_scale
] //

ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
ctrls += [input_image_checkbox, current_tab]
ctrls += [uov_method, uov_input_image]
ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt]
ctrls += ip_ctrls
```

其中 

+ `prompt`：初始 prompt，并非模型推理时候的 prompt。
+ `negative_prompt`：初始 negtive prompt。
+ `style_selections`：style 以 prompt 的形式填充。
+ `performance_selection`：fooocus 中提供了 `speed`, `quality`, `extreme speed` 三种方案分别采用 step 30, step 60 和 lcm 模式下（用的 lcm lora）的 step 8 进行推理。
+ `aspect_ratios_selection`：图片大小
+ `image_number`：fooocus 通过 for 循环生成图片，并非采用增大 batch size 实现。
+ `image_seed`：关闭 random 之后就可以设置。
+ `sharpness`：类似与锐化程度
+ `guidance_scale`：有些地方叫做 `cfg scale`，可以根据模型训练时候的情况进行配置。

- `refiner_switch`：SDXL 中，从 base 切换到 `refine` 模型的时间点。通常取 0.8。
- `lora_ctrls`：要添加的 lora 模型以及对应的 weight。
- `input_image_checkbox` 和`current_tab`：fooocus 中提供了 `Upscale or Variantion`, `Image Prompt`, `Inpaint or Outpaint`, `Describe` 四种模式。
- `uov_method`：`Upscale or Variantion` 提供了 `Vary(Subtle)`, `Upscale(1.5x)` 等其他方案。
- `outpaint_selections`：支持 `['Left', 'Right', 'Top', 'Bottom']` 中的一项
- `ip_ctrls`: image prompt。支持输入几张图片，然后选择不同的图片参考模式。

除了主要输出外，在 `modules.advanced_parameters` 中还有一些其他参数可以用来进一步配置推理，比如 `sampler`, `scheduler` 等

### 推理分析

推理部分在 `modules.async_worker.py`

#### 模型加载

fooocus 的模型加载有不同策略，个人喜欢采用 

```bash
python entry_with_update.py --language zh --port 8000 --always-gpu --disable-offload-from-vram
```

默认情况下，fooocus 会根据使用者的显卡，来选择不同的 VRAM offload 策略，感觉频繁的 offload 对显卡似乎不太好。

#### prompt 处理

1. fooocus 会对文本进行简单的清洗：比如去除对于的空格或换行符等。

2.  **wild card：** 类似 auto 1111 的 wild card 插件，在 prompt 中如果出现 `__flower__` ，那么就会从 `wildcard/flower.txt` 中随机挑选 prompt 插入。

3.  **添加 style：** 根据选择的 `style`，在 prompt 中添加对应的内容。（参考 `sdxl_styles` 文件夹）。

4.  **fooocus_expansion** ：style 中使用 `Fooocus V2` 时，会对输入的 prompt 进行额外的填充，用微调后的 GPT 2 来对 prompt 进行扩展填充，此处使用的为 GPT 2 的架构，但词表改动较大。 (权重在该[连接](https://huggingface.co/lllyasviel/misc/tree/main)下`fooocus_expansion.bin` 文件)

5.  **encoding：** 采用 clip 模型（`ldm_patched/modules/sd/CLIP`）进行 encoding，对每个 style 填充后的 prompt 和 neg prompt 分别进行编码。大概流程为：

   - `tokenize_with_weights` ：如果 prompt 当中有如 `(happy:1.5)` 类型的 prompt，那么这部分 prompt 回和 auto1111 进行类似的权重处理。源码在[这](https://github.com/lllyasviel/Fooocus/blob/main/ldm_patched/modules/sd1_clip.py#L398)。比如输入为 `"This is a sample string with (parentheses) and (some other:1.5) special characters."` 的话，那么输入会先被处理成 `[('This is a sample string with ', 1.0), ('parentheses', 1.1), (' and ', 1.0), ('some other', 1.5), (' special characters.', 1.0)]` 而后进行 token，转化为 `[(token, weight),] ` 的形式

   - `encode_token_weights`：参考官方 `ClipTokenWeightEncoder` 的实现。假设 prompt 当中有存在 weight 的 token 那么对于这些有权重的 token，他们的 hidden state 会被进行调整，调整方式为：

     ```python
     z_empty = encode(self.special_tokens)
     z_token = (z_token-z_empty)*weight + z_empty
     ```

     总结来看，weight 是根据特殊 token 进行调整的。

6. 经过所有处理后，prompt 会被转化为

```python
[[concat_hidden_state, {"pooled_output": pool_hidden_state}]]
```

的形式，其中 `concat_hidden_state` 为 torch tensor，shape 大致为 `[1, 77* num_style, 2048]` （具体 shape 根据模型而定，但 hidden state 均根据 `len_input` 这个维度拼接）

::: tip

在推理过程中，可以缓存好出现频率高的 prompt 及其对应 hidden state。 对于部分 SD 落地应用，CLIP 的部分甚至可以变成离线任务，以此节省计算资源。

:::

#### 图片 prompt

fooocus 支持使用 Ip-Adapter，controlnet canny， depth 等垫图操作。

image prompt 中主要采用 [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) 涉及到的模型有：

+ [clip_vision_vit_h.safetensors](https://huggingface.co/lllyasviel/misc/tree/main)：
+ [fooocus_ip_negative.safetensors](https://huggingface.co/lllyasviel/misc/tree/main)：Fooocus 团队训练的 pre-computed negative embedding
+ [ip-adapter-plus-face_sdxl_vit-h.bin](https://huggingface.co/lllyasviel/misc/tree/main) 或 [ip-adapter-plus_sdxl_vit-h.bin](https://huggingface.co/lllyasviel/misc/tree/main)：IP-Adapter 的 sdxl 模型权重。Fooocus 使用的权重比 IP-Adapter 官方的要大一点。不确定是根据官方的权重进行微调还是有其他操作。

+ [control-lora-canny-rank128.safetensors](https://huggingface.co/lllyasviel/misc/tree/main)：canny 风格图片借鉴。
+ [fooocus_xl_cpds_128.safetensors](https://huggingface.co/lllyasviel/misc/tree/main)：fooocus 团队根据  [“Contrast Preserving Decolorization (CPD)”](http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html) 算法修改得到的模型，可以实现和 controlnet 中 depth 类似的功能。



#### 扩散过程

sampler 的入口函数可以参考 fooocus 的 `ldm_patched.modules.samplers` 及 `ldm_patched.modules.sample`。

各个 sampler 的实现可以参考`k_diffusion.sampling` 。



### 额外参数

- `disable_preview`：可以关闭 UI 上的生成预览
- `sampler_name`：默认用的 `dpmpp_2m_sde_gpu`，可自定义采样方案。
- `scheduler_name`：默认用的 `karas`，可自定义扩散过程的迭代算法。
- `generate_image_grid`：在 UI 上，可以选择不以网格的形式展示生成的图片
- `overwrite_step`：自定义模型推理的总 step。
- `overwrite_switch`：自定义 base 切换到 refine 的时间点。
- `overwrite_width` 和 `overwrite_height`：自定义图片输出大小
- `overwrite_vary_strength` 和 `overwrite_upscale_strength`：进行图片 uov 模式时，自定义的 vary 或者 upscale 大小。
- `refiner_swap_method`：支持 `joint`, `separate` 和 `vae`。官方的描述是， Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup，这样确保了扩散模型采样的一致性。
- `freeu_enabled`
- `freeu_b1`
- `freeu_b2`
- `freeu_s1`
- `freeu_s2`
- `debugging_inpaint_preprocessor`
- `inpaint_disable_initial_latent`
- `inpaint_engine`
- `inpaint_strength`：同 denoising_strength。越小，生成的图片越接近参考图。
- `inpaint_respective_field`

- `mixing_image_prompt_and_vary_upscale`
- `mixing_image_prompt_and_inpaint`
- `controlnet_softness`
- `canny_low_threshold`
- `canny_high_threshold`

- `adm_scaler_positive`
- `adm_scaler_negative`
- `adm_scaler_end`
- `adaptive_cfg`