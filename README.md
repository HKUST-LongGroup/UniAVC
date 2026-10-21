<p align="center">
  <img src="assets/logo.svg?v=2" alt="UniAVC" width="620" />
</p>

<h3 align="center">Unified Audio-Video Creation via Progressive Training and Asymmetric Scheduling</h3>

<div align="center">

<a href="https://arxiv.org/abs/2610.24791"><img src="https://img.shields.io/badge/arXiv-2610.24791-b31b1b.svg" alt="arXiv"></a>
<a href="https://hkust-longgroup.github.io/UniAVC/"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"></a>
<a href="https://huggingface.co/TK3105/UniAVC"><img src="https://img.shields.io/badge/🤗-Model%20Weights-yellow" alt="Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License"></a>
<img src="https://visitor-badge.laobi.icu/badge?page_id=HKUST-LongGroup.UniAVC" alt="Visitors">

**Kien T. Pham<sup>1</sup>, Yuhao Liu<sup>2</sup>, Chong Hou Choi<sup>2</sup>, Zhenqi He<sup>1</sup>, Yanghao Wang<sup>1</sup>, Chenyang Lei<sup>2</sup>, Rui Liu<sup>2</sup>, Qifeng Chen<sup>1</sup>, Long Chen<sup>1</sup>**

<sup>1</sup>The Hong Kong University of Science and Technology &nbsp;&nbsp; <sup>2</sup>Huawei Research

<a href="mailto:tkpham@connect.ust.hk">tkpham@connect.ust.hk</a> &nbsp;&middot;&nbsp; <a href="mailto:longchen@ust.hk">longchen@ust.hk</a>

</div>

---

## 🎬 Demo

<div><video src="https://github.com/user-attachments/assets/fbd6466a-416e-448b-853a-45e9d78bad2c" controls width="100%"></video></div>

---

## Overview

UniAVC is a unified model that natively handles text-conditioned joint audio-video generation, bidirectional cross-modal synthesis, and instruction-guided editing of either or both modalities without relying on task-specific adapters. To achieve this versatility, the model utilizes a progressive training paradigm that first masters unimodal generation and editing capabilities before dedicating scarce paired datasets entirely to establishing cross-modal correspondence and temporal synchronization. The framework further introduces an asymmetric noise-level scheduling technique that derives decoupled per-branch noise trajectories and explicitly supervises extreme boundary states to enable intent-aware editing and post-inference self-refinement. Extensive empirical evaluations demonstrate that this unified approach delivers performance comparable to specialized, state-of-the-art baselines across a wide range of generation and editing benchmarks.

## Highlights

### Comparison with the state of the art

One sample per editing family, applying the same instruction with UniAVC and with
InstructAV2AV and MiniMax-H3.

#### Subject Editing

> Transform the man into a cyberpunk painting style with high-saturation neon colors and sharp digital illustration strokes, while keeping all characters, actions, and composition unchanged.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/2714d3e3-a5c2-44bd-9425-195c440f31ba" controls width="100%"></video><sub>Input</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/362b08e7-1968-416b-b6c1-93b6d3558fe6" controls width="100%"></video><sub><b>UniAVC</b></sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/0325bae2-c966-43dd-b444-6d5a9926ccd0" controls width="100%"></video><sub>InstructAV2AV</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/fdcf3d3d-6ac0-4eec-ae21-a48438a72afd" controls width="100%"></video><sub>MiniMax-H3</sub></td>
</tr>
</table>

#### Subject Addition

> Add a person wearing white clothes, standing still in the center of the frame, looking at the camera with a blank expression.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/8a6d68d7-860c-4398-a180-a90b670005f3" controls width="100%"></video><sub>Input</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/644c371c-cff3-45c8-8d90-5d2597797818" controls width="100%"></video><sub><b>UniAVC</b></sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/22e5d439-4421-4e77-b5dd-adaac811e526" controls width="100%"></video><sub>InstructAV2AV</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/0e063370-a404-41ad-b797-86aa016899b2" controls width="100%"></video><sub>MiniMax-H3</sub></td>
</tr>
</table>

#### Subject Removal

> Remove the woman wearing a white lab coat, a stethoscope, and looking confused or distressed from the source video, while preserving the background of a blurred office environment, a desk lamp, the wall calendar, and the shelving behind her.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/daa72578-1ade-4c14-a1d8-ecccd8aea0b7" controls width="100%"></video><sub>Input</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/a8d78c47-54bf-4f98-93ff-812368b9f600" controls width="100%"></video><sub><b>UniAVC</b></sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/f1aa71bf-6b0c-4d33-92c7-e733e804252c" controls width="100%"></video><sub>InstructAV2AV</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/f692416b-e5c7-44f4-a1ea-91d6af6a0d55" controls width="100%"></video><sub>MiniMax-H3</sub></td>
</tr>
</table>

#### Speech Editing

> Alter the character's dialogue to: "We were in the drum circle like, wow, this is where real musical magic happens right now!" while keeping visual elements and background sounds identical to the original scene.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/efcf6cbe-5e41-47bb-a3bf-f06e34a8847c" controls width="100%"></video><sub>Input</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/887536a3-b4d2-4bc5-921e-6b33a3ec1f74" controls width="100%"></video><sub><b>UniAVC</b></sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/8520eb36-cd95-42b7-bea3-7de6add89dd0" controls width="100%"></video><sub>InstructAV2AV</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/060cd939-7a7b-4e4d-bb90-01c7c78726a9" controls width="100%"></video><sub>MiniMax-H3</sub></td>
</tr>
</table>

#### Background Editing

> Replace the background with a private study at late night, with quiet city night lights visible outside the window, while keeping the person's image, clothing, actions, and contemplative head-down expression in the foreground completely consistent.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/845569e9-5908-4ad7-bda7-80febd8d04f2" controls width="100%"></video><sub>Input</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/3cd2350e-8b04-4ee6-bed1-f35e633f61ce" controls width="100%"></video><sub><b>UniAVC</b></sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/f21603c4-d703-4e8c-be35-a9a96252fb63" controls width="100%"></video><sub>InstructAV2AV</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/23de1a40-36f2-4ea6-9fdd-be780e7dbc39" controls width="100%"></video><sub>MiniMax-H3</sub></td>
</tr>
</table>

### Editing versatility

Each grid holds the source clip and the results of three different instructions applied to that same clip.

#### Subject Editing

Restyle the subject while preserving background, lighting and posture.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/34bfecc8-70a2-4ad1-8492-13566e1b9584" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/456253b7-1aca-46c4-ba0c-da89aa0e4a5f" controls width="100%"></video><sub><b>1.</b> Racing suit instead of the T-shirt</sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/fbbb2b09-8b09-4d84-b852-2c6b8ff24d4a" controls width="100%"></video><sub><b>2.</b> Astronaut pressure suit instead of the T-shirt</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/ce1a86ce-ee70-4daf-9179-9922de753f29" controls width="100%"></video><sub><b>3.</b> Black wetsuit with blue shoulder panels</sub></td>
</tr>
</table>

#### Subject Addition

Insert a new person into the shot.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/37e3a2af-1776-46a4-acce-b14385a15521" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/4f4d3f66-9772-476e-9508-1cc7db43e70e" controls width="100%"></video><sub><b>1.</b> Add a man with short brown hair</sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/4c33cd43-5e20-4b13-946c-ca9854303d27" controls width="100%"></video><sub><b>2.</b> Add a woman with long dark hair</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/af6bc175-d840-4045-9e8a-6be35c7bbb53" controls width="100%"></video><sub><b>3.</b> Add an older man with white hair and glasses</sub></td>
</tr>
</table>

#### Subject Removal

Erase a person and reconstruct what was behind them. This family ships two examples with one instruction each.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/a696f1c6-ef35-474b-82e8-68ef91361278" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/6b005055-2c6c-4982-a1c8-e111af080c4e" controls width="100%"></video><sub><b>1.</b> Remove the man in the white shirt and dark tie</sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/597ec1dd-55ba-4ff8-868e-ad84251bd02b" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/79d3d540-0a88-407d-98af-62dbc38a47d1" controls width="100%"></video><sub><b>2.</b> Remove the woman with the blue patterned scarf</sub></td>
</tr>
</table>

#### Speech Editing

Replace a line of dialogue with new words and matched lip-sync.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/021a86f9-c28f-4194-870f-3f7e5efad06a" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/068ab346-b563-4fe4-872f-0aa281ca5e94" controls width="100%"></video><sub><b>1.</b> "Scroll back to the log from Tuesday afternoon."</sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/0ad6356b-7b73-4346-85c7-c1a7822e9693" controls width="100%"></video><sub><b>2.</b> "That value should never have gone negative."</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/addeb97c-219d-49eb-a6d4-55ccb2b52e3f" controls width="100%"></video><sub><b>3.</b> "Look at the screen, the error code matches the original test."</sub></td>
</tr>
</table>

#### Background Editing

Relocate the background scene, keeping subject appearance and expression intact.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/df6398f9-7f28-43f5-bca6-3944cd0a512d" controls width="100%"></video><sub>Source</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/48558c3d-d929-46e2-8a2a-0f1c7985b1a9" controls width="100%"></video><sub><b>1.</b> A tranquil beach at dusk</sub></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/21095fa4-516e-4680-bb71-313f90d12c66" controls width="100%"></video><sub><b>2.</b> A rooftop at dusk, laundry and swifts</sub></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/38c80e36-0205-45e0-90d7-1495fab6c43f" controls width="100%"></video><sub><b>3.</b> A railway platform at dusk, a train pulling away</sub></td>
</tr>
</table>

### Joint generation

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/53039ba4-d600-403f-92e5-bff7af6b8878" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/60dc86ca-defa-434b-9340-ba638303755d" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/97f1ac18-32af-4845-b88f-7d079d9f3b04" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/2a421afb-af22-46ee-b453-cbb11cf39a4e" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/3a25d480-5ebb-430b-bf24-51267c3f6e3b" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/f9ea762a-7980-4521-8469-6fa0f2793666" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/b5d8e00b-a263-4861-a107-6f0379c000a0" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/939dd463-19a8-44fc-8da5-e2018423b993" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/ef5e9d81-4afe-478f-87e5-f36ec189dfdb" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/e8667e23-13e5-496e-a47d-6dcb04ec327a" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/b98f382f-f150-40f1-9aca-0300db807725" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/89a24d8d-b75f-4f41-8d89-44c583ca091f" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/284b2dae-06c8-4611-8328-72e68e5349ed" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/0fc02367-2ca9-4bb3-971e-3e7f00399cfe" controls width="100%"></video></td>
</tr>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/2e3efef3-9ddc-4631-8439-658c56e69cc2" controls width="100%"></video></td><td width="50%"><video src="https://github.com/user-attachments/assets/ef6cdb76-b13c-4688-a563-710b1c22ad57" controls width="100%"></video></td>
</tr>
</table>

## Installation

```bash
conda create -n uniavc python=3.10 -y
conda activate uniavc

# Install a torch build matching your CUDA, then the rest.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

**Tested environment:** Python 3.10, CUDA 12.8, PyTorch 2.8, on a single 80 GB GPU.

### Model weights

`bash scripts/download_weights.sh` fetches the weights, plus Gemma-3 for text
conditioning:

```bash
bash scripts/download_weights.sh
```

Or by hand:

```bash
pip install -U huggingface_hub

# The released model.
hf download TK3105/UniAVC model.safetensors \
    --local-dir models/UniAVC

# Gemma-3 text encoder.
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir models/gemma-3-12b-it-qat-q4_0-unquantized
```

Once downloaded, the weights are stored as follows:

```
models/
├── UniAVC/
│   └── model.safetensors                  # the DiT + the VAEs, vocoder, post-modules
└── gemma-3-12b-it-qat-q4_0-unquantized/
```

## Usage


### Joint audio-video editing

Edit the video and its soundtrack from a single instruction. Each of the five families
gets one clip and one instruction:

```bash
# Subject Editing
python inference/infer_tav2av.py \
    --video samples/sample_1.mp4 \
    --prompt "Replace the man's dark brown T-shirt with a blue cycling jersey with white stripes on the sleeves. Keep the pull-up bar, the beige wall and the lighting unchanged." \
    -o sample_1_out.mp4

# Subject Addition
python inference/infer_tav2av.py \
    --video samples/sample_2.mp4 \
    --prompt "Include a man in a grey coat seated in the front seat, saying: \"We should have left an hour ago. Nobody checked the traffic or the weather.\"" \
    -o sample_2_out.mp4

# Subject Removal
python inference/infer_tav2av.py \
    --video samples/sample_3.mp4 \
    --prompt "Remove the table lamp from the foreground and fill in the window and curtains behind it. Keep the man and his dialogue unchanged." \
    -o sample_3_out.mp4

# Speech Editing
python inference/infer_tav2av.py \
    --video samples/sample_4.mp4 \
    --prompt "Have her say: \"I did not expect the city to be this quiet at dawn.\"" \
    -o sample_4_out.mp4

# Background Editing
python inference/infer_tav2av.py \
    --video samples/sample_5.mp4 \
    --prompt "Replace the background with a warm, sunlit kitchen with wooden cabinets. Keep the man, his clothing and his speech unchanged." \
    -o sample_5_out.mp4
```

The two headline modes are extra flags on this script.

Post-inference self-refinement: re-denoise only one branch against the other, held as
an accepted reference.

```bash
python inference/infer_tav2av.py \
    --video samples/sample_4.mp4 \
    --prompt "Have her say: \"I did not expect the city to be this quiet at dawn.\"" \
    --post-refine audio --refine-lambda 0.5 \
    -o sample_4_refined.mp4
```

At the default `--refine-lambda 0.5` this runs `round(0.5 * 30) = 15` refinement steps.
The count scales with lambda so the refine pass keeps the main pass's per-step
resolution; `--refine-steps` overrides it.

Intent-aware editing: hold one branch intact by conditioning, and let it constrain the
other.

```bash
python inference/infer_tav2av.py \
    --video samples/sample_5.mp4 \
    --prompt "The room is now a quiet library, with faint page turns and distant footsteps." \
    --intent-aware video \
    -o sample_5_audio_edit.mp4
```

### Joint audio-video generation

No input media: both branches are generated from the prompt alone.

```bash
python inference/infer_t2av.py \
    --prompt "A man in a grey shirt sits at a wooden table in a quiet cafe, a window with soft daylight behind him and a few empty chairs further back. He says: \"We can start whenever you are ready, but let me finish this first.\" Static medium shot." \
    -o sample_t2av.mp4
```

### Audio → video generation

Given a soundtrack, generate the matching video; the input audio is preserved.

```bash
python inference/infer_ta2v.py \
    --audio samples/sample_4.mp4 \
    --prompt "Medium close-up of a woman with long dark hair speaking directly to camera, standing in a sunlit kitchen with pale oak cabinets and a steaming kettle on the counter behind her, warm late-afternoon light through a side window, shallow depth of field, static shot." \
    -o sample_4_ta2v.mp4
```

### Video → audio generation

Given a video, generate the matching soundtrack; the input video is preserved.

```bash
python inference/infer_tv2a.py \
    --video samples/sample_1.mp4 \
    --prompt "The man is out of breath after a set of pull-ups. A barbell is racked nearby in a bare gym, with voices and movement echoing off a hard wall." \
    -o sample_1_tv2a.mp4
```

### Options

Every option has a working default, and the commands above need nothing beyond
`--video` / `--audio` / `--prompt`.

**Output**

| Option | Default | Notes |
|---|---|---|
| `--height`, `--width` | `704`, `1280` | Both must be multiples of 32. |
| `--num-frames` | `121` | Must be 8k+1. |
| `--frame-rate` | `24` | |
| `-o`, `--output` | `<task>_output.mp4` | |

**Sampling and memory**

| Option | Default | Notes |
|---|---|---|
| `--num-inference-steps` | `30` | Denoising steps. |
| `--seed` | `42` | |
| `--offload` | on | `--no-offload` keeps all weights resident; see *Hardware*. |
| `--device` | `cuda` | |
| `--no-tiled` | off | Decode the video without VAE tiling. Needs more VRAM. |

**Guidance**, set per branch.

| Task | `video-text` | `audio-text` | `video-reference` | `audio-reference` |
|---|---|---|---|---|
| `tav2av` | 3.0 | 2.0 | 1.5 | 3.0 |
| `t2av`, `ta2v`, `tv2a` | 6.0 | 6.0 | 1.0 | 1.0 |

The flags are `--video-text-cfg-scale`, `--audio-text-cfg-scale`,
`--video-reference-cfg-scale` and `--audio-reference-cfg-scale`.

## Demo

A Gradio app is included:

```bash
python app.py
```

## Hardware

The released checkpoint is a 19B-parameter DiT in bfloat16, and text conditioning
adds Gemma-3 12B. The two together are ~65 GB, so weights are offloaded by default:
each phase loads only what it needs (the text encoder while encoding, the DiT while
denoising, the VAEs while decoding) and the rest waits on CPU. Peak residency is the
DiT alone, ~39 GB, plus activations. `--no-offload` keeps everything on the card for
speed and needs substantially more VRAM than a single 80 GB GPU provides at the
default 720p resolution.

## Acknowledgements

UniAVC is built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),
whose model-loading framework and LTX-2.3 implementation the vendored `uniavc/` package
derives from. The base generative model is
[LTX-2.3](https://github.com/Lightricks/LTX-Video) from Lightricks. Text conditioning uses
[Gemma-3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) from
Google. We thank [JAVEdit](https://github.com/RyanChenYN/JAVEdit) for the useful training data and their benchmark for evaluation; 
[MiniMax-H3](https://www.minimax.io/), [InstructAV2AV](https://github.com/suimuc/InstructAV2AV) for the
comparison baselines; and the wider open-source audio-video community that make this work complete.

## Citation

If you find UniAVC useful for your research, please consider giving a star ⭐ and citation 📝 :D

```bibtex
@article{pham2026uniavc,
  title   = {UniAVC: Unified Audio-Video Creation via Progressive Training and Asymmetric Scheduling},
  author  = {Pham, Kien T. and Liu, Yuhao and Choi, Chong Hou and He, Zhenqi and Wang, Yanghao
             and Lei, Chenyang and Liu, Rui and Chen, Qifeng and Chen, Long},
  journal = {arXiv preprint arXiv:2610.24791},
  year    = {2026}
}
```

## License

UniAVC's original contributions are released under CC BY-NC 4.0, for non-commercial use with attribution. Third-party components retain their own licenses.

The released model weights are a derivative of LTX-2.3 and are governed separately by the [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2).
