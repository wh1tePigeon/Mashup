# Mashup — Deep Learning for Audio Track Mashup Generation

This repository is a project on **automatic generation of audio mashups** by replacing the vocal of one track with the voice of another speaker or singer. The pipeline combines **music source separation**, optional **cocktail-fork (speech / music / effects) separation**, and **singing voice conversion (SoftVC VITS / so-vits-svc)**.

---

## Table of contents

1. [Overview](#overview)
2. [Problem statement](#problem-statement)
3. [Pipeline](#pipeline)
4. [Models](#models)
5. [Data collection and preprocessing](#data-collection-and-preprocessing)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Checkpoints](#checkpoints)
9. [Inference (create a mashup)](#inference-create-a-mashup)
10. [Training](#training)
11. [Configuration](#configuration)
12. [Limitations and future work](#limitations-and-future-work)
13. [Credits](#credits)
14. [License](#license)
15. [References](#references)

---

## Overview

Before modern AI tools, voice-swap mashups were built by hand: isolating vocals (phase inversion, crude filters), extracting a target voice, then aligning tempo, key, and mix. That process was slow, skill-heavy, and left residual music and noise in the vocal stem.

This project studies how **deep learning** can automate one common mashup type: **replace singer A’s vocal with singer/speaker B’s voice**, while keeping the original accompaniment and preserving musically important properties (notably pitch over time) so the new vocal still fits the instrumental.

---

## Problem statement

End-to-end mashup generation here is composed of several classical audio ML problems:

| Subproblem | Role in the mashup |
|---|---|
| **Music source separation** | Split a song into vocal and accompaniment (and optionally other stems). |
| **Cocktail fork problem** | When target “voice” comes from noisy speech (film, Telegram voice notes, talks), separate **speech / music / effects**. |
| **Singing voice conversion (SVC)** | Change identity of the source vocal to the target speaker **without changing linguistic content**, while preserving pitch and timing for remixing. |

Unlike plain speech voice conversion (e.g. deepfake-style speech), **singing** conversion must keep F0 contour and timing so the converted vocal can be mixed back with the instrumental.

---

## Pipeline

The main entry point is [`mashup.py`](./mashup.py). For a given input song (audio or video) it:

```
Input track
    │
    ▼
┌─────────────────────────────┐
│ 1. CascadedNet separation   │  → vocal stem + background stem
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. SoftVC VITS conversion   │  → vocal in target speaker’s timbre
│    (HuBERT + Whisper PPG    │
│     + pitch + speaker emb.) │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 3. Concatenate / mix        │  → final mashup audio
└─────────────────────────────┘
```

High-level code path:

1. **Separate** vocals with CascadedNet (`inference_cascaded`).
2. **Convert** the isolated vocal with SoftVC VITS (`inference_vits`).
3. **Mix** converted vocal with the original background (`concat_tracks`).

Hydra config: [`source/configs/mashup.yaml`](./source/configs/mashup.yaml).

---

## Models

### 1. Music source separation

Classical strong baselines include **Wave-U-Net** and **Hybrid Demucs** (high quality, but more resource-heavy). Public training data is scarce; the standard benchmark corpus **MUSDB18** has only 150 songs. Quality is often reported with **SDR** (Signal-to-Distortion Ratio), though SDR does not always match perceptual quality.

**In this repo:** a **CascadedNet**-style spectrogram separator is used for vocal / background stems (`source/model/cascaded/`, training via `source/train_model/train_cascaded.py`).

### 2. Cocktail fork problem (speech / music / effects)

Many real “target voice” sources are not clean studio vocals: they mix speech, music, and SFX/noise. The **cocktail fork** task separates a soundtrack into three stems: speech, music, and effects.

Training data historically relies on mixtures such as **Divide and Remaster (DnR)**, built from:

- **FSD50K** — environmental / effect sounds  
- **FMA** — music clips  
- **LibriSpeech** — clean speech  

Related work adapts **Band-Split RNN (BSRNN)** / cinematic bandit-style architectures for this three-stem separation.

**In this repo:** BSRNN training and inference live under `source/model/bsrnn/`, `source/train_model/train_bsrnn.py`, and `source/inference/bsrnn/`, with DnR-oriented datasets in `source/datasets/dnr/`.

### 3. Singing voice conversion — SoftVC VITS (so-vits-svc)

**VITS** is an end-to-end TTS model (posterior encoder, prior / text encoder with normalizing flow, stochastic duration predictor, HiFi-GAN decoder + discriminator, MAS alignment). SoftVC adapts VITS for **voice conversion from audio**:

- No phoneme alignment / duration prediction for TTS text.
- **Speaker-independent linguistic features** from audio via **HuBERT**, **Whisper** encoder (PPG-like features), and/or **ContentVec**.
- **Pitch tracking** (e.g. **CREPE** / World / torchcrepe) to preserve melody.
- The waveform decoder (vocoder generator) is pluggable; see below.

**In this repo:** SoftVC VITS synthesizer, discriminators, and inference are under `source/model/vits/` and `source/inference/vits/`. Feature extraction helpers are in `source/utils/preprocess_dataset.py` and `source/utils/pitch.py`.

#### Vocoder generator variants

SoftVC VITS turns latent features into audio via a neural vocoder. Both selectable generators inject an **NSF (Neural Source-Filter)** harmonic excitation derived from F0 (`SourceModuleHnNSF` in `source/model/vits/generator/nsf.py`), which helps preserve singing pitch. The active decoder is chosen with `hp.vocoder_name` in [`source/configs/vits/gen/gen_conf.yaml`](./source/configs/vits/gen/gen_conf.yaml) and wired in `SynthesizerTrn` / `SynthesizerInfer` (`source/model/vits/synthesizer.py`).

| `vocoder_name` | Implementation | Description |
|---|---|---|
| **`nsf-hifigan`** (default) | `NSFHifiGANGenerator` in [`nsfhifigan.py`](./source/model/vits/generator/nsfhifigan.py) | HiFi-GAN-style generator with NSF pitch source. Uses classic HiFi-GAN residual blocks (`ResBlock1` / `ResBlock2`), transposed-conv upsampling, and adds harmonic excitation at each upsample stage. Closer to the original VITS / so-vits-svc decoder. |
| **`bigvgan`** | `Generator` in [`generator.py`](./source/model/vits/generator/generator.py) (+ [`bigv.py`](./source/model/vits/generator/bigv.py), [`alias/`](./source/model/vits/generator/alias/)) | BigVGAN-style generator with NSF. Uses anti-aliased multi-periodicity (AMP) residual blocks and **Snake** periodic activations (`SnakeAlias`) instead of plain HiFi-GAN resblocks, aimed at fewer aliasing artifacts and higher-fidelity waveforms. Fallback default if `vocoder_name` is unknown. |
| **`MelGAN`** | [`melgan.py`](./source/model/vits/generator/melgan.py) | Classic MelGAN upsampling generator with residual blocks. Present in the codebase with a matching MelGAN discriminator. |


**Discriminators** (adversarial training) live under `source/model/vits/discriminator/`:

- **BigVGAN-style** (default in config): multi-period (MPD) + multi-resolution (MRD) — [`disc/disc_conf.yaml`](./source/configs/vits/disc/disc_conf.yaml)
- **HiFi-GAN-style**: MPD + multi-scale (MSD)
- **MelGAN-style**: multi-scale N-layer discriminators

To switch the SoftVC decoder at train/inference time, set in `gen_conf.yaml` (or via Hydra override):

```yaml
# source/configs/vits/gen/gen_conf.yaml
vocoder_name: "nsf-hifigan"   # or "bigvgan"
```

```bash
python source/train_model/train_vits.py gen.hp.vocoder_name=bigvgan
```

Checkpoints are named with the vocoder id (e.g. `checkpoint-nsf-hifigan-epochN.pth`), so use a checkpoint trained with the same `vocoder_name` you select at inference.

---

## Data collection and preprocessing

Training so-vits-svc needs **clean samples of the target speaker**. Cleaner data → better conversion.

### Public corpora

- **OpenSinger** — ~50 hours of studio singing (Chinese), 25 male / 41 female singers, short segments (~5 s) with transcripts.
- **LibriSpeech (clean)** — ~25–30 minutes per speaker; enough for a basic SVC speaker.

### Custom voices

| Source type | Preprocessing |
|---|---|
| Vocals from songs | Separate vocals with the music source separation model. |
| Speech (messages, talks, film) | Prefer cocktail-fork separation to remove music / SFX. |
| Telegram voice messages | Export chat history (voice messages + JSON metadata), then parse with `source/utils/parse_voice_messages.py`. |

### Post-processing

1. **VAD** — remove long silences (e.g. **SpeechBrain**).
2. **Slice** into segments roughly **2–25 seconds** (too short = weak content; too long = expensive; Whisper encoder max length is **30 s**).

Utility scripts: `source/utils/slice_audio.py`, `source/utils/preprocess_dataset.py`, `source/utils/process_audio.py`, `source/utils/separate_many_files.py`.

---

## Project structure

```
Mashup/
├── mashup.py                 # End-to-end mashup inference entrypoint
├── requirements.txt
├── input/                    # Place input media / speaker embeddings
├── source/
│   ├── configs/              # Hydra configs (mashup, cascaded, vits, bsrnn)
│   ├── model/                # CascadedNet, BSRNN, SoftVC VITS, HuBERT
│   ├── inference/            # Cascaded / VITS / BSRNN / Whisper inference
│   ├── train_model/          # Training scripts
│   ├── trainer/              # Trainer implementations
│   ├── datasets/             # Cascaded, VITS, DnR, classic speech datasets
│   ├── loss/                 # Loss modules
│   ├── metric/               # Metrics (e.g. SNR for BSRNN)
│   └── utils/                # Audio I/O, pitch, preprocessing, Telegram export
└── checkpoints/              # Download pretrained weights here (see below)
```

---

## Installation

**Requirements:** Python 3.10+ recommended, CUDA GPU strongly preferred for training and conversion.

```bash
git clone https://github.com/wh1tePigeon/Mashup
cd Mashup

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```


---

## Checkpoints

Download pretrained checkpoints from Google Drive and place them under the project root (paths referenced in configs):

**[Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1bJBM7BoL1AFII9OFHhD26TmClVupUzir?usp=drive_link)**

Typical layout expected by [`source/configs/mashup.yaml`](./source/configs/mashup.yaml):

```
checkpoints/
├── vits/          # SoftVC VITS weights (e.g. sovits5.0_*.pt)
├── hubert/        # soft HuBERT for content vectors
└── whisper/       # Whisper encoder checkpoint for PPG features
```

Also prepare a **target speaker embedding** (`.spk.npy`) and any cached features referenced in the config.

---

## Inference (create a mashup)

1. Put your input track under `input/` (audio or video supported by the loaders).
2. Edit `source/configs/mashup.yaml`:
   - `filepath` — path to the source song (`$ROOT` is the project root at runtime).
   - `vits.checkpoint_path`, `vits.spk`, feature/checkpoint paths as needed.
3. Optionally hardcode `FILEPATH` in `mashup.py`, or pass Hydra overrides.
4. Run:

```bash
python mashup.py
# or, for example:
python mashup.py filepath="$ROOT/input/your_song.mp3"
```

Outputs (by default under `output/`):

- `output/cascaded/...` — separated vocal and background  
- `output/vits/...` — converted vocal  
- `output/final/...` — remixed mashup  

You can also run cascaded or VITS inference modules independently via their configs under `source/configs/*/inference.yaml`.

---

## Training

Configs live in `source/configs/`. Training entrypoints:

| Model | Script | Config root |
|---|---|---|
| CascadedNet (vocal / accompaniment) | `source/train_model/train_cascaded.py` | `source/configs/cascaded/` |
| SoftVC VITS (so-vits-svc) | `source/train_model/train_vits.py` | `source/configs/vits/` |
| BSRNN (cocktail fork / cinematic stems) | `source/train_model/train_bsrnn.py` | `source/configs/bsrnn/` |

Examples:

```bash
# Music source separation
python source/train_model/train_cascaded.py

# Singing voice conversion
python source/train_model/train_vits.py

# Cocktail-fork style separation
python source/train_model/train_bsrnn.py
```

Hydra overrides work as usual, e.g. `n_gpu=1 trainer.epochs=100`. Logging supports TensorBoard / Weights & Biases via `source/logger/`.

---

## Configuration

Key mashup settings (`source/configs/mashup.yaml`):

- `cascaded.*` — separation output directory and save flags  
- `vits.*` — SoftVC checkpoint, speaker embedding, pitch / PPG / HuBERT feature paths  
- `concatenate.*` — mix paths and final output directory  

Model-specific architecture, optimizer, loss, dataset, and trainer configs are split into subfolders under each model’s config directory.

For SoftVC VITS, the waveform decoder is controlled by `gen.hp.vocoder_name` (`nsf-hifigan` or `bigvgan`) — see [Vocoder generator variants](#vocoder-generator-variants).

---

## Limitations and future work

As noted in the coursework report:

- Outputs still contain **artifacts** at separation and conversion stages.
- **Zero-shot** voice conversion (little or no target data) was not the focus of this work.
- A **GUI / web app / Telegram bot** would make the pipeline usable by non-technical users.

---

## Credits

This repository is a heavily modified fork of the [ASR project template](https://github.com/WrathOfGrapes/asr_project_template).

---

## License

MIT License — see [`LICENSE`](./LICENSE).

---

## References

Selected works discussed in the report:

1. Stoller et al. — [Wave-U-Net (music source separation)](https://arxiv.org/abs/1806.03185)  
2. Defferrard et al. — [FMA](https://arxiv.org/abs/1612.01840)  
3. Fonseca et al. — [FSD50K](https://arxiv.org/abs/2010.00475)  
4. Hsu et al. — [HuBERT](https://arxiv.org/abs/2106.07447)  
5. Huang et al. — [Multi-Singer](https://arxiv.org/abs/2106.11524) / [OpenSinger](https://arxiv.org/abs/2206.03055)  
6. Kim et al. — [Glow-TTS](https://arxiv.org/abs/2005.11129); [VITS](https://arxiv.org/abs/2106.06103)  
7. Kim et al. — [CREPE](https://arxiv.org/abs/1802.06182)  
8. Kong et al. — [HiFi-GAN](https://arxiv.org/abs/2010.05646)  
9. Lee et al. — [BigVGAN](https://arxiv.org/abs/2211.03502)  
10. Luo & Yu — [Band-Split RNN](https://arxiv.org/abs/2106.05628)  
11. Panayotov et al. — [LibriSpeech](https://www.isca-speech.org/archive/interspeech_2015/panayotov15_interspeech.html)  
12. Petermann et al. — [Cocktail Fork Problem](https://arxiv.org/abs/2008.04470)  
13. Qian et al. — [ContentVec](https://arxiv.org/abs/2210.15653)  
14. Radford et al. — [Whisper](https://openai.com/research/whisper)  
15. Rafii et al. — [MUSDB18](https://sigsep.github.io/datasets/musdb.html)  
16. Ravanelli et al. — [SpeechBrain](https://arxiv.org/abs/2106.04624)  
17. Rouard et al. — [Hybrid Demucs](https://arxiv.org/abs/2306.09541)  
18. Watcharasupat et al. — [Generalized bandsplit NN for cinematic separation](https://arxiv.org/abs/2307.08326)  
19. Zhou et al. — [VITS-based singing voice conversion (SVCC 2023)](https://arxiv.org/abs/2309.09106)
