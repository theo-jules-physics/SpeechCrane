# SpeechCrane: A Modular Text-to-Speech System

## Introduction

__SpeechCrane is a modular Text-to-Speech (TTS) system__ designed to facilitate the creation and implementation of various TTS architectures. The project aims to provide a flexible framework for experimenting with different TTS models, focusing on modularity and ease of use.

### Current Status (v0.1)

In this initial public release (v0.1), SpeechCrane focuses on the vocoder component of TTS systems. Two vocoder architectures are currently implemented:

1. __HiFiGAN:__ A high-resolution generative adversarial network for mel-spectrogram to waveform conversion.

2. __DiffWave:__ A differentiable waveform synthesizer that directly generates the waveform from the mel-spectrogram.

These vocoders are functional for the mel-spectrogram to waveform conversion. While the full end-to-end TTS pipeline is under development, it is not yet included in this public release.

__Note:__ The current trained models produce understandable but robotic-sounding voices. I am actively working on improving the quality of the synthesized speech.

### Features

* Modular architecture for easy implementation of new TTS models

* Support for both GAN-based (HiFiGAN) and diffusion-based (DiffWave) vocoders

* Designed for training on consumer-grade GPUs

* Flexible configuration system

* Automatic preprocessing of datasets

* WandB integration for experiment tracking (optional)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/theo-jules-physics/SpeechCrane.git
cd SpeechCrane
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

__Note:__ The requirements.txt file includes the CPU version of PyTorch. 
For optimal performance, it's recommended to install the CUDA-enabled version of PyTorch separately

## Usage

### Dataset Preparation

SpeechCrane requires a specific dataset format:

1. Create a dataset folder (path specified in `configs/data.json`)

2. In the dataset folder, create a `metadata.csv` file with the following columns (separated by |):
* file: base name of the audio file
* text: transcript of the audio file
* speaker_id: speaker ID

3. Organize audio files in the following structure:
```
dataset_folder/audio/speaker_id/audio_file
```

### Configuration

The system uses JSON configuration files located in the configs/ directory:
* `data.json`: Specifies dataset characteristics and preprocessing parameters
* `hifigan.json` / `diffwave.json`: Model-specific architectures and training parameters
* `training.json`: General training hyperparameters and settings

### Training

To start training a model, run the following command:

```bash
python model_training.py
```

The script will automatically preprocess the dataset and begin training based on the configuration files.

### Inference

To synthesize speech using a trained model, run the following command:

```bash
python inference.py --model_path /path/to/model/folder --mel_path /path/to/mel_spectrogram.npy
```

The generated audio will be saved as `gen_audio/output.wav`.


## Roadmap

### Short-term Goals
* Implement output masking for HiFiGAN training
* Resolve preprocessing issues
* Verify gradient accumulation functionality
* Add quick training tests
* Complete documentation
* Train and release example HiFiGAN and DiffWave models

### Medium-term Goals
* Verify weight normalization implementation
* Add pre-emphasis and de-emphasis for audio waveforms
* Develop a training profiler
* Optimize preprocessed data storage
* Enhance non-WandB logging and checkpointing
* Implement audio checkpoints for training progress tracking
* Develop acoustic models (text-to-mel) for HiFiGAN and DiffWave
* Create a unified inference class
* Add batch processing for inference

### Long-term Goals
* Implement additional architectures (VITS, VITS2, HierSpeech2, FreGrad, FastSpeech 2, GlowTTS, etc.)
* Support multi-GPU training
* Introduce more validation metrics
* Optimize models and training processes
* Develop a user-friendly GUI
* Implement Voice Conversion and Speaker Adaptation

## Contributing

SpeechCrane is an open-source project, and contributions are welcome! 
It is my first open-source project, so I appreciate any feedback or suggestions for improvement.
If you have any questions or ideas, feel free to reach out to me at theo.jules.physics@gmail.com.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements

SpeechCrane builds upon the work of numerous Text-to-Speech (TTS) models and papers. We are grateful to the authors of these works for their contributions to the field of speech synthesis. Below is a list of key architectures and papers that have influenced this project:


#### 2024

* __Fre-Grad:__
    * __Year:__ 2024
    * __Paper:__ [FreGrad: Lightweight and Fast Frequency-aware Diffusion Vocoder](https://arxiv.org/abs/2401.10032)
    * __Authors:__ Tan Dat Nguyen, Ji-Hoon Kim, Youngjoon Jang, Jaehun Kim, Joon Son Chung
    * __Official Implementation:__ [Fre-Grad](https://github.com/kaistmm/fregrad)

* __NaturalSpeech 3:__
    * __Year:__ 2024
    * __Paper:__ [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models](https://arxiv.org/abs/2403.03100)
    * __Authors:__ Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, Zhizheng Wu, Tao Qin, Xiang-Yang Li, Wei Ye, Shikun Zhang, Jiang Bian, Lei He, Jinyu Li, Sheng Zhao

#### 2023

* __MB-iSTFT-VITS:__
    * __Year:__ 2023
    * __Paper:__ [Lightweight and High-Fidelity End-to-End Text-to-Speech with Multi-Band Generation and Inverse Short-Time Fourier Transform](https://arxiv.org/abs/2210.15975)
    * __Authors:__ Masaya Kawamura, Yuma Shirahata, Ryuichi Yamamoto, Kentaro Tachibana
    * __Official Implementation:__ [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)

* __StyleTTS 2:__
    * __Year:__ 2023
    * __Paper:__ [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://arxiv.org/abs/2306.07691)
    * __Authors:__ Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani
    * __Official Implementation:__ [StyleTTS2](https://github.com/yl4579/StyleTTS2)

* __HierSpeech++:__
    * __Year:__ 2023
    * __Paper:__ [HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis](https://arxiv.org/abs/2311.12454)
    * __Authors:__ Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee
    * __Official Implementation:__ [HierSpeechpp](https://github.com/sh-lee-prml/HierSpeechpp)

* __VITS2:__
    * __Year:__ 2023
    * __Paper:__ [VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design](https://arxiv.org/abs/2307.16430)
    * __Authors:__ Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim
    * __Other implementations:__ 
      - [vits2s](https://github.com/daniilrobnikov/vits2) | Daniil Robnikov
      - [vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) | p0p4k et al.

* __NaturalSpeech 2:__
    * __Year:__ 2023
    * __Paper:__ [NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers](https://arxiv.org/abs/2304.09116)
    * __Authors:__ Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, Jiang Bian
    * __Other implementations:__ 
      - [naturalspeech2-pytorch](https://github.com/lucidrains/naturalspeech2-pytorch) | Phil Wang et al.

#### 2022

* __NaturalSpeech:__
    * __Year:__ 2022
    * __Paper:__ [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/pdf/2205.04421)
    * __Authors:__ Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, Frank Soong, Tao Qin, Sheng Zhao, Tie-Yan Liu

#### 2021

* __UnivNet:__
    * __Year:__ 2021
    * __Paper:__ [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)
    * __Authors:__ Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, Juntae Kim
    * __Other implementations:__ 
      - [univnet](https://github.com/maum-ai/univnet) | Kang-wook Kim et al.

* __DiffWave:__
    * __Year:__ 2021
    * __Paper:__ [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761)
    * __Authors:__ Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro
    * __Other implementations:__ 
      - [diffwave](https://github.com/lmnt-com/diffwave) | Sharvil Nanavati et al.

* __YourTTS:__
    * __Year:__ 2021-2023
    * __Paper:__ [YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone](https://arxiv.org/pdf/2112.02418)
    * __Authors:__ Edresson Casanova, Julian Weber, Christopher Shulby, Arnaldo Candido Junior, Eren Gölge, Moacir Antonelli Ponti
    * __Official Implementation:__ [YourTTS](https://github.com/Edresson/YourTTS)

* __VITS:__
    * __Year:__ 2021
    * __Paper:__ [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
    * __Authors:__ Jaehyeon Kim, Jungil Kong, Juhee Son
    * __Official Implementation:__ [VITS](https://github.com/jaywalnut310/vits)
    * __Other implementations:__ 
      - [VITS and derivatives](https://github.com/34j/awesome-vits)

* __Fre-GAN:__
    * __Year:__ 2021
    * __Paper:__ [Fre-GAN: Adversarial Frequency-consistent Audio Synthesis](https://arxiv.org/abs/2106.02297)
    * __Authors:__ Ji-Hoon Kim, Sang-Hoon Lee, Ji-Hyun Lee, Seong-Whan Lee
    * __Other implementations:__ 
      - [Fre-GAN-pytorch](https://github.com/rishikksh20/Fre-GAN-pytorch) | Rishikesh and George Grigorev.
      
 * __GradTTS:__
    * __Year:__ 2021
    * __Paper:__ [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337)
    * __Authors:__ Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov
    * __Official Implementation:__ [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)  
   
#### 2020

* __HiFi-GAN:__
  * __Year:__ 2020
  * __Paper:__ [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
  * __Authors:__ Jungil Kong, Jaehyeon Kim, Jaekyoung Bae
  * __Official Implementation:__ [HiFi-GAN](https://github.com/jik876/hifi-gan)
  
* __Fastspeech2:__
    * __Year:__ 2020
    * __Paper:__ [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
    * __Authors:__ Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu
    * __Other implementations:__ 
      - [Fastspeech2](https://github.com/ming024/FastSpeech2) | Chung-Ming Chien, Chien-yu Huang
      - [Fastspeech2](https://github.com/rishikksh20/FastSpeech2) | Rishikesh et al.

* __GlowTTS:__
    * __Year:__ 2020
    * __Paper:__ [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129)
    * __Authors:__ Jaehyeon Kim, Sungwon Kim, Jungil Kong, Sungroh Yoon
    * __Official Implementation:__ [Glow-TTS](https://github.com/jaywalnut310/glow-tts)

#### 2019

* __FastSpeech__:
    * __Year:__ 2019
    * __Paper:__ [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
    * __Authors:__ Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu

* __WaveGrad:__
    * __Year:__ 2020
    * __Paper:__ [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
    * __Authors:__ Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan
    * __Other implementations:__ 
      - [wavegrad](https://github.com/lmnt-com/wavegrad) | Sharvil Nanavati et al.
      - [WaveGrad](https://github.com/ivanvovk/WaveGrad) | Ivan Vovk

* __MelGAN:__
    * __Year:__ 2019
    * __Paper:__ [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
    * __Authors:__ Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville
    * __Official Implementation:__ [melgan-neurips](https://github.com/descriptinc/melgan-neurips)

#### 2018

* __WaveGlow:__
    * __Year:__ 2018
    * __Paper:__ [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
    * __Authors:__ Ryan Prenger, Rafael Valle, Bryan Catanzaro
    * __Official Implementation:__ [WaveGlow](https://github.com/NVIDIA/waveglow)

#### 2016
      
* __WaveNet:__
  * __Year:__ 2016
  * __Paper:__ [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
  * __Authors:__ Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
  * __Other implementations:__ 
  - [WaveNet](https://github.com/golbin/WaveNet) | Jin Kim and Hyeokryeol Yang
  - [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet) | Vincent Herrmann and Junsoo Lee
  
#### Review Papers
* [Recent Advances in End-to-End Automatic Speech Recognition](https://arxiv.org/abs/2111.01690) | 2022 | Jinyu Li
* [A Survey on Neural Speech Synthesis](https://arxiv.org/abs/2106.15561) | 2021 | Xu Tan, Tao Qin, Frank Soong, Tie-Yan Liu
* [A review of deep learning techniques for speech processing](https://arxiv.org/abs/2305.00359) | 2023 | Ambuj Mehrish, Navonil Majumder, Rishabh Bhardwaj, Rada Mihalcea, Soujanya Poria

#### Other TTS projects
* [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
* [coqui-tts](https://github.com/coqui-ai/TTS)
* [MeloTTS](https://github.com/myshell-ai/MeloTTS)

I am grateful to the authors of these works for their contributions to the field of speech synthesis.

