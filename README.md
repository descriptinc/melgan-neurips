# Official repository for the paper MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis

Previous works have found that generating coherent raw audio waveforms with GANs is challenging. In this [paper](https://arxiv.org/abs/1910.06711), we show that it is possible to train GANs reliably to generate high quality coherent waveforms by introducing a set of architectural changes and simple training techniques. Subjective evaluation metric (Mean Opinion Score, or MOS) shows the effectiveness of the proposed approach for high quality mel-spectrogram inversion. To establish the generality of the proposed techniques, we show qualitative results of our model in speech synthesis, music domain translation and unconditional music synthesis. We evaluate the various components of the model through ablation studies and suggest a set of guidelines to design general purpose discriminators and generators for conditional sequence synthesis tasks. Our model is non-autoregressive, fully convolutional, with significantly fewer parameters than competing models and generalizes to unseen speakers for mel-spectrogram inversion. Our pytorch implementation runs at more than 100x faster than realtime on GTX 1080Ti GPU and more than 2x faster than real-time on CPU, without any hardware specific optimization tricks. Blog post with samples and accompanying code coming soon.

Visit our [website](https://melgan-neurips.github.io) for samples. You can try the speech correction application [here](https://www.descript.com/overdub) created based on the end-to-end speech synthesis pipeline using MelGAN.

Check the [slides](melgan_slides.pdf) if you aren't attending the NeurIPS 2019 conference to check out our poster.


## Code organization

    ├── README.md             <- Top-level README.
    ├── set_env.sh            <- Set PYTHONPATH and CUDA_VISIBLE_DEVICES.
    │
    ├── mel2wav
    │   ├── dataset.py           <- data loader scripts
    │   ├── modules.py           <- Model, layers and losses
    │   ├── utils.py             <- Utilities to monitor, save, log, schedule etc.
    │
    ├── scripts
    │   ├── train.py                    <- training / validation / etc scripts
    │   ├── generate_from_folder.py


## Preparing dataset
Create a raw folder with all the samples stored in `wavs/` subfolder.
Run these commands:
   ```command
   ls wavs/*.wav | tail -n+10 > train_files.txt
   ls wavs/*.wav | head -n10 > test_files.txt
   ```

## Training Example
    . source set_env.sh 0
    # Set PYTHONPATH and use first GPU
    python scripts/train.py --save_path logs/baseline --path <root_data_folder>


## PyTorch Hub Example
    import torch
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    vocoder.inverse(audio)  # audio (torch.tensor) -> (batch_size, 80, timesteps)
