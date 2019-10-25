from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)

    args.save_path.mkdir(exist_ok=True, parents=True)

    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wavname = fname.name
        wav, sr = librosa.core.load(fname)

        mel, _ = vocoder(torch.from_numpy(wav)[None])
        recons = vocoder.inverse(mel).squeeze().cpu().numpy()

        librosa.output.write_wav(args.save_path / wavname, recons, sr=sr)


if __name__ == "__main__":
    main()
