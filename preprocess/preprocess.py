import os
import sys
import argparse
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import audio
import utils.utils as utils
from tqdm import tqdm
import pyworld as pw

import warnings
warnings.filterwarnings("ignore")

def extract_mel(wav, hparams):
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    return mel_spectrogram.T, wav

def extract_pitch(wav, hps):
    # rapt may be better
    f0, _ = pw.harvest(wav.astype(np.float64),
                   hps.sample_rate,
                   frame_period=hps.hop_size / hps.sample_rate * 1000)
    return f0

def process_utterance(hps, data_root, item):
    out_dir = data_root

    wav_path = os.path.join(data_root, "wavs",
                            "{}.wav".format(item))
    wav = audio.load_wav(wav_path,
                         raw_sr=hps.data.sample_rate,
                         target_sr=hps.data.sample_rate,
                         win_size=hps.data.win_size,
                         hop_size=hps.data.hop_size)

    mel, _ = extract_mel(wav, hps.data)
    out_mel_dir = os.path.join(out_dir, "mels")
    os.makedirs(out_mel_dir, exist_ok=True)
    mel_path = os.path.join(out_mel_dir, item)
    np.save(mel_path, mel)

    pitch = extract_pitch(wav, hps.data)
    out_pitch_dir = os.path.join(out_dir, "pitch")
    os.makedirs(out_pitch_dir, exist_ok=True)
    pitch_path = os.path.join(out_pitch_dir, item)
    np.save(pitch_path, pitch)

def process(args, hps):
    print(os.path.join(hps.data.data_dir, "wavs"))
    if(not os.path.exists(os.path.join(hps.data.data_dir, "file.list"))):
        with open(os.path.join(hps.data.data_dir, "file.list") ,"w") as out_file:
            files = os.listdir(os.path.join(hps.data.data_dir, "wavs"))
            for f in files:
                out_file.write(f.strip().split(".")[0] + '\n')
    metadata = [
        item.strip() for item in open(
            os.path.join(hps.data.data_dir, "file.list")).readlines()
    ]
    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    results = []
    for item in metadata:
        results.append(executor.submit(partial(process_utterance, hps, hps.data.data_dir, item)))
    return [result.result() for result in tqdm(results)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config.json',
                        help='json files for configurations.')
    parser.add_argument('--num_workers', type=int, default=int(cpu_count()) // 2)

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    process(args, hps)


if __name__ == "__main__":
    main()
