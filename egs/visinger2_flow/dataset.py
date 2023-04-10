
import os
import sys
import string
import random
import numpy as np
import math
import json
from torch.utils.data import DataLoader
import torch

sys.path.append('../..')
from utils.audio import load_wav
from text import npu

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, hparams, fileid_list_path):
        self.hparams = hparams
        self.fileid_list = self.get_fileid_list(fileid_list_path)
        random.seed(hparams.train.seed)
        random.shuffle(self.fileid_list)
        if(hparams.data.n_speakers > 0):
            self.utt2spk = self.get_spk_map(os.path.join(hparams.data.data_dir, "utt2spk"))

    def get_spk_map(self, utt2spk_path):
        utt2spk = {}
        print(utt2spk_path)
        with open(utt2spk_path, encoding='utf-8') as f:
            for line in f.readlines():
                utt, spk = line.strip().split()
                utt2spk[utt] = int(spk)
        return utt2spk

    def get_fileid_list(self, fileid_list_path):
        fileid_list = []
        with open(fileid_list_path, 'r') as f:
            for line in f.readlines():
                fileid_list.append(line.strip())

        return fileid_list

    def __len__(self):
        return len(self.fileid_list)

class SingDataset(BaseDataset):
    def __init__(self, hparams, data_dir, fileid_list_path, label_list_path):
        BaseDataset.__init__(self, hparams, os.path.join(data_dir, fileid_list_path))
        self.hps = hparams

        with open(os.path.join(data_dir, label_list_path), "r") as in_file:
            self.id2label = {}
            for line in in_file.readlines():
                fileid, txt, phones, pitchid, dur, gtdur, slur = line.split('|')
                self.id2label[fileid] = [phones, pitchid, dur, slur, gtdur]

        self.mel_dir = os.path.join(data_dir, "mels")
        self.f0_dir = os.path.join(data_dir, "pitch")
        self.wav_dir = os.path.join(data_dir, "wavs")
        # self.__filter__()

    def __filter__(self):
        new_fileid_list = []
        print("before filter: ", len(self.fileid_list))
        for file_id in self.fileid_list:
            _is_qualified = True
            if(not os.path.exists(os.path.join(self.label_dir, self.fileid_list[index] + '.lab')) or 
                not os.path.exists(os.path.join(self.dur_dir, self.fileid_list[index] + '.lab')) or 
                not os.path.exists(os.path.join(self.mel_dir, self.fileid_list[index] + '.npy')) or 
                not os.path.exists(os.path.join(self.pitch_dir, self.fileid_list[index] + '.npy'))):
                _is_qualified = False
            if(_is_qualified):
                new_fileid_list.append(file_id)
        self.fileid_list = new_fileid_list
        print("after filter: ", len(self.fileid_list))

    def interpolate_f0(self, data):
        '''
        对F0进行插值处理
        '''
        data = np.reshape(data, (data.size, 1))

        vuv_vector = np.zeros((data.size, 1),dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return ip_data, vuv_vector

    def parse_label(self, pho, pitchid, dur, slur, gtdur):
        phos = []
        pitchs = []
        durs = []
        slurs = []
        gtdurs = []

        for index in range(len(pho.split())):
            phos.append(npu.symbol_converter.ttsing_phone_to_int[pho.strip().split()[index]])
            pitchs.append(npu.symbol_converter.ttsing_opencpop_pitch_to_int[pitchid.strip().split()[index]])
            durs.append(float(dur.strip().split()[index]))
            slurs.append(int(slur.strip().split()[index]))
            gtdurs.append(float(gtdur.strip().split()[index]))

        phos = np.asarray(phos, dtype=np.int32)
        pitchs = np.asarray(pitchs, dtype=np.int32)
        durs = np.asarray(durs, dtype=np.float32)
        slurs = np.asarray(slurs, dtype=np.int32)
        gtdurs = np.asarray(gtdurs, dtype=np.float32)

        #gtdurs = np.ceil(gtdurs / (self.hps.data.hop_size / self.hps.data.sample_rate))
        acc_duration = np.ceil(np.cumsum(gtdurs) / (self.hps.data.hop_size / self.hps.data.sample_rate))
        acc_duration = np.append(0, acc_duration)
        gtdurs = np.diff(acc_duration)

        phos = torch.LongTensor(phos)
        pitchs = torch.LongTensor(pitchs)
        durs = torch.FloatTensor(durs)
        slurs = torch.LongTensor(slurs)
        gtdurs = torch.LongTensor(gtdurs)
        return phos, pitchs, durs, slurs, gtdurs

    def __getitem__(self, index):
        pho, pitchid, dur, slur, gtdur = self.id2label[self.fileid_list[index]]
        pho, pitchid, dur, slur, gtdur = self.parse_label(pho, pitchid, dur, slur, gtdur)
        sum_dur = gtdur.sum()
        
        mel = np.load(os.path.join(self.mel_dir, self.fileid_list[index] + '.npy'))
        assert mel.shape[1] == 80
        if(mel.shape[0] != sum_dur):
            if(abs(mel.shape[0] - sum_dur) > 40):
                print("dataset error mel: ",mel.shape, sum_dur)
                return None
            if(mel.shape[0] > sum_dur):
                mel = mel[:sum_dur]
            else:
                mel = np.concatenate([mel, mel.min() * np.ones([sum_dur - mel.shape[0], self.hps.data.acoustic_dim])], axis=0)
        mel = torch.FloatTensor(mel).transpose(0, 1)

        f0 = np.load(os.path.join(self.f0_dir, self.fileid_list[index] + '.npy')).reshape([-1])
        f0, _ = self.interpolate_f0(f0)
        f0 = f0.reshape([-1])
        if(f0.shape[0] != sum_dur):
            if(abs(f0.shape[0] - sum_dur) > 40):
                print("dataset error f0 : ",f0.shape, sum_dur)
                return None
            if(f0.shape[0] > sum_dur):
                f0 = f0[:sum_dur]
            else:
                f0 = np.concatenate([f0, np.zeros([sum_dur - f0.shape[0]])], axis=0)
        f0 = torch.FloatTensor(f0).reshape([1, -1])
        
        wav = load_wav(os.path.join(self.wav_dir, self.fileid_list[index] + '.wav'), 
                       raw_sr=self.hparams.data.sample_rate,
                       target_sr=self.hparams.data.sample_rate, 
                       win_size=self.hparams.data.win_size, 
                       hop_size=self.hparams.data.hop_size)
        wav = wav.reshape(-1)
        if(wav.shape[0] != sum_dur * self.hparams.data.hop_size):
            if(abs(wav.shape[0] - sum_dur * self.hparams.data.hop_size) > 40 * self.hparams.data.hop_size):
                print("dataset error wav : ", wav.shape, sum_dur)
                return None
            if(wav.shape[0] > sum_dur * self.hparams.data.hop_size):
                wav = wav[:sum_dur * self.hparams.data.hop_size]
            else:
                wav = np.concatenate([wav, np.zeros([sum_dur * self.hparams.data.hop_size - wav.shape[0]])], axis=0)
        wav = torch.FloatTensor(wav).reshape([1, -1])

        if(self.hparams.data.n_speakers > 0):
            spkid = self.utt2spk[self.fileid_list[index]]
        else:
            spkid = 0

        return pho, pitchid, dur, slur, gtdur, mel, f0, wav, spkid


class SingCollate():

    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_dim = self.hparams.data.acoustic_dim

    def __call__(self, batch):
        
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        
        max_phone_len = max([len(x[0]) for x in batch])
        max_pitchid_len = max([len(x[1]) for x in batch])
        max_dur_len = max([len(x[2]) for x in batch])
        max_slur_len = max([len(x[3]) for x in batch])
        max_gtdur_len = max([len(x[4]) for x in batch])
        max_mel_len = max([x[5].size(1) for x in batch])
        max_f0_len = max([x[6].size(1) for x in batch])
        max_wav_len = max([x[7].size(1) for x in batch])

        phone_lengths = torch.LongTensor(len(batch))
        pitchid_lengths = torch.LongTensor(len(batch))
        dur_lengths = torch.LongTensor(len(batch))
        slur_lengths = torch.LongTensor(len(batch))
        gtdur_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        f0_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        phone_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchid_padded = torch.LongTensor(len(batch), max_pitchid_len)
        dur_padded = torch.FloatTensor(len(batch), max_dur_len)
        slur_padded = torch.LongTensor(len(batch), max_slur_len)
        gtdur_padded = torch.LongTensor(len(batch), 1, max_gtdur_len)
        mel_padded = torch.FloatTensor(len(batch), self.hparams.data.acoustic_dim, max_mel_len)
        f0_padded = torch.FloatTensor(len(batch), 1, max_f0_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkid_padded = torch.LongTensor(len(batch))

        phone_padded.zero_()
        pitchid_padded.zero_()
        dur_padded.zero_()
        slur_padded.zero_()
        gtdur_padded.zero_()
        mel_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        spkid_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            phone = row[0]
            phone_padded[i, :phone.size(0)] = phone
            phone_lengths[i] = phone.size(0)

            pitchid = row[1]
            pitchid_padded[i, :pitchid.size(0)] = pitchid
            pitchid_lengths[i] = pitchid.size(0)

            dur = row[2]
            dur_padded[i, :dur.size(0)] = dur
            dur_lengths[i] = dur.size(0)

            slur = row[3]
            slur_padded[i, :slur.size(0)] = slur
            slur_lengths[i] = slur.size(0)

            gtdur = row[4]
            gtdur_padded[i, :, :gtdur.size(0)] = gtdur
            gtdur_lengths[i] = gtdur.size(0)

            mel = row[5]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)
            
            f0 = row[6]
            f0_padded[i, :, :f0.size(1)] = f0
            f0_lengths[i] = f0.size(1)

            wav = row[7]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
 
            spkid = row[8]
            spkid_padded[i] = spkid
            
        data_dict = {}
        data_dict["phone"] = phone_padded
        data_dict["phone_lengths"] = phone_lengths
        data_dict["pitchid"] = pitchid_padded
        data_dict["dur"] = dur_padded
        data_dict["slur"] = slur_padded
        data_dict["gtdur"] = gtdur_padded
        data_dict["mel"] = mel_padded
        data_dict["f0"] = f0_padded
        data_dict["wav"] = wav_padded
        data_dict["spkid"] = spkid_padded

        data_dict["mel_lengths"] = mel_lengths
        data_dict["f0_lengths"] = f0_lengths
        data_dict["wav_lengths"] = wav_lengths
        return data_dict


class DatasetConstructor():

    def __init__(self, hparams, num_replicas=1, rank=1):
        self.hparams = hparams
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_function = {"SingDataset": SingDataset}
        self.collate_function = {"SingCollate": SingCollate}
        self._get_components()

    def _get_components(self):
        self._init_datasets()
        self._init_collate()
        self._init_data_loaders()

    def _init_datasets(self):
        self._train_dataset = self.dataset_function[self.hparams.data.dataset_type](self.hparams, self.hparams.data.data_dir, self.hparams.data.training_filelist, self.hparams.data.training_labellist)
        self._valid_dataset = self.dataset_function[self.hparams.data.dataset_type](self.hparams, self.hparams.data.data_dir, self.hparams.data.validation_filelist, self.hparams.data.validation_labellist)

    def _init_collate(self):
        self._collate_fn = self.collate_function[self.hparams.data.collate_type](self.hparams)

    def _init_data_loaders(self):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self._train_dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=True)
        
        self.train_loader = DataLoader(self._train_dataset, num_workers=4, shuffle=False,
                                       batch_size=self.hparams.train.batch_size, pin_memory=True,
                                       drop_last=True, collate_fn=self._collate_fn, sampler=train_sampler)

        self.valid_loader = DataLoader(self._valid_dataset, num_workers=1, shuffle=False,
                                       batch_size=1, pin_memory=True,
                                       drop_last=True, collate_fn=self._collate_fn)

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

