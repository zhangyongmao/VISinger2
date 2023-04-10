import os
import sys
import json
import argparse
import itertools
import math
import time
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

sys.path.append('../..')
import modules.commons as commons
import utils.utils as utils

from dataset import DatasetConstructor

from models import (
  SynthesizerTrn,
  Discriminator
)

from modules.losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
)
from preprocess.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch

torch.backends.cudnn.benchmark = True
global_step = 0
use_cuda = torch.cuda.is_available()
print("use_cuda, ", use_cuda)


numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def main():
  """Assume Single Node Multi GPUs Training Only"""

  hps = utils.get_hparams()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(hps.train.port)

  if(torch.cuda.is_available()):
    n_gpus = torch.cuda.device_count()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
  else:
    cpurun(0, 1, hps)
    
def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.train.save_dir)
    logger.info(hps.train)
    logger.info(hps.data)
    logger.info(hps.model)
    utils.check_git_hash(hps.train.save_dir)
    writer = SummaryWriter(log_dir=hps.train.save_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  dataset_constructor = DatasetConstructor(hps, num_replicas=n_gpus, rank=rank)

  train_loader = dataset_constructor.get_train_loader()
  if rank == 0:
      valid_loader = dataset_constructor.get_valid_loader()

  net_g = SynthesizerTrn(hps).cuda(rank)
  net_d = Discriminator(hps, hps.model.use_spectral_norm).cuda(rank)
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, valid_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()

def cpurun(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.train.save_dir)
    logger.info(hps.train)
    logger.info(hps.data)
    logger.info(hps.model)
    utils.check_git_hash(hps.train.save_dir)
    writer = SummaryWriter(log_dir=hps.train.save_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))
  torch.manual_seed(hps.train.seed)
  dataset_constructor = DatasetConstructor(hps, num_replicas=n_gpus, rank=rank)

  train_loader = dataset_constructor.get_train_loader()
  if rank == 0:
      valid_loader = dataset_constructor.get_valid_loader()

  net_g = SynthesizerTrn(hps)
  net_d = Discriminator(hps, hps.model.use_spectral_norm)
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_g, optim_g)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, valid_loader], logger, [writer, writer_eval])
    
    scheduler_g.step()
    scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, data_dict in enumerate(train_loader):
    
    phone = data_dict["phone"]
    pitchid = data_dict["pitchid"]
    dur = data_dict["dur"]
    slur = data_dict["slur"]
    gtdur = data_dict["gtdur"]
    mel = data_dict["mel"]
    f0 = data_dict["f0"]
    wav = data_dict["wav"]
    spkid = data_dict["spkid"]
    
    phone_lengths = data_dict["phone_lengths"]
    mel_lengths = data_dict["mel_lengths"]
    wav_lengths = data_dict["wav_lengths"]
    f0_lengths = data_dict["f0_lengths"]

    # data
    if(use_cuda):
        phone, phone_lengths = phone.cuda(rank, non_blocking=True), phone_lengths.cuda(rank, non_blocking=True)
        pitchid = pitchid.cuda(rank, non_blocking=True)
        dur = dur.cuda(rank, non_blocking=True)
        slur  = slur.cuda(rank, non_blocking=True)
        gtdur = gtdur.cuda(rank, non_blocking=True)
        mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
        wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)
        f0, f0_lengths = f0.cuda(rank, non_blocking=True), f0_lengths.cuda(rank, non_blocking=True)
        spkid = spkid.cuda(rank, non_blocking=True)
    
    # forward
    y_hat, ids_slice, predict_dur, predict_lf0, LF0, y_ddsp, kl_div, predict_mel, mask = net_g(phone, phone_lengths, pitchid, dur, slur, gtdur, f0, mel, mel_lengths, spk_id=spkid)
    y_ddsp = y_ddsp.unsqueeze(1)

    # Discriminator
    y = commons.slice_segments(wav, ids_slice * hps.data.hop_size, hps.train.segment_size) # slice 
    y_ddsp_mel = mel_spectrogram_torch(
          y_ddsp.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )

    y_logspec = torch.log(spectrogram_torch(
          y.squeeze(1),
          hps.data.n_fft,
          hps.data.sample_rate,
          hps.data.hop_size,
          hps.data.win_size
    ) + 1e-7)

    y_ddsp_logspec = torch.log(spectrogram_torch(
          y_ddsp.squeeze(1),
          hps.data.n_fft,
          hps.data.sample_rate,
          hps.data.hop_size,
          hps.data.win_size
    ) + 1e-7)

    y_mel = mel_spectrogram_torch(
          y.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )

    y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc

    optim_d.zero_grad()
    loss_disc_all.backward()
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    optim_d.step()

    # loss
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
    
    loss_f0 = F.mse_loss(predict_lf0 * mask, LF0 * mask) * 10
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
    loss_mel_dsp = F.l1_loss(y_mel, y_ddsp_mel) * 45
    loss_spec_dsp = F.l1_loss(y_logspec, y_ddsp_logspec) * 45
 
   
    loss_dur = F.mse_loss(predict_dur[:, 0, :], gtdur.float().squeeze(1)) * 0.1
    loss_note_dur = F.mse_loss(predict_dur[:, 1, :], dur.float().squeeze(1)) * 0.1

    loss_mel_am = F.mse_loss(mel * mask, predict_mel * mask) #* 10

    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)

    loss_fm = loss_fm / 9 * 6
    loss_gen = loss_gen / 9 * 6
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_f0 + loss_mel_dsp + loss_dur + kl_div + loss_mel_am + loss_spec_dsp

    loss_gen_all = loss_gen_all / hps.train.accumulation_steps
    
    loss_gen_all.backward()
    if((global_step+1) % hps.train.accumulation_steps == 0):
      grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
      optim_g.step()
      optim_g.zero_grad()
    
    if rank==0:
      if (global_step+1) % (hps.train.accumulation_steps * 10) == 0:
        logger.info(["step&time", global_step, time.asctime( time.localtime(time.time()) )])
        logger.info(["mel&mel_dsp&spec_dsp: " ,loss_mel, loss_mel_dsp, loss_spec_dsp])
        logger.info(["f0: " ,loss_f0])
        logger.info(["adv&fm: " ,loss_gen, loss_fm])
        logger.info(["kl: " ,kl_div])
        logger.info(["am&dur: " , loss_mel_am, loss_dur])

        
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_gen_all, loss_mel, loss_f0]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/total": loss_gen_all, 
                       "loss/mel": loss_mel,
                       "loss/adv": loss_gen,
                       "loss/fm": loss_fm,
                       "loss/mel_ddsp": loss_mel_dsp,
                       "loss/spec_ddsp": loss_spec_dsp,
                       "loss/dur": loss_dur,
                       "loss/note_dur": loss_note_dur,
                       "loss/mel_am": loss_mel_am,
                       "loss/kl_div": kl_div,
                       "loss/f0": loss_f0,
                       "learning_rate": lr}
        
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        logger.info(['All training params(G): ', utils.count_parameters(net_g), ' M'])
        #print('Sub training params(G): ', \
        #      'text_encoder: ', utils.count_parameters(net_g.module.text_encoder), ' M, ', \
        #      'decoder: ', utils.count_parameters(net_g.module.decoder), ' M, ', \
        #      'mel_decoder: ', utils.count_parameters(net_g.module.mel_decoder), ' M, ', \
        #      'dec: ', utils.count_parameters(net_g.module.dec), ' M, ', \
        #      'dec_harm: ', utils.count_parameters(net_g.module.dec_harm), ' M, ', \
        #      'dec_noise: ', utils.count_parameters(net_g.module.dec_noise), ' M, ', \
        #      'posterior: ', utils.count_parameters(net_g.module.posterior_encoder), ' M, ', \
        #     )
        
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.train.save_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.train.save_dir, "D_{}.pth".format(global_step)))
        net_g.train()
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, data_dict in enumerate(eval_loader):
        
        phone = data_dict["phone"]
        pitchid = data_dict["pitchid"]
        dur = data_dict["dur"]
        slur = data_dict["slur"]
        # gtdur = data_dict["gtdur"]
        mel = data_dict["mel"]
        f0 = data_dict["f0"]
        wav = data_dict["wav"]
        spkid = data_dict["spkid"]
        
        phone_lengths = data_dict["phone_lengths"]
        mel_lengths = data_dict["mel_lengths"]
        wav_lengths = data_dict["wav_lengths"]
        f0_lengths = data_dict["f0_lengths"]

        # data
        if(use_cuda):
            phone, phone_lengths = phone.cuda(0), phone_lengths.cuda(0)
            pitchid = pitchid.cuda(0)
            dur = dur.cuda(0)
            slur  = slur.cuda(0)
            wav = wav.cuda(0)
            mel = mel.cuda(0)
            f0  = f0.cuda(0)
            spkid  = spkid.cuda(0)
        
        # remove else
        phone = phone[:1]
        phone_lengths = phone_lengths[:1]
        pitchid = pitchid[:1]
        dur = dur[:1]
        slur = slur[:1]
        wav = wav[:1]
        mel = mel[:1]
        f0 = f0[:1]
        spkid = spkid[:1]
        break
      y_hat, y_harm, y_noise = generator.module.infer(phone, phone_lengths, pitchid, dur, slur, spk_id=spkid)
      spec = spectrogram_torch(
            wav.squeeze(1), 
            hps.data.n_fft, 
            hps.data.sample_rate, 
            hps.data.hop_size, 
            hps.data.win_size
        )

      y_mel = mel_spectrogram_torch(
          wav.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:],
      "gen/harm": y_harm[0,:,:],
      "gen/noise": y_noise[0,:,:]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": wav[0,:,:wav_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sample_rate
    )
    generator.train()

if __name__ == "__main__":
  main()
