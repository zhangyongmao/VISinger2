# VISinger2

This repository is the official PyTorch implementation of [VISinger2](https://arxiv.org/abs/2211.02903).

### Updates
- Apr 10 2023: Add egs/visinger2_flow: add flow to VISinger2 to get a more flexible prior distribution.
- Jan 31 2023: Modify the extraction method of gt-dur in dataset.py. Replace the dsp-wav with a sinusoidal signal as input to the HiFi-GAN decoder.
- Jan 10 2023: Init commit.

## Pre-requisites
1. Install python requirements: pip install -r requirements.txt
2. Download the [Opencpop Dataset](https://wenet.org.cn/opencpop/).
3. prepare data like data/opencpop (wavs, trainset.txt, testset.txt, train.list, test.list)
4. modify the egs/visinger2/config.json (data/data_dir, train/save_dir)

## extract pitch and mel
```
cd egs/visinger2
bash bash/preprocess.sh config.json
```

## Training
```
cd egs/visinger2
bash bash/train.sh 0
```
We trained the model for 500k steps with batch size of 16.

## Inference
modify the model_dir, input_dir, output_dir in inference.sh
```
cd egs/visinger2
bash bash/inference.sh
```

Some audio samples can be found in [demo website](https://zhangyongmao.github.io/VISinger2/) and [bilibili](https://www.bilibili.com/video/BV1wX4y167rb/?share_source=copy_web&vd_source=4e678224f5616d7af7dfaf2401b5d574).

The pre-trained model trained using opencpop is [here](https://drive.google.com/file/d/1MgXLQuquPT2qu1__JNF010-tg48N0hZn/view?usp=share_link), the config.json is [here](https://drive.google.com/file/d/10GI9OUtE4fQ8om8MvycDYQpcP6lgHLNZ/view?usp=share_link), and the result of the test set synthesized by this pre-trained model is [here](https://drive.google.com/file/d/1JTMhtkexo5z3q0bpLoqh4EJmx1HjZyMr/view?usp=share_link).

## Acknowledgements
We referred to [VITS](https://github.com/jaywalnut310/vits), [HiFiGAN](https://github.com/jik876/hifi-gan), [gst-tacotron](https://github.com/syang1993/gst-tacotron)
and [ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch) to implement this. Thanks to [swagger-coder](https://github.com/swagger-coder) for help building visinger2_flow.

