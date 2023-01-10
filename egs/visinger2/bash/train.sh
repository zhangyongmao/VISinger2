
num_gpu=$1

cd $(dirname $(dirname $0))
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))
config=${exp_dir}/config.json

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

CUDA_VISIBLE_DEVICES=${num_gpu} python train.py -c config.json

