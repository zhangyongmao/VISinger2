#! /bin/bash

config=$1
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

cd ${base_dir}

echo "prepare mel pitch "
python "${base_dir}"/preprocess/preprocess.py --config=${exp_dir}/${config}


