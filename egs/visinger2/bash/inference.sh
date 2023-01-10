
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

CUDA_VISIBLE_DEVICES=0 python inference.py \
    -model_dir /root/logdir/visinger2 \
    -input_dir data/opencpop/testset.txt \
    -output_dir /root/logdir/visinger2/syn_result \

