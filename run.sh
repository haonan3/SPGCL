#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset chameleon --neg_selection random \
--load_params 1 --log_type txt --log_note test --save_folder logs &
sleep 10