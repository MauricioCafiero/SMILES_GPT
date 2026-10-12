#!/bin/zsh
# 50-epoch retrainer with volume persistence (commit after every ckpt save).
cd "$(dirname "$0")" || exit 1
caffeinate -w $$ &
../gpt-env/bin/modal run code/modal_app.py --epochs 50 --beam 10 --eval-rows 1001