#!/bin/zsh
# Full inference-path eval (code/eval.py, beam 10, all 1001 val rows) on Modal L4.
cd "$(dirname "$0")" || exit 1
caffeinate -w $$ &
../gpt-env/bin/modal run code/modal_app.py --eval-full-run --beam 10 "$@"
