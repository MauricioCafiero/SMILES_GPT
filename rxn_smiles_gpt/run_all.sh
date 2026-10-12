#!/bin/zsh
# USPTO-50k retro pipeline: build (cached) -> train -> eval.
# Launched detached via the Popen(start_new_session=True) pattern; log in runs/.
cd "$(dirname "$0")" || exit 1
caffeinate -w $$ &   # hold off macOS idle sleep for the whole run; dies with the script
PY=../gpt-env/bin/python

if [ ! -f data/retro_train.npz ]; then
  echo "=== build $(date) ==="
  $PY code/retro_pipeline.py build || exit 1
fi
if [ ! -f runs/retro_gpt.pt ]; then
  echo "=== train $(date) ==="
  $PY code/retro_pipeline.py train || exit 1
fi
echo "=== eval $(date) ==="
$PY code/retro_pipeline.py eval
echo "=== done $(date) ==="