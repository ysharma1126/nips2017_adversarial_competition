#!/bin/bash
#
# run_eval.sh is a script which evaluates an attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_eval.sh INPUT_DIR METADATA_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
METADATA_FILE=$2

pip install opencv_python-3.2.0.8-cp27-cp27mu-manylinux1_x86_64.whl

python eval.py \
  --input_dir="${INPUT_DIR}" \
  --csv_file="${METADATA_FILE}" \
  --checkpoint_path=model_ckpts
