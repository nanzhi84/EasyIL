#!/usr/bin/env bash
set -e

# Evaluate a trained diffusion policy model
# All parameters are configured in easyil/conf/eval.yaml

python -m easyil.eval
