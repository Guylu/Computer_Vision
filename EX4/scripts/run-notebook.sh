#!/bin/bash -i
cd "$(dirname "$0")/.."
conda activate wis-cv
jupyter notebook --no-browser ex4-alignment.ipynb
