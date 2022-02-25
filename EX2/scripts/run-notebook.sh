#!/bin/bash -i
cd "$(dirname "$0")/.."
conda activate wis-cv
jupyter notebook ex2-pyramids.ipynb
