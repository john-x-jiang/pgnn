#!/bin/bash

CUDA=cu116

pip install -U torch --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install -U torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html

pip install -r requirements.txt
