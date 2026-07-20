#!/bin/bash

module load conda
conda init
conda activate tf212gpu_new

python plots.py