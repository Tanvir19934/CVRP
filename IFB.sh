#!/bin/zsh

# Executes the following programs to find an IFB using the heuristic 
# and solves the problem optimally

python3 config.py
python3 hsc_ALNS_IFB.py
python3 hsc_MIP_start.py