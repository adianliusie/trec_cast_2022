#!/bin/bash

# activate environment
source /home/al826/rds/hpc-work/2022/trec/trec_cast_2022/test/bin/activate

#load any enviornment variables needed
source ~/.bashrc

TOKENIZERS_PARALLELISM=false

python $@
