#!/bin/sh
#SBATCH --job-name preprocess_pile
#SBATCH --ntasks-per-node 8
#SBATCH --partition [partition]
#SBATCH --account [account]
#SBATCH --qos [qos]
#SBATCH --mem 80G
#SBATCH --time 10:00:00
#SBATCH --output logs/preprocess_pile_%J.log


source /userfiles/[USER]/conda/bin/activate [ENV]

split=$1
straw-process-pile \
    --input-jsonl [INPUT]/$split.jsonl \
    --output-jsonl [OUT]/$split.jsonl \
    --nworkers 16 \
    --read-chunk-size 16 \
    --total-docs 10000000 \
    --min-text-length 15000 \
    --chunk-text True \
    --text-chunk-size 1024 \
    --pile-subsets "Books3,Gutenberg (PG-19),OpenWebText2,Pile-CC,Wikipedia (en)"
