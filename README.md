# Straw: A preprocessing and filtering tool for the [Pile](https://pile.eleuther.ai/)

## Installation

```
pip install -e .
```

## Usage

First, download and uncompress the validation subset of the pile dataset [here](https://pile.eleuther.ai/). 

You can run the processing pipeline with the following command:

```bash
straw-process-pile \
    --input-jsonl pile-set/val.jsonl \
    --output-jsonl pile-set/val-processed.jsonl \
    --nworkers 16 \
    --read-chunk-size 16 \
    --total-docs 250000 \
    --min-text-length 15000 \
    --chunk-text True \
    --text-chunk-size 1024 \
    --pile-subsets "Books3,Gutenberg (PG-19),OpenWebText2,Pile-CC,Wikipedia (en)"
```
