This repo contains the code for the [NAACL 2021 paper](https://aclanthology.org/2021.naacl-main.225/)
*Compositional Generalization for Neural Semantic Parsing via Span-level Supervised Attention* 
by Microsoft Semantic Machines.

## Download Dataset

First, download and unzip data files:

```
wget https://www.cs.cmu.edu/~pengchey/reg_attn_data.zip
unzip reg_attn_data.zip
```

## Running Models

### CalFlow-Compositional Skills (CalFlow-CS) Dataset

**Python Environments** Our code is based on `allennlp`. You may use the `conda`
environment in `data/calflow_cs/env.yml`. Note that the current codebase is based 
on a customized version of `allennlp` to support `wandb` logging and more fine-grained
training control. We plan to migrate the code to the official `allennlp` in the future.

To install the conda environment, run

```shell script
conda env create -n compositional_generalization -f data/env.yml
``` 

Our code is based on `allennlp`. The `BERT2SEQ` model is implemented in 
`models/seq2seq_with_copy.py`. The structured `Coarse-to-Fine` parser is located
in `models/decompositional_parser.py` 

**BERT2SEQ Models** The following command will run training and evaluation of 
the `BERT2SEQ` model on the CalFlow-CS dataset

```shell script
allennlp train \
  data/configs/config.calflow.seq2seq.bert.jsonnet \
  -s data/runs/tmp_run  \ # checkpoint saving path
  --include-package=models
```

Please refer to the configuration file (`*.jsonnet`) for details, and modify 
the `train_data_path` appropriately for training on other splits.

**Structured Parser**

```shell script
allennlp train \
  data/configs/config.calflow.structured.bert.jsonnet \
  -s data/runs/tmp_run  \ # checkpoint saving path
  --include-package=models
```

**Run All Experiments** Use the script `utils.calflow.run_experiments` to run experiments 
programtically. For example, the following command will run experiments for `Bert2Seq` and `Structured`
parsers using the training set with 16 compositional samples (`C_train=16`) with 5 random restarts:

```shell script
python -m utils.calflow.run_experiments \
  seq2seq.bert,structured.bert \
  --domain calflow.orgchart.event_create \
  --dataset-prefix data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num16 \
  --seeds 0,1,2,3,4 \
  --no-cuda  # remove this argument for GPU training
```

### CFQ (Compositional Freebase Questions) Dataset

We use the huggingface `transformers` library at commit id `df1ddced`. 
To checkout and install the library under `attn_reg_transformers/`, run

```shell script
git clone https://github.com/huggingface/transformers.git my_transformers
cd my_transformers
git checkout df1ddced
# modify the T5 model to expose cross-attention tensors in model outputs
git apply ../data/cfq/transformers_t5.patch
# make sure existing libraries in the conda environments are not over-writen 
pip install --no-cache-dir --upgrade-strategy only-if-needed -e .
```

Training code is located under `reg_attn_transformers`, which is based on
 `pytorch-lightening`. To use the code, install the requirements:
 
```shell script
cd reg_attn_transformers
pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt
```

The following command demonstrates training a `T5-base` model using 
span-level supervised attention with a regularization weight of `0.1` 

```shell script
python attention_regularized_seq2seq.py \
  --data_dir data/cfq/mcd2 \
  --output_dir data/runs/tmp_run \
  --model_name_or_path t5-base \
  --seed 0 \
  --n_val 1000 \
  --shuffle_val \
  --n_train -1 \
  --n_test -1 \
  --check_val_every_n_epoch 1 \
  --adafactor \
  --learning_rate 0.001 \
  --num_train_epochs 15 \
  --train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --eval_batch_size 20 \
  --max_source_length 200 \
  --max_target_length 300 \
  --val_max_target_length 300 \
  --eval_max_gen_length 300 \
  --test_max_target_length 300 \
  --warmup_steps 1100 \
  --gpus 1 \
  --val_metric acc \
  --early_stopping_patience 10 \
  --num_workers 4 \
  --do_train \
  --do_predict \
  --attention_regularization mse:all:src_normalize:11:0.1
```

### ATIS text-to-SQL Dataset

Our code (under `text2sql_experiments`) is based on https://github.com/inbaroren/improving-compgen-in-semparse. 
First, install the repo and following the official README 
to create a conda environment for `allennlp` v0.9 (you might need `pip install overrides==3.1.0` to make it work).

The following script will generate run sweep for our experiments. Note that the script will NOT 
run those experiments, just generating the `allennlp` command lines.

```shell script
# under /text2sql_experiments
pip install python-slugify
PYTHONPATH=. python scripts/create_commands.py
```

## Generate Span-level Alignment Datasets

We provide span-level alignment datasets for `CalFlow-CS`, `CFQ` and 
`ATIS text-to-sql` under `data/`. This section documents how to reproduce 
those datasets. 

### Calflow-CS Dataset

**Step 1: Filter single-turn utterances** Run the following command to 
filter single-turn context-independent utterances in `EventCreation` 
and `OrgChart` domains from the official SM-CalFlow dataset: 

```shell script
python -m utils.calflow.preprocess_dataset \
  --data-path /path/to/smcalflow.full.data/folder \
  --output-path data/smcalflow_cs \
  --seed 1
```

**Step 2: Train GIZA alignments**

```shell script
python -m utils.calflow.train_giza_alignment \
  data/smcalflow_cs/calflow.orgchart.event_create
```

**Step 3: Generate span-level alignments**

```shell script
python -m utils.calflow.extract_span_alignment_dataset
```

### CFQ (Compositional Freebase Questions) Dataset

**Step 1: Download Dataset** Download the CFQ dataset from 
https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz and extract 
`dataset.json` to `data/cfq`, our script will use `dataset.json` 
 to retrieve the metadata information for each example.

**Step 2: Get MCD Splits** Run the following script in the repo
https://github.com/google-research/google-research/tree/master/cfq  
to generate CFQ 
`MCD{n=1,2,3}` splits: 

```shell script
python -m preprocess_main --dataset=cfq \
  --split={mcd1,mcd2,mcd3} --save_path=data/cfq/{mcd1,mcd2,mcd3}
```

**Step 3: Run GIZA word alignments**
```shell script
python -m utils.cfq.train_giza_alignment
```

**Step 4: Extract Span-level Alignments**
```shell script
python -m utils.cfq.extract_span_alignment_dataset \
  data/cfq/mcd{1,2,3} \
  --dataset data/cfq/dataset.json
```

This will create dataset folders such as `data/cfq/mcd1_simplified_mid_rel_sw_fix_rel_after_obj`

### ATIS text-to-SQL Dataset

**Step 1: Download Dataset** Download data from https://github.com/inbaroren/improving-compgen-in-semparse/tree/main/data/sql%20data

**Step 2: Run GIZA word alignments**
```shell script
python -m utils.sql.train_giza_alignment
``` 

**Step 3: Extract Span-level Alignments**
```shell script
pythom -m utils.sql.extract_span_alignment_dataset
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Reference

```
@inproceedings{yin21naacl,
    title = {Compositional Generalization for Neural Semantic Parsing via Span-level Supervised Attention},
    author = {Pengcheng Yin and Hao Fang and Graham Neubig and Adam Pauls and Emmanouil Antonios Platanios and Yu Su and Sam Thomson and Jacob Andreas},
    booktitle = {Meeting of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    address = {Mexico City},
    month = {June},
    url = {https://www.aclweb.org/anthology/2021.naacl-main.225/},
    year = {2021}
}
```
