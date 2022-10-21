# Sentence-Incremental Neural Coreference Resolution

This repository contains software for the paper: Sentence-Incremental Neural Coreference Resolution, presented at EMNLP 2022.

Our implementation is based off Liyan Xu's code for 
[Revealing the Myth of Higher-Order Inference in Coreference Resolution](https://github.com/lxucs/coref-hoi/).
It has been tested for Python 3.6 but should also work with newer versions.

## Citation
```
@inproceedings{grenander-etal-2022-sentence,
 title = "Sentence-Incremental Neural Coreference Resolution",
 author = {Grenander, Matt and Cohen, Shay B. and Steedman, Mark},
 booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
 month = dec,
 year = "2022",
 address = "Abu Dhabi, United Arab Emirates and Virtual",
 publisher = "Association for Computational Lingustics",
}
```


## File Structure
**Files:**
* [run.py](run.py): main script for training and evaluation
* [model.py](model.py): Base coreference model and Part-Incremental model
* [sentence_incremental.py](sentence_incremental.py): Sentence-Incremental model
* [evaluate.py](evaluate.py): standalone evaluation script
* [predict.py](predict.py): script for prediction on custom input
* [setup_data.sh](setup_data.sh): converting raw OntoNotes files to CoNLL files, downloading official evaluation scripts
* [preprocess.py](preprocess.py): converting OntoNotes' CoNLL files to jsonlines files
* [tensorize.py](tensorize.py): convert jsonlines files to tensors
* [conll.py](conll.py): helper script for official evaluation
* [metrics.py](metrics.py): unofficial scoring scripts
* [pytorch_utils.py](pytorch_utils.py): helper functions for creating FFNNs
* [stack_lstm.py](stack_lstm.py): stack LSTM implementation
* [util.py](util.py): other helper functions
* [experiments.conf](experiments.conf): different model configurations

## Data Setup
* Install python packages: `pip install -r requirements.txt`
* Create a directory to hold data (`<data_dir>`)
### OntoNotes
* Download the [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19) corpus.
* Run (**using Python 2**):
```
./setup_data.sh <ontonotes_dir> <data_dir>
```
which will create `*.english.v4_gold_conll` files.
* Run (with `gdown` [installed first](https://pypi.org/project/gdown/)):
```
./setup_official_scorer.sh 
```
which downloads the official evaluation scripts into the `conll-2012` directory.
* Run the preprocess script, which creates `*.jsonlines` files: 
  * Part-Incremental setting: 
```
python preprocess --input_dir <data_dir> --output_dir <data_dir> --seg_len 512
```
  * Sentence-Incremental setting:
```
python preprocess --input_dir <data_dir> --output_dir <data_dir> --split_sents
```

### CODI-CRAC
* Download the CODI-CRAC corpus:
  * AMI, Light and Persuasion can be downloaded from [Codalab](https://competitions.codalab.org/competitions/30312#participate-get-data) (you will need to create an account). 
  * Switchboard must be acquired from LDC (corpus ID: LDC2021E05).
* Use our [fork of the conversion script](https://github.com/mgrenander/codi2021_scripts) to convert the data to jsonlines with the following commands:
```
import helper

# Part-Incremental setup
helper.convert_coref_ua_to_json(<ami_path>, <jsonlines_path>, SEGMENT_SIZE=512, TOKENIZER_NAME='xlnet-base-cased', sentences=False)
helper.convert_coref_ua_to_json(<light_path>, <jsonlines_path>, SEGMENT_SIZE=512, TOKENIZER_NAME='xlnet-base-cased', sentences=False)
# Etc.

# Sentence-Incremental setup
helper.convert_coref_ua_to_json(<ami_path>, <jsonlines_path>, SEGMENT_SIZE=512, TOKENIZER_NAME='xlnet-base-cased', sentences=True)
helper.convert_coref_ua_to_json(<light_path>, <jsonlines_path>, SEGMENT_SIZE=512, TOKENIZER_NAME='xlnet-base-cased', sentences=True)
# Etc.
```
  

## Training
Pick a configuration from `experiments.conf` and supply it to `run.py`, e.g.:

```
python run.py --config_name train_xlnet_part_inc --gpu 0

python run.py --config_name train_xlnet_sent_inc --gpu 0
```

If training for the first time, `run.py` will call `tensorize.py` to serialize the dataset and cache the dataset.
The Huggingface library should automatically download and cache XLNet.

The training scripts will create a directory `<data_dir>/<config_name>` and place relevant files in it:
- Logging files: `log_XXX.txt`
- Model checkpoints: `model_XXX.bin`

Tensorboard files are available under `<data_dir>/tensorboard`.

## Evaluation
We provide trained models [here](https://drive.google.com/drive/folders/1nDHs80QXKRT7C_4DuIPEhFSDlyz8kkTJ?usp=sharing).


Unzip them under `<data_dir>` and run `evaluate.py` using the template: 
```
python evaluate.py <config_name> <model_checkpoint> <gpu_id>
```

Some sample evaluation commands:
```
# OntoNotes
python evaluate.py train_xlnet_part_inc Mar15_03-50-14_61000 0

python evaluate.py train_xlnet_sent_inc Feb05_12-38-09_61000 0

# CODI-CRAC
python evaluate.py xlnet_ami_part_inc Mar15_03-50-14_61000 0

python evaluate.py xlnet_ami_sent_inc Feb05_12-38-09_61000 0

# Etc.
```

## Predict

Capture model output and write the predictions to a `jsonlines` file for the OntoNotes dev or test set. Usage:

```
predict.py --config_name CONFIG_NAME --model_identifier MODEL_IDENTIFIER 
           --output_path OUTPUT_PATH [--gpu_id GPU_ID]
           --input_path INPUT_PATH [--eval_data EVAL_DATA]

Arguments:
  --config_name CONFIG_NAME
                        Configuration name in experiments.conf
  --model_identifier MODEL_IDENTIFIER
                        Model identifier to load
  --output_path OUTPUT_PATH
                        Path to save output
  --gpu_id GPU_ID       GPU id; CPU by default
  --input_path INPUT_PATH
                        Path to jsonlines file
  --eval_data EVAL_DATA
                        Either 'dev' or 'test'
```