# VARAN 🦎: Official Implematration of Variational Inference for Self-Supervised Speech Models Fine-Tuning on Downstream Tasks
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2508.12061)
## How to run

This repository contains a source code for SER: Speech Emotion Recognition and ASR: Automatic Speech Recognition tasks. 

To train SER model run: `python scripts/train_ser.py <path_to_json_config>`. Metrics for the validation and test sets will be computed at each evaluation step specified in the config. 

For the ASR task, we do not share a training script, but we do share the model classes implementing the VARN layer aggregation method for CTC loss computation in the [data2vec_for_sequence_classification.py](src/models/data2vec_for_ctc.py) and [wavlm_for_sequence_classification.py](src/models/wavlm_for_sequence_classification) scripts.

### Available model configurations for SER:

1. Upstream model:
    - data2vec-base
    - data2vec-large
    - wavlm-base
    - wavlm-large
2. Fine-tuning strategy:
    - full fine-tuning 
    - LoRA  fine-tuning 
3. Layer aggreagtion method:
    - last layer
    - weighted sum
    - varan

Examples of the json configs can be found in [configs](configs/ser/ravdess) directory. 


## Reproduciability

While we don't provide pre-trained model weights, this code contains everything needed to reproduce our approach for academic research purposes.

### To reproduce our results:

1. We recommend running a hyperparameter sweep for each layer aggregation method and selecting the best configuration based on validation set performance.
2. The hyperparameters we used in our experiments are detailed in the paper.
3. Due to the environmental differences may affect results, so your final metrics might differ from those reported.

An example sweep configuration can be found in [varan_sweep.yaml](varan_sweep.yaml).

