# Synthesis_Step-by-Step_Official

This repository contains the datasets and codes for our paper "Let's Synthesize Step by Step:
Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models"

![](imges/MainPlot.jpg)

## Datasets

If you just want to explore our dataset. We provide a simple json version of the dataset ``./Datasets``. The format for each 
task is as follows:

1. **IMDb:** The dataset is formatted as: [[Review text], [label]]
2. **QNLI:** The dataset is in format: [[premises], [questions], [labels]]
3. **RTE:** The dataset has format: [[premises], [hypothesis], [label]]
4. **AdQA:** The dataset is formatted as: [[contexts], [questions], [answers], [start_idx], [end_idx], [id]]

## Click and Run Inference using our Dataset

We have created a simple click-and-run example for how to use our generated dataset. The code are in ``./ModelTraining``.
In order to run our code, you first need to install environment from ``./ModelTraining/environment.yml``.
For different task, the main file is organized as ``./ModelTraining/main_<task_name>.py``. It can auto store the misclassified data 
into the folder. You can play with our generated dataset by running corresponding main file.

## Data Generation
Due to the generation code is a bit complicated, please give us a little more time to finish the code cleaning. It will be coming soon

## Acknowledgement
We thank the anonymous reviewers for their feedback on our paper.
MS acknowledges support from the Swiss National Science Foundation (Project No. 197155), 
a Responsible AI grant by the Haslerstiftung; and an ETH Grant (ETH-19 21-1).

## Citation information

If you use this code, please cite our paper:

```
@inproceedings{wang2023let,
  title={Let’s Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models},
  author={Wang, Ruida and Zhou, Wangchunshu and Sachan, Mrinmaya},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={11817--11831},
  year={2023}
}
```