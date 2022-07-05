# EDU-AP: Elementary Discourse Unit based Argument Parser
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of the paper:

## [**EDU-AP: Elementary Discourse Unit based Argument Parser**]
[**Sougata Saha**](https://www.linkedin.com/in/sougata-saha-8964149a/), [**Souvik Das**](https://www.linkedin.com/in/souvikdas23/), [**Rohini Srihari**](https://www.acsu.buffalo.edu/~rohini/) 

The 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2022)

## Abstract
Neural approaches to end-to-end argument mining (AM) are often formulated as dependency parsing (DP), which relies on token-level sequence labeling and intricate post-processing for extracting argumentative structures from text. Although such methods yield reasonable results, operating solely with tokens increases the possibility of discontinuous and overly segmented structures due to minor inconsistencies in token level predictions. In this paper, we propose EDU-AP, an end-to-end argument parser, that alleviates such problems in dependency-based methods by exploiting the intrinsic relationship between elementary discourse units (EDUs) and argumentative discourse units (ADUs) and operates at both token and EDU level granularity. Further, appropriately using contextual information, along with optimizing a novel objective function during training, EDU-AP achieves significant improvements across all four tasks of AM compared to existing dependency-based methods.

### Dataset Used:
Persuasive Essays (PE) Corpus: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

### Model for EDU Segmentation:
Bi LSTM-CRF based discourse segmenter (NeuralEDUSeg) by Wang et al. (2018): https://github.com/PKU-TANGENT/NeuralEDUSeg

### Training and Inference
You can train and evaluate all experiments using the `runner.sh` script. Example: `nohup bash runner.sh 1 12 > log.txt 2>&1 &` runs experiment numbers 1 to 12 sequentially. All the different configurations for the experiments can be found in the `experiment_config.json` file.
In order to experiment with different parameters, you can directly execute the `run_training.py` script. Sample command below:

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 9999 ./run_training.py --batch_size 16 --num_epochs 15 --learning_rate 0.00002 --base_transformer "roberta-base"
```
Prior to training, please download the PE dataset into a folder named `./data/`, and run NeuralEDUSeg on the data to generate the EDUs.


