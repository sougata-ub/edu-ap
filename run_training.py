import torch
import argparse
from trainer import Trainer
from inference import Inference
import time
from tqdm import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import os.path
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


def train_baseline(N_EPOCHS, trainer, best_valid_loss, model_name, early_stopping=2):
    t_loss, v_loss = [], []
    t_loss_head, v_loss_head = [], []
    t_loss_deprel, v_loss_deprel = [], []
    best_epoch = 0
    early_stopping_marker = []

    for epoch in range(N_EPOCHS):
        print("Epoch: {}, Training ...".format(epoch))
        start_time = time.time()

        train_dataloader = trainer.get_baseline_dataset(trainer.train_length_dict)
        tr_l, tr_l_head, tr_l_deprel = trainer.train_baseline(train_dataloader)
        t_loss.append(tr_l)
        t_loss_head.append(tr_l_head)
        t_loss_deprel.append(tr_l_deprel)

        test_dataloader = trainer.get_baseline_dataset(trainer.test_length_dict, typ="test")
        vl_l, vl_l_head, vl_l_deprel = trainer.evaluate_baseline(test_dataloader)
        v_loss.append(vl_l)
        v_loss_head.append(vl_l_head)
        v_loss_deprel.append(vl_l_deprel)

        end_time = time.time()
        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        if vl_l < best_valid_loss:
            best_valid_loss = vl_l
            print("SAVING BEST MODEL!")
            torch.save(trainer.parser.state_dict(), model_name)
            best_epoch = epoch
            early_stopping_marker.append(False)
        else:
            early_stopping_marker.append(True)

        print("\n")
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Total Loss: {tr_l:.3f}')
        print(
            f'\tTr Edge Pred Loss: {tr_l_head:.3f} | Tr Edge Lbl Loss: {tr_l_deprel:.3f}')
        print(f'\tVal. Total Loss: {vl_l:.3f}')
        print(
            f'\tVl Edge Pred Loss: {vl_l_head:.3f} | Vl Edge Lbl Loss: {vl_l_deprel:.3f}')
        print("_________________________________________________________________")

        if all(early_stopping_marker[-early_stopping:]):
            print("Early stopping training as the Validation loss did NOT improve for last " + str(early_stopping) +\
                  " iterations.")
            break

    return t_loss, v_loss, t_loss_head, v_loss_head, t_loss_deprel, v_loss_deprel, best_valid_loss, best_epoch


def train_regular(N_EPOCHS, trainer, best_valid_loss, model_name):
    t_loss, v_loss = [], []
    t_loss_head, v_loss_head = [], []
    t_loss_deprel, v_loss_deprel = [], []
    t_loss_span, v_loss_span = [], []
    t_loss_typ, v_loss_typ = [], []
    best_epoch = 0

    for epoch in range(N_EPOCHS):
        print("Epoch: {}, Training ...".format(epoch))
        start_time = time.time()

        train_dataloader = trainer.get_batch_data(trainer.train_length_dict)
        tr_l, tr_l_head, tr_l_deprel, tr_l_span, tr_l_typ = trainer.train(train_dataloader)
        t_loss.append(tr_l)
        t_loss_head.append(tr_l_head)
        t_loss_deprel.append(tr_l_deprel)
        t_loss_span.append(tr_l_span)
        t_loss_typ.append(tr_l_typ)

        test_dataloader = trainer.get_batch_data(trainer.test_length_dict)
        vl_l, vl_l_head, vl_l_deprel, vl_l_span, vl_l_typ = trainer.evaluate(test_dataloader)
        v_loss.append(vl_l)
        v_loss_head.append(vl_l_head)
        v_loss_deprel.append(vl_l_deprel)
        v_loss_span.append(vl_l_span)
        v_loss_typ.append(vl_l_typ)

        end_time = time.time()
        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        if vl_l < best_valid_loss:
            best_valid_loss = vl_l
            print("SAVING BEST MODEL!")
            torch.save(trainer.parser.state_dict(), model_name)
            best_epoch = epoch

        print("\n")
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Total Loss: {tr_l:.3f}')
        print(
            f'\tTr Edge Pred Loss: {tr_l_head:.3f} | Tr Edge Lbl Loss: {tr_l_deprel:.3f} | Tr Span Loss: {tr_l_span:.3f} | Tr Type Loss: {tr_l_typ:.3f}')
        print(f'\tVal. Total Loss: {vl_l:.3f}')
        print(
            f'\tVl Edge Pred Loss: {vl_l_head:.3f} | Vl Edge Lbl Loss: {vl_l_deprel:.3f} | Vl Span Loss: {vl_l_span:.3f} | Vl Type Loss: {vl_l_typ:.3f}')
        print("_________________________________________________________________")

    return t_loss, v_loss, t_loss_head, v_loss_head, t_loss_deprel, v_loss_deprel, t_loss_span, v_loss_span, \
           t_loss_typ, v_loss_typ, best_valid_loss, best_epoch


def train_regular_single(N_EPOCHS, trainer, best_valid_loss, model_name, early_stopping=3):
    t_loss, v_loss = [], []
    t_loss_head, v_loss_head = [], []
    t_loss_deprel, v_loss_deprel = [], []
    t_loss_span, v_loss_span = [], []
    t_loss_typ, v_loss_typ = [], []
    best_epoch = 0
    early_stopping_marker = []

    for epoch in range(N_EPOCHS):
        print("Epoch: {}, Training ...".format(epoch))
        start_time = time.time()

        train_dataloader = trainer.get_single_data(trainer.train_length_dict)
        tr_l, tr_l_head, tr_l_deprel, tr_l_span, tr_l_typ = trainer.train_single(train_dataloader)
        t_loss.append(tr_l)
        t_loss_head.append(tr_l_head)
        t_loss_deprel.append(tr_l_deprel)
        t_loss_span.append(tr_l_span)
        t_loss_typ.append(tr_l_typ)

        test_dataloader = trainer.get_single_data(trainer.test_length_dict, typ="test")
        vl_l, vl_l_head, vl_l_deprel, vl_l_span, vl_l_typ = trainer.evaluate_single(test_dataloader)
        v_loss.append(vl_l)
        v_loss_head.append(vl_l_head)
        v_loss_deprel.append(vl_l_deprel)
        v_loss_span.append(vl_l_span)
        v_loss_typ.append(vl_l_typ)

        end_time = time.time()
        epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

        if vl_l < best_valid_loss:
            best_valid_loss = vl_l
            print("SAVING BEST MODEL!")
            torch.save(trainer.parser.state_dict(), model_name)
            best_epoch = epoch
            early_stopping_marker.append(False)
        else:
            early_stopping_marker.append(True)

        print("\n")
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Total Loss: {tr_l:.3f}')
        print(
            f'\tTr Edge Pred Loss: {tr_l_head:.3f} | Tr Edge Lbl Loss: {tr_l_deprel:.3f} | Tr Span Loss: {tr_l_span:.3f} | Tr Type Loss: {tr_l_typ:.3f}')
        print(f'\tVal. Total Loss: {vl_l:.3f}')
        print(
            f'\tVl Edge Pred Loss: {vl_l_head:.3f} | Vl Edge Lbl Loss: {vl_l_deprel:.3f} | Vl Span Loss: {vl_l_span:.3f} | Vl Type Loss: {vl_l_typ:.3f}')
        print("_________________________________________________________________")

        if all(early_stopping_marker[-early_stopping:]):
            print("Early stopping training as the Validation loss did NOT improve for last " + str(early_stopping) +\
                  " iterations.")
            break

    return t_loss, v_loss, t_loss_head, v_loss_head, t_loss_deprel, v_loss_deprel, t_loss_span, v_loss_span, \
           t_loss_typ, v_loss_typ, best_valid_loss, best_epoch


def main():
    n_layers = 4
    base_transformer = "microsoft/deberta-base"
    N_EPOCHS = 15
    batch_size = 16
    lr = 1e-5
    mname = base_transformer+"_parser.pt"
    device_num = 0
    training_file = "<Path to training file>"
    out_dim = 256
    model_type = "general"  # "baseline"
    skip_training = "False"
    add_convolution, sum_transpose, prompt_attention = "False", "False", "False"
    n_heads, n_attn_layers = 4, 2
    ann_path = None     # './data/ArgumentAnnotatedEssays-2.0 3/brat-project-final/'
    prompts_path = None     # './data/ArgumentAnnotatedEssays-2.0 3/prompts.csv'
    train_test_split_path = None    # "./data/ArgumentAnnotatedEssays-2.0 3/train-test-split.csv"
    edu_segmented_file = None   # <Path to training file>
    run_inference, create_training_dataset, add_additional_loss = "True", "True", "False"
    sigmoid_threshold, interpolation = 0.5, 0.05
    add_context = "none"
    experiment_number = "none"
    experiment_repeat = 1
    early_stopping = 3

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--base_transformer", type=str, help="Pretrained transformer.", default=base_transformer)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=lr)
    parser.add_argument("--model_name", type=str, help="Model Name.", default=mname)
    parser.add_argument("--training_file_name", type=str, help="File name cointaining examples", default=training_file)
    parser.add_argument("--device_num", type=int, help="CUDA device number", default=device_num)
    parser.add_argument("--out_dim", type=int, help="Output dim of Biaffine Layer", default=out_dim)
    parser.add_argument("--model_type", type=str, help="Type of model to train", default=model_type)
    parser.add_argument("--skip_training", type=str, help="Skip Training", default=skip_training)
    parser.add_argument("--add_convolution", type=str, help="Add Convolutions", default=add_convolution)
    parser.add_argument("--sum_transpose", type=str, help="Sum Transpose", default=sum_transpose)
    parser.add_argument("--prompt_attention", type=str, help="Prompt Attention", default=prompt_attention)
    parser.add_argument("--n_heads", type=int, help="# EDU self attention heads", default=n_heads)
    parser.add_argument("--n_attn_layers", type=int, help="# EDU self attention layers", default=n_attn_layers)
    parser.add_argument("--ann_path", type=str, help="annotations_path", default=ann_path)
    parser.add_argument("--prompts_path", type=str, help="prompts_path", default=prompts_path)
    parser.add_argument("--train_test_split_path", type=str, help="train_test_split_path", default=train_test_split_path)
    parser.add_argument("--edu_segmented_file", type=str, help="edu_segmented_file", default=edu_segmented_file)
    parser.add_argument("--create_training_dataset", type=str, help="create_training_dataset", default=create_training_dataset)
    parser.add_argument("--run_inference", type=str, help="run_inference", default=run_inference)
    parser.add_argument("--sigmoid_threshold", type=float, help="sigmoid_threshold", default=sigmoid_threshold)
    parser.add_argument("--interpolation", type=float, help="interpolation", default=interpolation)
    parser.add_argument("--add_additional_loss", type=str, help="add_additional_loss", default=add_additional_loss)
    parser.add_argument("--add_context", type=str, help="add_context", default=add_context)
    parser.add_argument("--experiment_number", type=str, help="experiment_number", default=experiment_number)
    parser.add_argument("--experiment_repeat", type=int, help="experiment_repeat", default=experiment_repeat)
    parser.add_argument("--early_stopping", type=int, help="early_stopping", default=early_stopping)

    argv = parser.parse_args()
    experiment_number = argv.experiment_number
    experiment_number = None if experiment_number == "none" else experiment_number
    base_transformer = argv.base_transformer
    N_EPOCHS = argv.num_epochs
    batch_size = argv.batch_size
    lr = argv.learning_rate
    mname = argv.model_name
    training_file = argv.training_file_name
    device_num = argv.device_num
    out_dim = argv.out_dim
    model_type = argv.model_type
    skip_training = argv.skip_training
    skip_training = True if skip_training == "True" else False
    add_convolution = argv.add_convolution
    add_convolution = True if add_convolution == "True" else False
    sum_transpose = argv.sum_transpose
    sum_transpose = True if sum_transpose == "True" else False
    prompt_attention = argv.prompt_attention
    prompt_attention = True if prompt_attention == "True" else False
    n_heads = argv.n_heads
    n_attn_layers = argv.n_attn_layers
    create_training_dataset = argv.create_training_dataset
    create_training_dataset = True if create_training_dataset == "True" else False
    run_inference = argv.run_inference
    run_inference = True if run_inference == "True" else False
    sigmoid_threshold = argv.sigmoid_threshold
    interpolation = argv.interpolation
    add_additional_loss = argv.add_additional_loss
    add_additional_loss = True if add_additional_loss == "True" else False
    add_context = argv.add_context
    experiment_repeat = argv.experiment_repeat
    early_stopping = argv.early_stopping

    ann_path = argv.ann_path
    prompts_path = argv.prompts_path
    train_test_split_path = argv.train_test_split_path
    edu_segmented_file = argv.edu_segmented_file
    path_prefix = "~/edu-ap/"
    run_time = str(int(time.time()))
    if not os.path.isdir(run_time):
        print("Creating New Directory: ",run_time)
        os.mkdir(run_time)
        model_dest = path_prefix+run_time+"/models/"
        data_dest = path_prefix + run_time + "/data/"
        result_dest = path_prefix + run_time + "/results/"
        os.mkdir(model_dest)
        os.mkdir(data_dest)
        os.mkdir(result_dest)
    else:
        model_dest = path_prefix + run_time + "/models/"
        data_dest = path_prefix + run_time + "/data/"
        result_dest = path_prefix + run_time + "/results/"

    for exp_repeat in tqdm(range(experiment_repeat)):
        print("Running Experiment Repeat", exp_repeat)
        best_valid_loss = float('inf')

        if experiment_number is not None:
            config_dict = json.load(open("./experiment_config.json", "r"))
            base_transformer = config_dict[experiment_number]["model"]
            add_additional_loss = config_dict[experiment_number]["loss_penalty"]
            add_context = config_dict[experiment_number]["context"]

            out_dim = 600
            batch_size = 8
            training_file = data_dest + "PE_essays_encoded_edu_segments_" + base_transformer + ".pkl"
            n_heads = 4
            n_attn_layers = 2
            ann_path = "./data/ArgumentAnnotatedEssays-2.0 3/brat-project-final/"
            prompts_path = "./data/ArgumentAnnotatedEssays-2.0 3/prompts.csv"
            train_test_split_path = "./data/ArgumentAnnotatedEssays-2.0 3/train-test-split.csv"
            edu_segmented_file = "./data/<EDU segmented file name>"
            sigmoid_threshold = 0.5
            create_training_dataset = True if not os.path.exists(training_file) else False
            early_stopping = 2

            model_name = "experiment_"+str(experiment_number)+"_"+str(exp_repeat)+".pt"
            log_name = "experiment_"+str(experiment_number)+"_"+str(exp_repeat)+"_LOG.json"
        else:
            model_name = mname.replace(".pt", "_"+str(exp_repeat)+".pt")
            log_name = model_name.replace("models", "results") + "_LOG.json"
            training_file = "./data/" + training_file if "./data/" not in training_file else training_file

        model_name = model_dest + model_name  # "./models/" + model_name
        # training_file = "./data/" + training_file
        log_file = result_dest + log_name  # model_name.replace("models", "results") + "_LOG.json"

        params = {"model_type": model_type, "base_transformer": base_transformer, "n_layers": n_layers,
                  "out_dim": out_dim, "lr": lr, "batch_size": batch_size, "training_file": training_file,
                  "add_convolution": add_convolution, "sum_transpose": sum_transpose,
                  "prompt_attention": prompt_attention, "n_heads": n_heads, "n_attn_layers": n_attn_layers,
                  "ann_path": ann_path, "prompts_path": prompts_path, "train_test_split_path": train_test_split_path,
                  "edu_segmented_file": edu_segmented_file, "sigmoid_threshold": sigmoid_threshold,
                  "create_training_dataset": create_training_dataset, "interpolation": interpolation,
                  "add_additional_loss": add_additional_loss, "add_context": add_context,
                  "early_stopping": early_stopping}

        print("\n:::::EXECUTION PARAMETERS:::::\n")
        print(params, "\n")
        execution_log = {"parameters": params, "training": {"losses": None}, "inference_file": None}

        if not skip_training:
            trainer = Trainer(model_type, base_transformer, n_layers, out_dim, device_num, lr, batch_size, training_file,
                              add_convolution, sum_transpose, prompt_attention, n_heads, n_attn_layers, ann_path,
                              prompts_path, train_test_split_path, edu_segmented_file, sigmoid_threshold,
                              create_training_dataset, interpolation, add_additional_loss, add_context)
            print("\n::::::Starting Training::::::\n")
            if model_type == "baseline":
                t_loss, v_loss, t_loss_head, v_loss_head, t_loss_deprel, \
                    v_loss_deprel, best_valid_loss, best_epoch = train_baseline(N_EPOCHS, trainer, best_valid_loss,
                                                                                model_name, early_stopping)
                t_loss_span, v_loss_span, t_loss_typ, v_loss_typ = None, None, None, None
            else:
                t_loss, v_loss, t_loss_head, v_loss_head, t_loss_deprel, v_loss_deprel, t_loss_span, \
                    v_loss_span, t_loss_typ, v_loss_typ, best_valid_loss, best_epoch = train_regular_single(N_EPOCHS,
                                                                                                            trainer,
                                                                                                            best_valid_loss,
                                                                                                            model_name,
                                                                                                            early_stopping)
            execution_log["training"]["losses"] = {"t_loss": t_loss, "v_loss": v_loss, "t_loss_head": t_loss_head,
                                                   "t_loss_deprel": t_loss_deprel, "v_loss_deprel": v_loss_deprel,
                                                   "t_loss_span": t_loss_span, "v_loss_span": v_loss_span,
                                                   "t_loss_typ": t_loss_typ, "v_loss_typ": v_loss_typ,
                                                   "best_valid_loss": best_valid_loss, "best_epoch": best_epoch}

        if run_inference:
            print("\n::::::Running Inference::::::\n")
            inference_exec = Inference(model_type, base_transformer, n_layers, out_dim, device_num, lr, batch_size,
                                       training_file, add_convolution, sum_transpose, prompt_attention, n_heads, n_attn_layers,
                                       ann_path, prompts_path, train_test_split_path, edu_segmented_file, sigmoid_threshold,
                                       False, interpolation, add_additional_loss, add_context, model_name)
            # create_training_dataset is always set to false during inference

            print("Executing Inference!")
            inference_exec.run_inference()
            execution_log["inference_file"] = inference_exec.inference_file

        print("Saving Execution JSON to:", log_file)
        with open(log_file, 'w') as fp:
            json.dump(execution_log, fp)


if __name__ == '__main__':
    main()
