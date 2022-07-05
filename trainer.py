import torch
import torch.nn as nn
import argparse
import time
import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import numpy as np
import copy
import random
import os.path
from transformers import BertTokenizerFast, DebertaTokenizerFast, BertConfig, DebertaConfig
from transformers import DebertaModel, BertModel, RobertaModel, RobertaConfig, RobertaTokenizerFast
from datetime import datetime
from models import Parser, BaselineParser
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import re

# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


class Trainer:
    def __init__(self, model_type, base_transformer, n_layers, out_dim, device_num, lr, batch_size, fname,
                 add_convolution, sum_transpose, prompt_attention, n_heads, n_attn_layers, ann_path, prompts_path,
                 train_test_split_path, edu_segmented_file, sigmoid_threshold=0.5, create_training_dataset=True,
                 interpolation=0.05, add_additional_loss=False, add_context="none"):

        self.device = torch.device("cuda:{}".format(device_num)) if torch.cuda.is_available() else "cpu"
        self.model_type, self.base_transformer = model_type, base_transformer
        self.add_convolution, self.sum_transpose, self.prompt_attention = add_convolution, sum_transpose, \
                                                                          prompt_attention
        self.n_layers, self.out_dim, self.lr, self.batch_size = n_layers, out_dim, lr, batch_size
        self.n_heads, self.n_attn_layers = n_heads, n_attn_layers
        self.sigmoid_threshold = sigmoid_threshold
        self.ann_path, self.prompts_path, self.train_test_split_path, \
            self.edu_segmented_file = ann_path, prompts_path, train_test_split_path, edu_segmented_file
        self.fname, self.interpolation, self.add_additional_loss = fname, interpolation, add_additional_loss
        self.add_context = add_context

        if model_type == "baseline":
            self.label_mapping = {'None': 0, 'C-App': 1, 'NA-App': 2, 'P-App': 3, 'MC-App': 4, 'supports': 5,
                                  'attacks': 6, 'for': 7, 'against': 8, 'default': 9}
            self.label_mapping_rev = {v: k for k, v in self.label_mapping.items()}
            self.parser, self.tokenizer = self.get_baseline_model()
            rel_criterion = nn.CrossEntropyLoss(ignore_index=0)
            bin_criterion = nn.BCEWithLogitsLoss()
            self.criterion_dct = {"criterion_head": bin_criterion,
                                  "criterion_deprel": rel_criterion}
            create_training_dataset = False
            self.batch_size = 1
        else:
            self.label_mapping = {"NA": 0, "For": 1, "Against": 2, "Support": 3, "Attack": 4, "Append": 5, "Default": 6}
            self.type_mapping = {"NonArg": 0, "MajorClaim": 1, "Claim": 2, "Premise": 3}
            self.adu_mapping = {"O": 0, "B": 1}
            self.label_mapping_rev = {v: k for k, v in self.label_mapping.items()}
            self.type_mapping_rev = {v: k for k, v in self.type_mapping.items()}
            self.adu_mapping_rev = {v: k for k, v in self.adu_mapping.items()}

            self.parser, self.tokenizer, self.special_token_idx = self.get_model()
            print("special_token_idx\n", self.special_token_idx)
            print("Num Parameters in model:", self.count_parameters())

            rel_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
            if self.add_additional_loss != "none":
                bin_criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                bin_criterion = nn.BCEWithLogitsLoss()
            span_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            typ_criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
            self.criterion_dct = {"criterion_head": bin_criterion,
                                  "criterion_deprel": rel_criterion,
                                  "criterion_adu_boundary": span_criterion,
                                  "criterion_adu_type": typ_criterion}

        self.optimizer = torch.optim.AdamW(self.parser.parameters(), lr=lr)

        if create_training_dataset:
            print("create_training_dataset flag is True. Hence will create the formatted training dataset",fname)
            self.example_dict = self.format_data_for_training()
        else:
            print("create_training_dataset is False. Hence will reuse existing training dataset", fname)
            self.example_dict = pickle.load(open(fname, "rb"))

        if model_type != "baseline":
            data_dict = self.get_data_dict()
            self.train_length_dict, self.test_length_dict = data_dict["train"], data_dict["test"]
        else:
            self.train_length_dict, self.test_length_dict = self.example_dict["train_dict"], \
                                                            self.example_dict["test_dict"]
        print("Data Dicts Loaded!")

    def count_parameters(self):
        return sum(p.numel() for p in self.parser.parameters() if p.requires_grad)

    def get_tokens(self, lst, y):
        lst = [i[0] for i in lst[1:]]
        tokenized = self.tokenizer(" ".join(lst).replace(" ##", ""), add_special_tokens=False)
        assert len(tokenized.input_ids) == y.shape[0] - 1
        return tokenized.input_ids, tokenized.attention_mask

    def get_baseline_dataset(self, data_dict, typ="train"):
        data_dict = copy.deepcopy(data_dict)
        input_ids, attention_masks = [], []
        for ix, i in enumerate(data_dict["x"]):
            ids, mask = self.get_tokens(i, data_dict["y"][ix])
            input_ids.append(ids)
            attention_masks.append(mask)

        num_list = list(range(len(input_ids)))
        if typ == "train":
            random.shuffle(num_list)
        counter, total_examples = 0, len(num_list)
        while counter < total_examples:
            idx = num_list[counter]
            in_id = torch.tensor(input_ids[idx], dtype=torch.long).unsqueeze(0)
            attn_msk = torch.tensor(attention_masks[idx], dtype=torch.long).unsqueeze(0)
            rel_mat = torch.tensor(data_dict["y"][idx], dtype=torch.long).unsqueeze(0)
            head_mat = (rel_mat != 0) * 1.0
            assert in_id.size() == attn_msk.size()
            counter += 1
            yield in_id, attn_msk, rel_mat, head_mat

    def get_baseline_model(self):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        base_model = BertModel.from_pretrained("bert-base-uncased")
        in_dim, out_dim = 768, 600
        n_deprel_classes = len(self.label_mapping)
        model = BaselineParser(base_model, in_dim, out_dim, n_deprel_classes).to(self.device)
        return model, tokenizer

    def get_model(self):
        if "deberta" in self.base_transformer:
            tokenizer = DebertaTokenizerFast.from_pretrained(self.base_transformer)
            config = DebertaConfig.from_pretrained(self.base_transformer)
        elif "bert-base" in self.base_transformer:
            tokenizer = BertTokenizerFast.from_pretrained(self.base_transformer)
            config = BertConfig.from_pretrained(self.base_transformer)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(self.base_transformer)
            config = RobertaConfig.from_pretrained(self.base_transformer)

        keys_to_add = ["<EDU>"]
        special_tokens_dict = {'additional_special_tokens': keys_to_add}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        special_token_idx = [tokenizer.get_vocab()[i] for i in keys_to_add]
        special_token_idx.append(tokenizer.cls_token_id)
        print(num_added_toks, "tokens added\n")
        config.output_hidden_states = True

        if "deberta" in self.base_transformer:
            base_model = DebertaModel.from_pretrained(self.base_transformer, config=config)
        elif "bert-base" in self.base_transformer:
            base_model = BertModel.from_pretrained(self.base_transformer, config=config)
        else:
            base_model = RobertaModel.from_pretrained(self.base_transformer, config=config)

        base_model.resize_token_embeddings(len(tokenizer))

        in_dim = config.hidden_size
        n_deprel_classes = len(self.label_mapping)
        n_type_classes = len(self.type_mapping)
        n_token_classes = len(self.adu_mapping)

        model = Parser(base_model, self.n_layers, in_dim, self.out_dim, n_deprel_classes, n_type_classes,
                       n_token_classes, self.add_convolution, self.sum_transpose, self.prompt_attention,
                       self.n_heads, self.n_attn_layers, self.add_context).to(self.device)

        return model, tokenizer, special_token_idx

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def make_tgt_dict(self, head, deprel, bound, adu_type):
        if self.add_additional_loss != "none":
            tgt_head = head
        else:
            tgt_head = head.contiguous().view(-1)
        return {"tgt_head": tgt_head, # head.contiguous().view(-1),
                "tgt_deprel": deprel.contiguous().view(-1),
                "tgt_adu_boundary": bound.contiguous().view(-1),
                "tgt_adu_type": adu_type.contiguous().view(-1)}

    def compute_loss(self, output_dct, criterion_dct, target_dct):
        loss_dict = {}
        for k, criterion in criterion_dct.items():
            logits = output_dct[k.replace("criterion", "logits")]

            if "head" not in k:
                logits = logits.contiguous().view(-1, logits.shape[-1])
                loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])
            else:
                if self.add_additional_loss == "both":
                    alpha = 0.85
                    num_stability = torch.tensor(1e-10).to(logits.device)
                    loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])# b, n, n
                    tgt = target_dct[k.replace("criterion", "tgt")] + target_dct[k.replace("criterion", "tgt")].transpose(2, 1)
                    assert (tgt > 1).sum().item() == 0
                    absolute_diff = torch.abs(torch.sigmoid(logits) - torch.sigmoid(logits.transpose(2, 1)))

                    additional_loss = tgt * torch.log(torch.max(absolute_diff, num_stability)) + \
                                      (1-tgt) * torch.log(torch.max(1-absolute_diff, num_stability))

                    # loss: -alpha(y_i log sig(y_i_pred) + (1-y_i) log 1-sig(y_i_pred)) \
                    # -(1-alpha) y_i log(1-sig(y_trans_i_pred))
                    # mask the diagonal and first column
                    mask = torch.ones_like(loss) - torch.diag_embed(torch.ones(loss.size()[:2])).to(loss.device)
                    mask = torch.cat([torch.zeros_like(mask[:, :, :1]), mask[:, :, 1:]], -1).to(logits.device)
                    loss = ((alpha * loss) - ((1 - alpha) * additional_loss)) * mask
                    loss = loss.mean()
                elif self.add_additional_loss == "one":
                    alpha = 0.85
                    num_stability = torch.tensor(1e-10).to(logits.device)
                    loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])  # b, n, n
                    additional_loss = target_dct[k.replace("criterion", "tgt")] * \
                                      torch.log(torch.max(1 - torch.sigmoid(logits.transpose(2, 1)),
                                                          num_stability))
                    # loss: -alpha(y_i log sig(y_i_pred) + (1-y_i) log 1-sig(y_i_pred)) \
                    # -(1-alpha) y_i log(1-sig(y_trans_i_pred))
                    # mask the diagonal and first column
                    mask = torch.ones_like(loss) - torch.diag_embed(torch.ones(loss.size()[:2])).to(loss.device)
                    mask = torch.cat([torch.zeros_like(mask[:, :, :1]), mask[:, :, 1:]], -1).to(logits.device)
                    loss = ((alpha * loss) - ((1 - alpha) * additional_loss)) * mask
                    loss = loss.mean()

                else:
                    logits = logits.contiguous().view(-1)
                    loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])
            loss_dict[k.replace("criterion", "loss")] = loss
        return loss_dict

    def classification_stats_suite(self, tgt_t, src_t, typ, ignore_val=-1):
        tgt, src = [], []
        for ix, i in enumerate(tgt_t):
            if i != ignore_val:
                tgt.append(i)
                src.append(src_t[ix])
        assert len(tgt) == len(src)
        cm = confusion_matrix(tgt, src)
        cs = classification_report(tgt, src)
        print("\n===== STATS FOR ", typ, "=====")
        print("Confusion metric : \n", cm)
        print("Classification Stats:\n", cs)
        print("==============================\n")

    def train(self, iterator):
        self.parser.train()
        ep_t_loss, head_loss, deprel_loss, adu_bound_loss, adu_typ_loss = 0, 0, 0, 0, 0
        batch_num = 0

        for batch in tqdm(iterator):
            self.optimizer.zero_grad()

            batch = tuple(t.to(self.device) for t in batch)
            edu_toks_enc, edu_toks_attn, adu_tags, rel_mat, head_mat, \
                pos_arr, edu_labels, prompt_toks, prompt_toks_attn = batch

            output_dct = self.parser(edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn)
            tgt_dict = self.make_tgt_dict(head_mat, rel_mat, adu_tags, edu_labels)

            loss_dict = self.compute_loss(output_dct, self.criterion_dct, tgt_dict)

            alpha, beta = self.interpolation, 1 - (self.interpolation * (len(loss_dict.keys())-1))
            assert beta > alpha and beta + (alpha * (len(loss_dict.keys())-1)) == 1.0
            loss = beta * loss_dict["loss_head"] + alpha * loss_dict["loss_deprel"] + \
                   alpha * loss_dict["loss_adu_boundary"] + alpha * loss_dict["loss_adu_type"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parser.parameters(), 1.0)
            self.optimizer.step()

            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            adu_typ_loss += loss_dict["loss_adu_type"].item()
            adu_bound_loss += loss_dict["loss_adu_boundary"].item()
            batch_num += 1

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num, adu_bound_loss / batch_num, \
               adu_typ_loss / batch_num

    def evaluate(self, iterator):
        self.parser.eval()

        ep_t_loss, head_loss, deprel_loss, adu_bound_loss, adu_typ_loss = 0, 0, 0, 0, 0
        head_src, head_tgt, deprel_src, deprel_tgt = [], [], [], []
        adu_bound_src, adu_bound_tgt = [], []
        adu_typ_tgt, adu_typ_src = [], []
        batch_num = 0

        for batch in tqdm(iterator):
            batch = tuple(t.to(self.device) for t in batch)
            edu_toks_enc, edu_toks_attn, adu_tags, rel_mat, head_mat, \
                pos_arr, edu_labels, prompt_toks, prompt_toks_attn = batch
            with torch.no_grad():
                output_dct = self.parser(edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn)

            tgt_dict = self.make_tgt_dict(head_mat, rel_mat, adu_tags, edu_labels)
            loss_dict = self.compute_loss(output_dct, self.criterion_dct, tgt_dict)

            alpha, beta = self.interpolation, 1 - (self.interpolation * (len(loss_dict.keys())-1))
            assert beta > alpha and beta + (alpha * (len(loss_dict.keys()) - 1)) == 1.0
            loss = beta * loss_dict["loss_head"] + alpha * loss_dict["loss_deprel"] + \
                   alpha * loss_dict["loss_adu_boundary"] + alpha * loss_dict["loss_adu_type"]

            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            adu_typ_loss += loss_dict["loss_adu_type"].item()
            adu_bound_loss += loss_dict["loss_adu_boundary"].item()
            batch_num += 1

            head_src.extend((torch.sigmoid(output_dct["logits_head"]) >= self.sigmoid_threshold).long().detach().view(-1).tolist())
            head_tgt.extend(tgt_dict["tgt_head"].detach().cpu().view(-1).tolist())

            deprel_src.extend(output_dct["logits_deprel"].argmax(dim=-1).detach().view(-1).tolist())
            deprel_tgt.extend(tgt_dict["tgt_deprel"].detach().cpu().view(-1).tolist())

            adu_typ_src.extend(output_dct["logits_adu_type"].argmax(dim=-1).detach().view(-1).tolist())
            adu_typ_tgt.extend(tgt_dict["tgt_adu_type"].detach().cpu().view(-1).tolist())

            adu_bound_src.extend(output_dct["logits_adu_boundary"].argmax(dim=-1).detach().view(-1).tolist())
            adu_bound_tgt.extend(tgt_dict["tgt_adu_boundary"].detach().cpu().view(-1).tolist())

        self.classification_stats_suite(head_tgt, head_src, "Edge Prediction", ignore_val=-1)
        self.classification_stats_suite(deprel_tgt, deprel_src, "Edge Type Classification", ignore_val=0)
        self.classification_stats_suite(adu_typ_tgt, adu_typ_src, "ADU Type Classification", ignore_val=-1)
        self.classification_stats_suite(adu_bound_tgt, adu_bound_src, "ADU Span Tagging", ignore_val=-1)

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num, adu_bound_loss / batch_num, \
               adu_typ_loss / batch_num

    def train_baseline(self, iterator):
        self.parser.train()
        ep_t_loss, head_loss, deprel_loss = 0, 0, 0
        batch_num = 0

        for batch in tqdm(iterator):
            self.optimizer.zero_grad()

            batch = tuple(t.to(self.device) for t in batch)
            edu_toks_enc, edu_toks_attn, rel_mat, head_mat = batch

            output_dct = self.parser(edu_toks_enc, edu_toks_attn)
            target_dct = {"tgt_head": head_mat.contiguous().view(-1), "tgt_deprel": rel_mat.contiguous().view(-1)}

            loss_dict = {}
            for k, criterion in self.criterion_dct.items():
                logits = output_dct[k.replace("criterion", "logits")]
                if "head" not in k:
                    logits = logits.contiguous().view(-1, logits.shape[-1])
                else:
                    logits = logits.contiguous().view(-1)
                loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])
                loss_dict[k.replace("criterion", "loss")] = loss

            loss = (1-self.interpolation) * loss_dict["loss_head"] + self.interpolation * loss_dict["loss_deprel"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parser.parameters(), 1.0)
            self.optimizer.step()

            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            batch_num += 1

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num

    def evaluate_baseline(self, iterator):
        self.parser.eval()

        ep_t_loss, head_loss, deprel_loss = 0, 0, 0
        head_src, head_tgt, deprel_src, deprel_tgt = [], [], [], []
        batch_num = 0

        for batch in tqdm(iterator):
            batch = tuple(t.to(self.device) for t in batch)
            edu_toks_enc, edu_toks_attn, rel_mat, head_mat = batch
            with torch.no_grad():
                output_dct = self.parser(edu_toks_enc, edu_toks_attn)

            target_dct = {"tgt_head": head_mat.contiguous().view(-1), "tgt_deprel": rel_mat.contiguous().view(-1)}
            loss_dict = {}
            for k, criterion in self.criterion_dct.items():
                logits = output_dct[k.replace("criterion", "logits")]
                if "head" not in k:
                    logits = logits.contiguous().view(-1, logits.shape[-1])
                else:
                    logits = logits.contiguous().view(-1)
                loss = criterion(logits, target_dct[k.replace("criterion", "tgt")])
                loss_dict[k.replace("criterion", "loss")] = loss

            loss = (1-self.interpolation) * loss_dict["loss_head"] + self.interpolation * loss_dict["loss_deprel"]

            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            batch_num += 1

            head_src.extend((torch.sigmoid(output_dct["logits_head"]) >= self.sigmoid_threshold).long().detach().view(-1).tolist())
            head_tgt.extend(target_dct["tgt_head"].detach().cpu().view(-1).tolist())

            deprel_src.extend(output_dct["logits_deprel"].argmax(dim=-1).detach().view(-1).tolist())
            deprel_tgt.extend(target_dct["tgt_deprel"].detach().cpu().view(-1).tolist())

        self.classification_stats_suite(head_tgt, head_src, "Edge Prediction", ignore_val=-1)
        self.classification_stats_suite(deprel_tgt, deprel_src, "Edge Type Classification", ignore_val=0)

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num

    def train_single(self, iterator):
        self.parser.train()
        ep_t_loss, head_loss, deprel_loss, adu_bound_loss, adu_typ_loss = 0, 0, 0, 0, 0
        batch_num = 0

        self.optimizer.zero_grad()
        for ix, batch in tqdm(enumerate(iterator)):
            edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn, lhc_input_ids, lhc_attn_mask, lhc_pos,\
                rhc_input_ids, rhc_attn_mask, rhc_pos, adu_tags, rel_mat, head_mat, edu_labels = batch

            output_dct = self.parser(edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn, lhc_input_ids,
                                     lhc_attn_mask, lhc_pos, rhc_input_ids, rhc_attn_mask, rhc_pos)

            tgt_dict = self.make_tgt_dict(head_mat, rel_mat, adu_tags, edu_labels)
            loss_dict = self.compute_loss(output_dct, self.criterion_dct, tgt_dict)

            alpha, beta = self.interpolation, 1 - (self.interpolation * (len(loss_dict.keys())-1))
            assert beta > alpha and beta + (alpha * (len(loss_dict.keys())-1)) == 1.0
            loss = beta * loss_dict["loss_head"] + alpha * loss_dict["loss_deprel"] + \
                   alpha * loss_dict["loss_adu_boundary"] + alpha * loss_dict["loss_adu_type"]

            loss = loss / self.batch_size
            loss.backward()

            if (ix + 1) % self.batch_size == 0 or ix + 1 == len(self.train_length_dict["edu_toks_enc"]):
                torch.nn.utils.clip_grad_norm_(self.parser.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            adu_typ_loss += loss_dict["loss_adu_type"].item()
            adu_bound_loss += loss_dict["loss_adu_boundary"].item()
            batch_num += 1

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num, adu_bound_loss / batch_num, \
               adu_typ_loss / batch_num

    def evaluate_single(self, iterator):
        self.parser.eval()

        ep_t_loss, head_loss, deprel_loss, adu_bound_loss, adu_typ_loss = 0, 0, 0, 0, 0
        head_src, head_tgt, deprel_src, deprel_tgt = [], [], [], []
        adu_bound_src, adu_bound_tgt = [], []
        adu_typ_tgt, adu_typ_src = [], []
        batch_num = 0

        for ix, batch in tqdm(enumerate(iterator)):
            edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn, lhc_input_ids, lhc_attn_mask, lhc_pos, \
                rhc_input_ids, rhc_attn_mask, rhc_pos, adu_tags, rel_mat, head_mat, edu_labels = batch

            with torch.no_grad():
                output_dct = self.parser(edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn,
                                         lhc_input_ids, lhc_attn_mask, lhc_pos, rhc_input_ids, rhc_attn_mask, rhc_pos)

            tgt_dict = self.make_tgt_dict(head_mat, rel_mat, adu_tags, edu_labels)
            loss_dict = self.compute_loss(output_dct, self.criterion_dct, tgt_dict)

            alpha, beta = self.interpolation, 1 - (self.interpolation * (len(loss_dict.keys())-1))
            assert beta > alpha and beta + (alpha * (len(loss_dict.keys()) - 1)) == 1.0
            loss = beta * loss_dict["loss_head"] + alpha * loss_dict["loss_deprel"] + \
                   alpha * loss_dict["loss_adu_boundary"] + alpha * loss_dict["loss_adu_type"]

            # loss = loss / self.accum_iter
            ep_t_loss += loss.item()
            head_loss += loss_dict["loss_head"].item()
            deprel_loss += loss_dict["loss_deprel"].item()
            adu_typ_loss += loss_dict["loss_adu_type"].item()
            adu_bound_loss += loss_dict["loss_adu_boundary"].item()
            batch_num += 1

            head_src.extend((torch.sigmoid(output_dct["logits_head"]) >= self.sigmoid_threshold).long().detach().view(-1).tolist())
            head_tgt.extend(tgt_dict["tgt_head"].detach().cpu().view(-1).tolist())

            deprel_src.extend(output_dct["logits_deprel"].argmax(dim=-1).detach().view(-1).tolist())
            deprel_tgt.extend(tgt_dict["tgt_deprel"].detach().cpu().view(-1).tolist())

            adu_typ_src.extend(output_dct["logits_adu_type"].argmax(dim=-1).detach().view(-1).tolist())
            adu_typ_tgt.extend(tgt_dict["tgt_adu_type"].detach().cpu().view(-1).tolist())

            adu_bound_src.extend(output_dct["logits_adu_boundary"].argmax(dim=-1).detach().view(-1).tolist())
            adu_bound_tgt.extend(tgt_dict["tgt_adu_boundary"].detach().cpu().view(-1).tolist())

        self.classification_stats_suite(head_tgt, head_src, "Edge Prediction", ignore_val=-1)
        self.classification_stats_suite(deprel_tgt, deprel_src, "Edge Type Classification", ignore_val=0)
        self.classification_stats_suite(adu_typ_tgt, adu_typ_src, "ADU Type Classification", ignore_val=-1)
        self.classification_stats_suite(adu_bound_tgt, adu_bound_src, "ADU Span Tagging", ignore_val=-1)

        return ep_t_loss / batch_num, head_loss / batch_num, deprel_loss / batch_num, adu_bound_loss / batch_num, \
               adu_typ_loss / batch_num

    def get_context(self, essay_id, para_id):
        all_ids = sorted(list(self.example_dict[essay_id].keys()))
        lhc_ids, rhc_ids = [i for i in all_ids if i < para_id], [i for i in all_ids if i > para_id]
        lhc, rhc = [], []
        for ix in lhc_ids:
            lhc.append(self.example_dict[essay_id][ix]["adu_input_ids"])
        for ix in rhc_ids:
            rhc.append(self.example_dict[essay_id][ix]["adu_input_ids"])
        return lhc, rhc

    def get_data_dict(self):
        train_edu_tokens, test_edu_tokens = [], []
        train_edu_labels, test_edu_labels = [], []
        train_rel_mat, test_rel_mat = [], []
        train_adu_tags, test_adu_tags = [], []
        train_lens, test_lens = [], []
        train_edu_input_ids, test_edu_input_ids = [], []
        train_offset_mappings, test_offset_mappings = [], []
        train_prompt, test_prompt = [], []
        train_lhc, test_lhc = [], []
        train_rhc, test_rhc = [], []

        for k1, v1 in self.example_dict.items():
            for k, v in v1.items():
                lhc, rhc = self.get_context(k1, k)
                if v["split"] == "TRAIN":
                    train_edu_tokens.append(v["edu_tokens"])
                    train_edu_labels.append(v["edu_labels"])
                    train_edu_input_ids.append(v["adu_input_ids"])
                    train_rel_mat.append(v["relationship_matrix"])
                    train_adu_tags.append(v["adu_tags"])
                    train_offset_mappings.append(v["adu_offset_mappings"])
                    train_lens.append(v["relationship_matrix"].shape[0])
                    train_prompt.append(v["prompt"])
                    train_lhc.append(lhc)
                    train_rhc.append(rhc)
                else:
                    test_edu_tokens.append(v["edu_tokens"])
                    test_edu_labels.append(v["edu_labels"])
                    test_edu_input_ids.append(v["adu_input_ids"])
                    test_rel_mat.append(v["relationship_matrix"])
                    test_adu_tags.append(v["adu_tags"])
                    test_offset_mappings.append(v["adu_offset_mappings"])
                    test_lens.append(v["relationship_matrix"].shape[0])
                    test_prompt.append(v["prompt"])
                    test_lhc.append(lhc)
                    test_rhc.append(rhc)

        data_dict = {"train": {"edu_toks": train_edu_tokens, "edu_toks_enc": train_edu_input_ids,
                               "edu_labels": train_edu_labels, "rel_mat": train_rel_mat,
                               "adu_tags": train_adu_tags, "turns": train_lens,
                               "prompt": train_prompt, "lhc": train_lhc, "rhc": train_rhc},
                     "test": {"edu_toks": test_edu_tokens, "edu_toks_enc": test_edu_input_ids,
                              "edu_labels": test_edu_labels, "rel_mat": test_rel_mat,
                              "adu_tags": test_adu_tags, "turns": test_lens,
                              "prompt": test_prompt, "lhc": test_lhc, "rhc": test_rhc}
                     }
        return data_dict

    def group_examples_by_edu_length(self, dct):
        length_dict = {}
        for ix in range(len(dct["adu_tags"])):
            if length_dict.get(dct["turns"][ix], None) is None:
                length_dict[dct["turns"][ix]] = {"edu_toks_enc": [dct["edu_toks_enc"][ix]],
                                                 "edu_labels": [dct["edu_labels"][ix]],
                                                 "rel_mat": [dct["rel_mat"][ix]],
                                                 "adu_tags": [dct["adu_tags"][ix]],
                                                 "prompt": [dct["prompt"][ix]]}
            else:
                length_dict[dct["turns"][ix]]["edu_toks_enc"].append(dct["edu_toks_enc"][ix])
                length_dict[dct["turns"][ix]]["edu_labels"].append(dct["edu_labels"][ix])
                length_dict[dct["turns"][ix]]["rel_mat"].append(dct["rel_mat"][ix])
                length_dict[dct["turns"][ix]]["adu_tags"].append(dct["adu_tags"][ix])
                length_dict[dct["turns"][ix]]["prompt"].append(dct["prompt"][ix])
        return length_dict

    def pad_sequence(self, pad, batch):
        maxlen = max([len(i) for i in batch])
        for ix in range(len(batch)):
            batch[ix].extend([pad] * (maxlen - len(batch[ix])))

    def get_batch_data(self, data_dict):
        data_dict = copy.deepcopy(data_dict)
        counter = 0
        total_examples = sum([len(v["edu_toks_enc"]) for v in data_dict.values()])
        example_tally = {k: list(range(len(v["edu_toks_enc"]))) for k, v in data_dict.items()}

        for k, v in example_tally.items():
            np.random.shuffle(v)

        while counter < total_examples:
            available_lengths = list(example_tally.keys())
            selected_length = random.sample(available_lengths, 1)[0]
            idx_list = example_tally[selected_length][:self.batch_size]
            counter += len(idx_list)

            # Remove selected elements from tally
            for i in idx_list:
                example_tally[selected_length].remove(i)

            # Remove dictionary key if no elements are present in it's tally list
            if len(example_tally[selected_length]) == 0:
                done_length = example_tally.pop(selected_length, None)
                assert len(done_length) == 0

            edu_toks_enc = [data_dict[selected_length]["edu_toks_enc"][idx] for idx in idx_list]
            edu_labels = [data_dict[selected_length]["edu_labels"][idx] for idx in idx_list]
            rel_mat = [data_dict[selected_length]["rel_mat"][idx] for idx in idx_list]
            adu_tags = [data_dict[selected_length]["adu_tags"][idx] for idx in idx_list]
            adu_tags = [[self.adu_mapping[j] for j in i] for i in adu_tags]
            prompt_toks = [data_dict[selected_length]["prompt"][idx] for idx in idx_list]

            self.pad_sequence(self.tokenizer.pad_token_id, edu_toks_enc)
            self.pad_sequence(self.tokenizer.pad_token_id, prompt_toks)
            self.pad_sequence(-1, adu_tags)

            edu_toks_enc = np.asarray(edu_toks_enc)
            edu_toks_attn = (edu_toks_enc != self.tokenizer.pad_token_id) * 1

            prompt_toks = np.asarray(prompt_toks)
            prompt_toks_attn = (prompt_toks != self.tokenizer.pad_token_id) * 1

            edu_labels = np.asarray(edu_labels)
            dummy_labels = np.ones((edu_labels.shape[0], 1)) * -1
            edu_labels = np.concatenate([dummy_labels, edu_labels], -1)

            adu_tags = np.asarray(adu_tags)
            rel_mat = np.asarray(rel_mat)
            head_mat = (rel_mat != 0) * 1
            pos_lst = [[ix for ix, j in enumerate(i) if j in self.special_token_idx] for i in edu_toks_enc.tolist()]
            pos_arr = np.asarray(pos_lst).T

            assert edu_toks_enc.shape == edu_toks_attn.shape == adu_tags.shape
            assert prompt_toks.shape == prompt_toks_attn.shape
            print(pos_arr.T.shape, edu_labels.shape)
            # assert pos_arr.T.shape == edu_labels[:, 1:].shape

            yield torch.tensor(edu_toks_enc, dtype=torch.long), torch.tensor(edu_toks_attn, dtype=torch.long), \
                  torch.tensor(adu_tags, dtype=torch.long), torch.tensor(rel_mat, dtype=torch.long), \
                  torch.tensor(head_mat, dtype=torch.float32), torch.tensor(pos_arr, dtype=torch.long), \
                  torch.tensor(edu_labels, dtype=torch.long), torch.tensor(prompt_toks, dtype=torch.long),\
                  torch.tensor(prompt_toks_attn, dtype=torch.long)

    def get_discourse_pos(self, lst):
        return [ix for ix, i in enumerate(lst) if i in self.special_token_idx]

    def get_single_data(self, data_dict, typ="train"):
        data_dict2 = copy.deepcopy(data_dict)
        num_list = list(range(len(data_dict["edu_toks_enc"])))
        if typ == "train":
            random.shuffle(num_list)

        counter, total_examples = 0, len(num_list)
        edu_toks_enc_lst, prompt_lst, lhc_lst, rhc_lst, edu_labels_lst, rel_mat_lst, \
        adu_tags_lst = copy.deepcopy(data_dict2["edu_toks_enc"]), copy.deepcopy(data_dict2["prompt"]), \
                       copy.deepcopy(data_dict2["lhc"]), copy.deepcopy(data_dict2["rhc"]), copy.deepcopy(
            data_dict2["edu_labels"]), \
                       copy.deepcopy(data_dict2["rel_mat"]), copy.deepcopy(data_dict2["adu_tags"])
        lhc_lst = [[[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]] if len(i) == 0 else i for i in lhc_lst]
        rhc_lst = [[[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]] if len(i) == 0 else i for i in rhc_lst]

        while counter < total_examples:
            ix = num_list[counter]
            curr_input_ids = torch.tensor(edu_toks_enc_lst[ix], dtype=torch.long).unsqueeze(0)
            curr_attn_mask = torch.ones_like(curr_input_ids)

            prompt_input_ids = torch.tensor(prompt_lst[ix], dtype=torch.long).unsqueeze(0)
            prompt_attn_mask = torch.ones_like(prompt_input_ids)

            lhc_input_ids, rhc_input_ids = lhc_lst[ix], rhc_lst[ix]
            self.pad_sequence(self.tokenizer.pad_token_id, lhc_input_ids)
            self.pad_sequence(self.tokenizer.pad_token_id, rhc_input_ids)

            lhc_input_ids = torch.tensor(lhc_input_ids, dtype=torch.long)
            rhc_input_ids = torch.tensor(rhc_input_ids, dtype=torch.long)
            lhc_attn_mask = (lhc_input_ids != self.tokenizer.pad_token_id).long()
            rhc_attn_mask = (rhc_input_ids != self.tokenizer.pad_token_id).long()

            curr_pos = torch.tensor(self.get_discourse_pos(edu_toks_enc_lst[ix]), dtype=torch.long).unsqueeze(0).T
            #         prompt_pos = torch.tensor(self.get_discourse_pos(prompt_lst[ix]), dtype=torch.long).T
            lhc_pos = [torch.tensor(self.get_discourse_pos(i), dtype=torch.long).unsqueeze(0).T.to(self.device) for i in lhc_lst[ix]]
            rhc_pos = [torch.tensor(self.get_discourse_pos(i), dtype=torch.long).unsqueeze(0).T.to(self.device) for i in rhc_lst[ix]]

            edu_labels, rel_mat, adu_tags = edu_labels_lst[ix], rel_mat_lst[ix], adu_tags_lst[ix]
            adu_tags = torch.tensor([self.adu_mapping[i] for i in adu_tags], dtype=torch.long).unsqueeze(0)

            edu_labels = torch.tensor(edu_labels, dtype=torch.long).unsqueeze(0)
            dummy_labels = torch.ones((edu_labels.shape[0], 1)) * -1
            edu_labels = torch.cat([dummy_labels, edu_labels], -1).long()

            rel_mat = torch.tensor(rel_mat, dtype=torch.long).unsqueeze(0)
            head_mat = (rel_mat != 0) * 1
            #         print(curr_input_ids.shape, prompt_input_ids.shape, adu_tags.shape,
            #               lhc_input_ids.shape, rhc_input_ids.shape, edu_labels.shape)
            assert curr_input_ids.shape == curr_attn_mask.shape == adu_tags.shape
            assert prompt_input_ids.shape == prompt_attn_mask.shape
            assert lhc_input_ids.shape == lhc_attn_mask.shape
            assert rhc_input_ids.shape == rhc_attn_mask.shape
            assert curr_pos.T.shape == edu_labels.shape
            # assert curr_pos.T.shape == edu_labels[:, 1:].shape
            counter += 1

            yield curr_input_ids.to(self.device), curr_attn_mask.to(self.device), curr_pos.to(self.device), \
                  prompt_input_ids.to(self.device), prompt_attn_mask.to(self.device), lhc_input_ids.to(self.device), \
                  lhc_attn_mask.to(self.device), lhc_pos, rhc_input_ids.to(self.device), rhc_attn_mask.to(self.device), \
                  rhc_pos, adu_tags.to(self.device), rel_mat.to(self.device),  head_mat.float().to(self.device), \
                  edu_labels.to(self.device)

    def correct_spelling(self, txt):
        spelling_correction = {"responsibl": "responsible", "communicatio": "communication",
                               "educatio": "education", "environmen": "environment", "governmen": "government"}
        for k, v in spelling_correction.items():
            if len(re.findall("\\b" + k + "\\b", txt)) > 0:
                txt = txt.replace(k, v)
        return txt

    def load_annotations(self):
        all_filenames = [self.ann_path + i.split(".")[0] for i in os.listdir(self.ann_path) \
                         if (i.endswith("ann") or i.endswith("txt")) and i.startswith("essay")]
        all_filenames = list(set(all_filenames))
        print(len(all_filenames), "annotation files found in path",self.ann_path,".")
        all_ann_dict = {}

        for name in tqdm(all_filenames):
            with open(name + ".ann") as f:
                ann = f.readlines()
            tmp = {}
            for i in ann:
                ann_line = i.strip().split("\t")

                if ann_line[0].startswith("T"):
                    typ, st, en = ann_line[1].split()
                    st, en = int(st), int(en)
                    tmp[ann_line[0]] = {"type": typ, "start": st, "end": en, "text": self.correct_spelling(ann_line[-1]),
                                        "supports": [], "attacks": []}

                elif ann_line[0].startswith("A"):
                    _, t_node, stance = ann_line[1].split()
                    tmp[t_node]["stance"] = stance

                elif ann_line[0].startswith("R"):
                    typ, a1, a2 = ann_line[1].split()
                    a1, a2 = a1.split(":")[-1], a2.split(":")[-1]
                    tmp[a1][typ].append(a2)

                else:
                    print("\nNOT RECOGNIZED!!\n")

            all_ann_dict[name.split("/")[-1]] = tmp
        return all_ann_dict

    def get_edu_dataframe(self, all_ann_dict, ann_dict):
        default_config = {"type": "NonArg", "stance": "NA", "supports": [], "attacks": []}
        lst = []
        for essay_id, v in ann_dict.items():
            for para_id, v2 in v.items():
                for ix, i in enumerate(list(zip(v2["edu_list"], v2["edu_annotation"]))):
                    edu = i[0]
                    ann, st, en = i[1]
                    dct = all_ann_dict[essay_id].get(ann, default_config)
                    lst.append([essay_id, para_id, ix, edu, ann, st, en, dct["type"], dct.get("stance", "NA"),
                                dct["supports"], dct["attacks"]])

        lst_df = pd.DataFrame(lst, columns=["essay_id", "para_id", "edu_id", "edu", "ann", "st", "en",
                                            "type", "stance", "supports", "attacks"])
        print("Edu dataframe shape:", lst_df.shape)
        return lst_df

    def get_prompts_dict(self):
        prompts = pd.read_csv(self.prompts_path, encoding='unicode_escape', sep=";")
        prompt_input_ids = self.tokenizer(list(prompts["PROMPT"]), return_offsets_mapping=False,
                                          add_special_tokens=True).input_ids
        prompts_dict = {row["ESSAY"].replace(".txt", "").strip(): prompt_input_ids[ix] for ix, row in prompts.iterrows()}
        return prompts_dict

    def text2tag(self, txt, st, en, typ):
        tag = ""
        for ix, i in enumerate(txt):
            if i == " ":
                tag += " "
            elif ix >= st and ix <= en and typ != "NonArg":
                tag += "B"
            else:
                tag += "O"

        if typ != "NonArg":
            assert "".join(set(tag[st:en].replace(" ", ""))) == "B"
        tag_lst = ["".join(set(i)) for i in tag.split()]
        assert len(tag_lst) == len(txt.split())
        return tag_lst

    def tokenize_and_tag_v2(self, txt, st, en, typ, ix, add_edu_marker=True):
        if en == -1:
            en = len(txt)

        if add_edu_marker:
            marker = "<EDU>"#self.tokenizer.cls_token #"<EDU> "
            if ix == 0:
                marker = self.tokenizer.cls_token + marker
            txt = marker + txt
            st += len(marker)
            en += len(marker)

        toks = self.tokenizer(txt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids, offset_mapping = toks.input_ids, toks.offset_mapping
        tok_lst, tok_tag_lst = [], []
        for i in offset_mapping:
            tok_lst.append(txt[i[0]:i[1]])
            if en > i[0] and st < i[1] and typ != "NonArg":
                tok_tag_lst.append("B")
            else:
                tok_tag_lst.append("O")

        assert len(tok_lst) == len(tok_tag_lst) == len(input_ids)
        return tok_lst, tok_tag_lst, input_ids, offset_mapping

    def label_support_attack(self, mat, essay_id, para_id, vl, ann, typ, edu_id_min_max_dict):
        from_idx = edu_id_min_max_dict[(essay_id, para_id, vl)][('edu_id', 'max')]
        to_idx = edu_id_min_max_dict[(essay_id, para_id, ann)][('edu_id', 'max')]
        mat[from_idx][to_idx] = self.label_mapping[typ]
        return mat

    def label_stance(self, mat, essay_id, para_id, ann, typ, edu_id_min_max_dict):
        to_idx = edu_id_min_max_dict[(essay_id, para_id, ann)][('edu_id', 'max')]
        from_idx = mat.shape[0] - 1
        mat[from_idx][to_idx] = self.label_mapping[typ]
        return mat

    def label_major_claim(self, mat, essay_id, para_id, ann, typ, edu_id_min_max_dict):
        all_claims = []
        for ix, i in enumerate(mat[-1]):
            if i in [1, 2]:
                all_claims.append([ix, i])

        mat = self.label_stance(mat, essay_id, para_id, ann, typ, edu_id_min_max_dict)
        for tup in all_claims:
            from_idx = edu_id_min_max_dict[(essay_id, para_id, ann)][('edu_id', 'max')]
            mat[from_idx][tup[0]] = tup[1]
            mat[-1][tup[0]] = 0
        return mat

    def make_example_from_para(self, df, edu_id_min_max_dict):
        prev = None
        mat = np.zeros((df.shape[0] + 1, df.shape[0] + 1))
        essay_id = df.iloc[0]["essay_id"]
        para_id = df.iloc[0]["para_id"]
        edu_labels, txt_tokens, txt_tags = [], [], []
        input_ids, offset_mappings = [], []

        for ix, row in df.iterrows():
            if len(row["supports"]) > 0:
                for sup in row["supports"]:
                    mat = self.label_support_attack(mat, essay_id, para_id, sup, row["ann"], "Support",
                                                    edu_id_min_max_dict)

            if len(row["attacks"]) > 0:
                for att in row["attacks"]:
                    mat = self.label_support_attack(mat, essay_id, para_id, att, row["ann"], "Attack",
                                                    edu_id_min_max_dict)

            if row["stance"] in ["For", "Against"]:
                mat = self.label_stance(mat, essay_id, para_id, row["ann"], row["stance"], edu_id_min_max_dict)

            if prev is not None and prev == row["ann"]:
                mat[row["edu_id"]][row["edu_id"] - 1] = self.label_mapping["Append"]

            prev = row["ann"]
            edu_labels.append(self.type_mapping[row["type"]])
            tok_lst, tok_tag_lst, in_ids, off_map = self.tokenize_and_tag_v2(row["edu"], row["st"], row["en"],
                                                                             row["type"], ix)
            txt_tokens.extend(tok_lst)
            txt_tags.extend(tok_tag_lst)
            input_ids.extend(in_ids)
            offset_mappings.extend(off_map)

        for ix, row in df.iterrows():
            if row["type"] == "MajorClaim":
                mat = self.label_major_claim(mat, essay_id, para_id, row["ann"], "Default", edu_id_min_max_dict)

        t1 = np.concatenate([mat[:, -1:], mat[:, :-1]], 1)
        mat = np.concatenate([t1[-1:, :], t1[:-1, :]])

        assert len(txt_tokens) == len(txt_tags) == len(input_ids)
        return mat, edu_labels, txt_tokens, txt_tags, input_ids, offset_mappings

    def format_data_for_training(self):
        all_ann_dict = self.load_annotations()
        ann_dict = pickle.load(open(self.edu_segmented_file, "rb"))

        lst_df = self.get_edu_dataframe(all_ann_dict, ann_dict)
        edu_id_min_max_dict = lst_df.groupby(["essay_id", "para_id", "ann"]).agg({"edu_id": ["max", "min"]}).to_dict(
            orient="index")

        prompts_dict = self.get_prompts_dict()

        train_test_split = pd.read_csv(self.train_test_split_path, sep=";", encoding='unicode_escape')
        train_test_split_dict = train_test_split.set_index("ID").to_dict(orient="index")

        example_dict = {}

        for ix, row in tqdm(lst_df[["essay_id", "para_id"]].drop_duplicates().reset_index(drop=True).iterrows()):
            tmpdf = lst_df[(lst_df["essay_id"] == row["essay_id"]) &
                           (lst_df["para_id"] == row["para_id"])].reset_index(drop=True)
            edu_rel_mat, edu_lbl, edu_toks, \
            adu_tags, input_ids, offset_mappings = self.make_example_from_para(tmpdf, edu_id_min_max_dict)
            dct = {"relationship_matrix": edu_rel_mat,
                   "edu_labels": edu_lbl,
                   "edu_tokens": edu_toks + [self.tokenizer.sep_token],
                   "adu_tags": adu_tags + len([self.tokenizer.sep_token]) * ["O"],
                   "adu_input_ids": input_ids + [self.tokenizer.sep_token_id],
                   "adu_offset_mappings": offset_mappings,
                   "split": train_test_split_dict[row["essay_id"]]["SET"],
                   "prompt": prompts_dict[row["essay_id"]]
                   }
            if example_dict.get(row["essay_id"], None) is not None:
                example_dict[row["essay_id"]][row["para_id"]] = dct
            else:
                example_dict[row["essay_id"]] = {row["para_id"]: dct}

        print("Saving to", self.fname)
        pickle.dump(example_dict, open(self.fname, "wb"))
        print("\nSANITY\n", self.tokenizer.decode(example_dict["essay001"][4]["adu_input_ids"]),"\n")
        return example_dict
