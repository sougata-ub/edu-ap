import torch
import torch.nn as nn
import torch.nn.functional as F
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
from transformers import DebertaModel, BertModel
from datetime import datetime
from trainer import Trainer
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import inference_utils

# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


class Inference(Trainer):
    def __init__(self, model_type, base_transformer, n_layers, out_dim, device_num, lr, batch_size, fname,
                 add_convolution, sum_transpose, prompt_attention, n_heads, n_attn_layers, ann_path, prompts_path,
                 train_test_split_path, edu_segmented_file, sigmoid_threshold, create_training_dataset, interpolation,
                 add_additional_loss, add_context, finetuned_model, results_loc="results"):
        super().__init__(model_type, base_transformer, n_layers, out_dim, device_num, lr, batch_size, fname,
                         add_convolution, sum_transpose, prompt_attention, n_heads, n_attn_layers, ann_path,
                         prompts_path, train_test_split_path, edu_segmented_file, sigmoid_threshold,
                         create_training_dataset, interpolation, add_additional_loss, add_context)
        self.finetuned_model = finetuned_model
        state_dict = torch.load(finetuned_model)
        self.parser.load_state_dict(state_dict)
        self.parser.eval()
        self.inference_file = finetuned_model.replace("models", results_loc) + "_inference_file.pkl"
        self.stats_file = finetuned_model.replace("models", results_loc) + "_stats.csv"

    def binary_classes(self, lbl):
        if lbl == "NA":
            return 0
        else:
            return 1

    def remap_labels(self, lbl):
        return self.label_mapping[lbl]

    def run_inference(self):
        if self.model_type == "baseline":
            stat_list, additional_data = self.baseline_inference_predictor()
        else:
            stat_list, additional_data = self.inference_predictor()

        stat_list_df = pd.DataFrame(stat_list, columns=["task", "metric", "class label", "num samples", "value"])
        stat_list_df["model"] = self.finetuned_model
        stat_list_df.to_csv(self.stats_file, index=False)

        print("\n:::::STATISTICS:::::\n")
        print(stat_list_df)
        print("=====================================\n")
        print("Dumped Stats results to file", self.stats_file, "!!\n")
        pickle.dump(additional_data, open(self.inference_file, "wb"))
        print("Dumped inference results to file", self.inference_file, "!!")

    def df2pct(self, cnt_df):
        tmp_df = pd.DataFrame(cnt_df).reset_index()
        tmp_df.columns = ["labels", "freq"]
        tmp_df["pct"] = tmp_df["freq"] / tmp_df["freq"].sum()
        tmp_df = tmp_df.set_index("labels").to_dict(orient="index")
        return tmp_df

    def post_process_predictions(self, arr):
        for ix_r, row in enumerate(arr):
            for ix_c, col in enumerate(row):
                if ix_r == 0:
                    if col == 3:
                        arr[ix_r][ix_c] = 1
                    elif col == 4:
                        arr[ix_r][ix_c] = 2
                else:
                    if col == 1:
                        arr[ix_r][ix_c] = 3
                    elif col == 2:
                        arr[ix_r][ix_c] = 4
        return arr

    def baseline_inference_predictor(self):
        stat_list = []
        test_dataloader = self.get_baseline_dataset(self.test_length_dict, typ="test")
        all_matched_list = []
        all_gold_mapping_dict, all_pred_mapping_dict = {}, {}
        all_gold_rel_dict, all_pred_rel_dict = {}, {}
        for ix, batch in tqdm(enumerate(test_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            edu_toks_enc, edu_toks_attn, rel_mat, head_mat = batch
            with torch.no_grad():
                output_dct = self.parser(edu_toks_enc, edu_toks_attn)

                head_pred_probab = torch.sigmoid(output_dct["logits_head"]).detach().cpu().numpy()
                head_pred = (head_pred_probab >= 0.5).astype(int)
                edge_pred = output_dct["logits_deprel"].argmax(dim=-1).detach().cpu().numpy()
                arr_pred = (head_pred * edge_pred).squeeze(0)
                np.fill_diagonal(arr_pred, 0)  # Post processing: Remove self relationships
                arr_gold = (rel_mat * head_mat).squeeze(0).cpu().numpy().astype(int)

                toks = edu_toks_enc.squeeze().tolist()
                span_pred = arr_pred.diagonal(-1).tolist()  # Diagonal -1 contains segment information
                span_gold = arr_gold.diagonal(-1).tolist()

                # Compute ADU span identification: Get golden & pred mapping dict, and store it
                gold_mapping_dict, pred_mapping_dict = inference_utils.get_adu_span_dicts(toks, span_gold,
                                                                                          span_pred,
                                                                                          self.tokenizer)
                all_gold_mapping_dict[ix], all_pred_mapping_dict[ix] = gold_mapping_dict, pred_mapping_dict

                # Find matching ADU spans between golden and predicted sets
                matched_list = inference_utils.pairwise_span_match_baseline(ix, gold_mapping_dict,
                                                                            pred_mapping_dict)
                all_matched_list.extend(matched_list)

                # Relationship matching
                from_list_gold, to_list_gold = np.where(np.isin(arr_gold, [5, 6, 7, 8, 9]))
                from_list_pred, to_list_pred = np.where(np.isin(arr_pred, [5, 6, 7, 8, 9]))

                extracted_rels_pred = inference_utils.extract_relations(ix, from_list_pred, to_list_pred,
                                                                        pred_mapping_dict, arr_pred,
                                                                        head_pred_probab.squeeze(0))
                extracted_rels_gold = inference_utils.extract_relations(ix, from_list_gold, to_list_gold,
                                                                        gold_mapping_dict, arr_gold,
                                                                        np.ones(arr_gold.shape))

                all_gold_rel_dict[ix] = extracted_rels_gold
                all_pred_rel_dict[ix] = extracted_rels_pred

        all_matched_df, pm_f1, pm_p, pm_r, \
            fm_f1, fm_p, fm_r, pm_n, fm_n = inference_utils.compute_span_matches(all_matched_list)
        stat_list.extend([["partial-match:span", "f1", "True", pm_n, pm_f1],
                          ["partial-match:span", "precision", "True", pm_n, pm_p],
                          ["partial-match:span", "recall", "True", pm_n, pm_r],
                          ["full-match:span", "f1", "True", fm_n, fm_f1],
                          ["full-match:span", "precision", "True", fm_n, fm_p],
                          ["full-match:span", "recall", "True", fm_n, fm_r]])

        all_matched_df, span_labels = inference_utils.compute_span_label_matches(all_matched_df, all_gold_mapping_dict,
                                                                                 all_pred_mapping_dict,
                                                                                 self.label_mapping_rev)
        stat_list.extend(span_labels)

        relation_stats = inference_utils.match_relation_triples(all_matched_df, all_gold_rel_dict,
                                                                all_pred_rel_dict, self.label_mapping_rev)
        stat_list.extend(relation_stats)
        additional_data = {"all_matched_df": all_matched_df, "all_gold_rel_dict": all_gold_rel_dict,
                           "all_pred_rel_dict": all_pred_rel_dict, "all_gold_mapping_dict": all_gold_mapping_dict,
                           "all_pred_mapping_dict": all_pred_mapping_dict}
        return stat_list, additional_data

    def process_inference_predictor(self, span_matches, rel_matches):
        stat_list = []
        span_matches_df = pd.DataFrame(span_matches, columns=["essay_id", "para_id", "gold_adu_id", "pred_adu_id",
                                                              "span_gold", "span_pred", "match_pct", "gold_lbl",
                                                              "pred_lbl"])
        span_matches_df["partial_match"] = span_matches_df.apply(lambda x: \
                                                                     inference_utils.label_match(x["span_gold"],
                                                                                                 x["span_pred"],
                                                                                                 x["match_pct"]), 1)
        span_matches_df["full_match"] = span_matches_df.apply(lambda x: \
                                                                  inference_utils.label_match(x["span_gold"],
                                                                                              x["span_pred"],
                                                                                              x["match_pct"], 1.0), 1)
        partial_match_df = span_matches_df[span_matches_df["partial_match"].isin(["tp", "tn"])]
        full_match_df = span_matches_df[span_matches_df["full_match"].isin(["tp", "tn"])]

        partial_dict = span_matches_df.partial_match.value_counts().to_dict()
        full_dict = span_matches_df.full_match.value_counts().to_dict()

        pm_f1 = inference_utils.df2f1(partial_dict)
        pm_pr = inference_utils.df2precision(partial_dict)
        pm_rl = inference_utils.df2recall(partial_dict)
        pm_n = partial_dict["tp"]

        fm_f1 = inference_utils.df2f1(full_dict)
        fm_pr = inference_utils.df2precision(full_dict)
        fm_rl = inference_utils.df2recall(full_dict)
        fm_n = full_dict["tp"]

        stat_list.extend([["partial-match:span", "f1", "True", pm_n, pm_f1],
                          ["partial-match:span", "precision", "True", pm_n, pm_pr],
                          ["partial-match:span", "recall", "True", pm_n, pm_rl],
                          ["full-match:span", "f1", "True", fm_n, fm_f1],
                          ["full-match:span", "precision", "True", fm_n, fm_pr],
                          ["full-match:span", "recall", "True", fm_n, fm_rl]])

        span_label_class_report_partial = classification_report(list(partial_match_df["gold_lbl"]),
                                                                list(partial_match_df["pred_lbl"]),
                                                                zero_division=0, output_dict=True)
        span_label_class_report_full = classification_report(list(full_match_df["gold_lbl"]),
                                                             list(full_match_df["pred_lbl"]),
                                                             zero_division=0, output_dict=True)

        dummy_mapping = {"Claim": "Claim", "MajorClaim": "MajorClaim", "NonArg": "NonArg", "Premise": "Premise"}
        span_label_partial_report = inference_utils.get_report_metrics("partial-match:span-label",
                                                                       span_label_class_report_partial, dummy_mapping)
        span_label_full_report = inference_utils.get_report_metrics("full-match:span-label",
                                                                    span_label_class_report_full, dummy_mapping)
        stat_list.extend(span_label_partial_report + span_label_full_report)

        rel_matches_df = pd.DataFrame(rel_matches, columns=["essay_id", "para_id", "gold_tup", "expected_tup",
                                                            "expected_tup_match_pct", "gold_tup_rel",
                                                            "expected_tup_exists", "expected_tup_rel_exists",
                                                            "expected_tup_rel"])
        rel_matches_df["freq"] = 1
        rel_matches_df = rel_matches_df.fillna(-1)

        rel_matches_df["expected_tup_exists_partial"] = rel_matches_df.apply(lambda x: \
                                                                             inference_utils.mark_false_negative(
                                                                                 x["expected_tup_exists"],
                                                                                 x["expected_tup_match_pct"], 0.5), 1)
        rel_matches_df["expected_tup_rel_exists_partial"] = rel_matches_df.apply(lambda x: \
                                                                                     inference_utils.mark_false_negative(
                                                                                         x["expected_tup_rel_exists"],
                                                                                         x["expected_tup_match_pct"],
                                                                                         0.5), 1)
        rel_matches_df["expected_tup_exists_full"] = rel_matches_df.apply(lambda x: \
                                                                              inference_utils.mark_false_negative(
                                                                                  x["expected_tup_exists"],
                                                                                  x["expected_tup_match_pct"], 1.0), 1)
        rel_matches_df["expected_tup_rel_exists_full"] = rel_matches_df.apply(lambda x: \
                                                                                  inference_utils.mark_false_negative(
                                                                                      x["expected_tup_rel_exists"],
                                                                                      x["expected_tup_match_pct"],
                                                                                      1.0), 1)
        rel_matches_df["expected_tup_rel_part"] = rel_matches_df.apply(lambda x: \
                                                                           inference_utils.change_rel_label(
                                                                               x["expected_tup_rel"],
                                                                               x["expected_tup_match_pct"], 0.5), 1)
        rel_matches_df["expected_tup_rel_full"] = rel_matches_df.apply(lambda x: \
                                                                           inference_utils.change_rel_label(
                                                                               x["expected_tup_rel"],
                                                                               x["expected_tup_match_pct"], 1.0), 1)

        partial_dict_rel = rel_matches_df.expected_tup_exists_partial.value_counts().to_dict()
        full_dict_rel = rel_matches_df.expected_tup_exists_full.value_counts().to_dict()
        partial_dict_rel_lbl = rel_matches_df.expected_tup_rel_exists_partial.value_counts().to_dict()
        full_dict_rel_lbl = rel_matches_df.expected_tup_rel_exists_full.value_counts().to_dict()

        prel_f1 = inference_utils.df2f1(partial_dict_rel)
        prel_pr = inference_utils.df2precision(partial_dict_rel)
        prel_rl = inference_utils.df2recall(partial_dict_rel)
        prel_n = partial_dict_rel["tp"]

        frel_f1 = inference_utils.df2f1(full_dict_rel)
        frel_pr = inference_utils.df2precision(full_dict_rel)
        frel_rl = inference_utils.df2recall(full_dict_rel)
        frel_n = full_dict_rel["tp"]

        prel_lbl_f1 = inference_utils.df2f1(partial_dict_rel_lbl)
        prel_lbl_pr = inference_utils.df2precision(partial_dict_rel_lbl)
        prel_lbl_rl = inference_utils.df2recall(partial_dict_rel_lbl)
        prel_lbl_n = partial_dict_rel_lbl["tp"]

        frel_lbl_f1 = inference_utils.df2f1(full_dict_rel_lbl)
        frel_lbl_pr = inference_utils.df2precision(full_dict_rel_lbl)
        frel_lbl_rl = inference_utils.df2recall(full_dict_rel_lbl)
        frel_lbl_n = full_dict_rel_lbl["tp"]

        lst = [["partial-match:relations", "f1", "True", prel_n, prel_f1],
               ["partial-match:relations", "precision", "True", prel_n, prel_pr],
               ["partial-match:relations", "recall", "True", prel_n, prel_rl],
               ["full-match:relations", "f1", "True", frel_n, frel_f1],
               ["full-match:relations", "precision", "True", frel_n, frel_pr],
               ["full-match:relations", "recall", "True", frel_n, frel_rl],
               ["partial-match:relations-label", "f1", "True", prel_lbl_n, prel_lbl_f1],
               ["partial-match:relations-label", "precision", "True", prel_lbl_n, prel_lbl_pr],
               ["partial-match:relations-label", "recall", "True", prel_lbl_n, prel_lbl_rl],
               ["full-match:relations-label", "f1", "True", frel_lbl_n, frel_lbl_f1],
               ["full-match:relations-label", "precision", "True", frel_lbl_n, frel_lbl_pr],
               ["full-match:relations-label", "recall", "True", frel_lbl_n, frel_lbl_rl]]
        partial_stats_rpt = classification_report(list(rel_matches_df["gold_tup_rel"].apply(int)),
                                                  list(rel_matches_df["expected_tup_rel_part"].apply(int)),
                                                  zero_division=0, output_dict=True)
        full_stats_rpt = classification_report(list(rel_matches_df["gold_tup_rel"].apply(int)),
                                               list(rel_matches_df["expected_tup_rel_full"].apply(int)),
                                               zero_division=0, output_dict=True)

        label_mapping_rev = {str(k): str(v) for k, v in self.label_mapping_rev.items()}
        for k, v in partial_stats_rpt.items():
            if label_mapping_rev.get(k, None) is not None and k in ["1", "2", "3", "4", "6"]:
                lst.append(["partial-match:relations-label", "f1", label_mapping_rev[k], v["support"],
                            v["f1-score"]])

        for k, v in full_stats_rpt.items():
            if label_mapping_rev.get(k, None) is not None and k in ["1", "2", "3", "4", "6"]:
                lst.append(["full-match:relations-label", "f1", label_mapping_rev[k], v["support"],
                            v["f1-score"]])
        stat_list.extend(lst)
        return stat_list, span_matches_df, rel_matches_df

    def inference_predictor(self):
        span_matches, rel_matches = [], []
        counter = 0
        for essay_id, v1 in tqdm(self.example_dict.items()):
            for para_id, v2 in v1.items():
                if v2["split"] == "TEST":
                    batch = self.format_example(essay_id, para_id, v2)
                    edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn, lhc_input_ids, lhc_attn_mask, \
                        lhc_pos, rhc_input_ids, rhc_attn_mask, rhc_pos, adu_tags, rel_mat, head_mat, edu_labels = batch

                    with torch.no_grad():
                        # print(essay_id, para_id)
                        output_dct = self.parser(edu_toks_enc, edu_toks_attn, pos_arr, prompt_toks, prompt_toks_attn,
                                                 lhc_input_ids, lhc_attn_mask, lhc_pos, rhc_input_ids, rhc_attn_mask,
                                                 rhc_pos)
                        head_pred = (torch.sigmoid(output_dct["logits_head"]) >= self.sigmoid_threshold).long().detach().cpu().numpy()
                        np.fill_diagonal(head_pred.squeeze(), 0.0)
                        edge_pred = output_dct["logits_deprel"].argmax(dim=-1).detach().cpu().numpy()
                        span_pred = output_dct["logits_adu_boundary"].argmax(dim=-1).detach().cpu().numpy()

                        assert head_pred.shape == edge_pred.shape
                        assert rel_mat.shape == head_mat.shape

                        adu_type_pred_prob, adu_type_pred = F.softmax(output_dct["logits_adu_type"].squeeze(0), -1).max(
                            dim=-1)
                        adu_type_pred_prob, adu_type_pred = adu_type_pred_prob.tolist(), adu_type_pred.tolist()
                        edu_labels_gold = [self.type_mapping_rev[i] for i in edu_labels.squeeze().tolist()[1:]]
                        edu_labels_pred = [self.type_mapping_rev[i] for i in adu_type_pred[1:]]
                        adu_type_pred_prob = adu_type_pred_prob[1:]
                        assert len(edu_labels_gold) == len(edu_labels_pred) == len(adu_type_pred_prob)

                        label_probab_lst_pred = list(zip(edu_labels_pred, adu_type_pred_prob))
                        label_probab_lst_gold = list(zip(edu_labels_gold, [1.0] * len(edu_labels_gold)))

                        arr_pred = (head_pred * edge_pred).squeeze(0)
                        np.fill_diagonal(arr_pred, 0)  # Post processing: Remove self relationships
                        arr_pred = self.post_process_predictions(arr_pred) # Post processing: Only For/Against relationship possible from root.
                        arr_gold = (rel_mat * head_mat).squeeze(0).cpu().numpy().astype(int)

                        """ EDU Span Matching """
                        span_gold = adu_tags.detach().cpu().numpy()
                        span_match, rel_match = self.pairwise_span_match(essay_id, para_id, edu_toks_enc, span_gold,
                                                                         span_pred, arr_gold, arr_pred,
                                                                         label_probab_lst_gold, label_probab_lst_pred)
                        span_matches.extend(span_match)
                        rel_matches.extend(rel_match)

                        counter += 1
        stat_list, span_matches_df, rel_matches_df = self.process_inference_predictor(span_matches, rel_matches)

        return stat_list, {"span_matches_df": span_matches_df, "rel_matches_df": rel_matches_df}

    def get_context(self, essay_id, para_id):
        all_ids = sorted(list(self.example_dict[essay_id].keys()))
        lhc_ids, rhc_ids = [i for i in all_ids if i < para_id], [i for i in all_ids if i > para_id]
        lhc, rhc = [], []
        for ix in lhc_ids:
            lhc.append(self.example_dict[essay_id][ix]["adu_input_ids"])
        for ix in rhc_ids:
            rhc.append(self.example_dict[essay_id][ix]["adu_input_ids"])
        return lhc, rhc

    def get_discourse_pos(self, lst):
        return [ix for ix, i in enumerate(lst) if i in self.special_token_idx]

    def format_example(self, essay_id, para_id, dct):
        lhc, rhc = self.get_context(essay_id, para_id)
        lhc = [[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]] if len(lhc) == 0 else lhc
        rhc = [[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]] if len(rhc) == 0 else rhc
        lhc_input_ids, rhc_input_ids = copy.deepcopy(lhc), copy.deepcopy(rhc)
        self.pad_sequence(self.tokenizer.pad_token_id, lhc_input_ids)
        self.pad_sequence(self.tokenizer.pad_token_id, rhc_input_ids)
        lhc_input_ids = torch.tensor(lhc_input_ids, dtype=torch.long)
        rhc_input_ids = torch.tensor(rhc_input_ids, dtype=torch.long)
        lhc_attn_mask = (lhc_input_ids != self.tokenizer.pad_token_id).long()
        rhc_attn_mask = (rhc_input_ids != self.tokenizer.pad_token_id).long()

        lhc_pos = [torch.tensor(self.get_discourse_pos(i), dtype=torch.long).unsqueeze(0).T.to(self.device) for i in lhc]
        rhc_pos = [torch.tensor(self.get_discourse_pos(i), dtype=torch.long).unsqueeze(0).T.to(self.device) for i in rhc]

        edu_toks_enc = dct["adu_input_ids"]
        assert len(edu_toks_enc) == len(dct["edu_tokens"])

        edu_toks_enc = [edu_toks_enc]
        prompt_toks_enc = [dct["prompt"]]
        pos_lst = [[ix for ix, j in enumerate(i) if j in self.special_token_idx] for i in edu_toks_enc]

        edu_toks_enc = np.asarray(edu_toks_enc)  # 1 x seq len
        prompt_toks_enc = np.asarray(prompt_toks_enc)  # 1 x seq len

        edu_toks_attn = (edu_toks_enc != self.tokenizer.pad_token_id) * 1  # 1 x seq len
        prompt_toks_attn = (prompt_toks_enc != self.tokenizer.pad_token_id) * 1  # 1 x seq len

        rel_mat = np.asarray([dct["relationship_matrix"]])  # 1 x n_adu x n_adu
        head_mat = (rel_mat != 0) * 1  # 1 x n_adu x n_adu
        adu_tags = np.asarray([[self.adu_mapping[i] for i in dct["adu_tags"]]])  # 1 x seq len
        pos_arr = np.asarray(pos_lst).T  # n_adu x 1
        edu_labels = np.asarray([[-1] + dct["edu_labels"]])  # 1, n_classes

        assert edu_toks_enc.shape == edu_toks_attn.shape == adu_tags.shape
        assert edu_labels.shape[1] == rel_mat.shape[1]
        assert pos_arr.T.shape == edu_labels.shape
        return (torch.tensor(edu_toks_enc, dtype=torch.long).to(self.device),
                torch.tensor(edu_toks_attn, dtype=torch.long).to(self.device),
                torch.tensor(pos_arr, dtype=torch.long).to(self.device),
                torch.tensor(prompt_toks_enc, dtype=torch.long).to(self.device),
                torch.tensor(prompt_toks_attn, dtype=torch.long).to(self.device), lhc_input_ids.to(self.device),
                lhc_attn_mask.to(self.device), lhc_pos, rhc_input_ids.to(self.device), rhc_attn_mask.to(self.device),
                rhc_pos, torch.tensor(adu_tags, dtype=torch.long).to(self.device),
                torch.tensor(rel_mat, dtype=torch.long).to(self.device),
                torch.tensor(head_mat, dtype=torch.float32).to(self.device),
                torch.tensor(edu_labels, dtype=torch.long).to(self.device))

    def get_paths(self, triples_dict, remove_append=True):
        # print("Triples:", triples_dict)
        path_traversals = []
        for k, v in triples_dict.items():
            tmp = [(k, v["rel"], v["to"])]
            curr = v["to"]
            flag = True
            while curr != -1 and flag:
                try:
                    dct = triples_dict[curr]
                    tmp.append((curr, dct["rel"], dct["to"]))
                    curr = dct["to"]
                except:
                    flag = False

            path_traversals.append(tmp)

        path_traversals_cp = copy.deepcopy(path_traversals)
        for i in path_traversals:
            flag = True
            for j in path_traversals_cp:
                if i != j:
                    if all([k in j and len(i) < len(j) for k in i]):
                        path_traversals_cp.remove(i)
                        flag = False
                        break
        if remove_append:
            lst = []
            for i in path_traversals_cp:
                tmp = [j for j in i if j[1] != "Append"]
                if len(tmp) > 0:
                    lst.append(tmp)
            return lst
        else:
            return path_traversals_cp

    def modify_relationships(self, arr, rel):
        connected_segments = self.find_connected_segments(arr)
        connected_segments_dict = {max(i): i for i in connected_segments}
        connected_segments_dict_unrolled = {i: k for k, v in connected_segments_dict.items() for i in v}
        rel_cp = []
        for i in rel:
            src_root, dest_root = connected_segments_dict_unrolled.get(i[0], i[0]), \
                                  connected_segments_dict_unrolled.get(i[1], i[1])
            if i[2] != "Append":
                if src_root == dest_root:
                    pass
                else:
                    rel_cp.append((src_root, dest_root, i[2]))
            else:
                rel_cp.append(i)
        return list(set(rel_cp))

    def get_relationships(self, arr):
        rel_triples = []
        for to_id, to_lst in enumerate(arr):
            for from_id, rel_value in enumerate(to_lst):
                if from_id != to_id and rel_value != 0:
                    if rel_value == 5:
                        if from_id + 1 == to_id:
                            rel_triples.append((from_id - 1, to_id - 1, self.label_mapping_rev[rel_value]))
                    else:
                        rel_triples.append((from_id - 1, to_id - 1, self.label_mapping_rev[rel_value]))

        return self.modify_relationships(arr, rel_triples)

    def majority_from_list(self, lst):
        """ Given a list of labels, find the majority class, with rules to break ties"""
        typ_cnt = Counter(lst)
        _, max_val = Counter(lst).most_common(1)[0]
        max_tags = [k for k, v in typ_cnt.items() if v == max_val]
        if "MajorClaim" in max_tags:
            max_tag = "MajorClaim"
        elif "Claim" in max_tags:
            max_tag = "Claim"
        elif "Premise" in max_tags:
            max_tag = "Premise"
        else:
            max_tag = "NonArg"
        return max_tag

    def fix_connected_component(self, comp, dct):
        lst = []
        for i in comp:
            tmp = []
            for j in sorted(i):
                if len(dct[j]) == 0:
                    if len(tmp) > 0:
                        lst.append(tmp)
                    tmp = []
                    lst.append([j])
                else:
                    tmp.append(j)
            if len(tmp) > 0:
                lst.append(tmp)
        assert len(set([j for i in lst for j in i])) == len(set([j for i in comp for j in i]))
        return lst

    def find_connected_segments(self, arr):
        """ Clusters edu_ids whose segments are predicted by be connected by the relationship matrix"""
        edu_list = list(range(1, arr.shape[0]))
        segments = []
        for edu_id in edu_list:
            if edu_id + 1 in edu_list and arr[edu_id + 1][edu_id] == 5:
                pass
            else:
                vl, idx = 5, edu_id
                tmp = []
                while vl == 5 and idx > 0:
                    tmp.append(idx - 1)
                    vl = arr[idx][idx - 1]
                    idx -= 1
                segments.append(tmp)
        return segments

    def get_span_tokens(self, lbl_toks, majority_class):
        span_lst = []
        prev = False
        for ix, i in enumerate(lbl_toks):
            if i[1] == 1:
                span_lst.append(i[0])
                prev = True
            else:
                if ix != 0 and prev and ix + 1 < len(lbl_toks) and lbl_toks[ix + 1][1] == 1 and majority_class == 1:
                    span_lst.append(i[0])
                    prev = True
                else:
                    prev = False
        return span_lst

    def select_predominant(self, zippped_list):
        dct = Counter([i[1] for i in zippped_list])
        if dct[1] >= dct[0]:
            return 1
        else:
            return 0

    def get_adu_span_dict(self, input_ids, span_pred):
        """ Extracts token spans from input_ids, using the span_pred values.
            Also handles irregularities in span labelling by assigning incorrect
            labels to majority class, depending on the affiliation of it's neighbours """
        spans, tmp = [], []
        adu_spans, adu_span_dict = [], {}
        for i in list(zip(input_ids.squeeze(0).tolist(), span_pred.squeeze(0).tolist())):
            if i[0] in self.special_token_idx:
                if len(tmp) > 0:
                    spans.append(tmp)
                    tmp = []
            else:
                tmp.append([i[0], i[1]])
        if len(tmp) > 0:
            spans.append(tmp)

        for lbl_toks in spans:
            majority_class = self.select_predominant(lbl_toks)
            adu_spans.append(self.get_span_tokens(lbl_toks, majority_class))

        for ix, i in enumerate(adu_spans):
            adu_span_dict[ix] = i  # tokenizer.decode(i)

        return adu_span_dict

    def find_overlap(self, s1, s2):
        """ Calculates overlap of elements in two lists"""
        s1_toks, s2_toks = set(s1), set(s2)
        if s1 == s2:
            return 1.0

        if len(s1_toks) == 0 and len(s2_toks) == 0:
            return 1.0

        if len(s1_toks) == 0 or len(s2_toks) == 0:
            return 0

        intersect = s1_toks.intersection(s2_toks)
        wrt_s1, wrt_s2 = len(intersect) / len(s1_toks), len(intersect) / len(s2_toks)
        return min([wrt_s1, wrt_s2])

    def get_best_label(self, lbl_probab):
        df = pd.DataFrame(lbl_probab, columns=["label", "probab"])
        df["freq"] = 1
        df = df.groupby(["label"]).agg({"freq": "sum", "probab": "mean"}).reset_index()
        df = df.sort_values(["freq", "probab"], ascending=False)
        max_freq = df.iloc[0]["freq"]
        max_prob = df.iloc[0]["probab"]
        label_list = list(df[(df["freq"] == max_freq) & (df["probab"] == max_prob)]["label"])

        if "MajorClaim" in label_list:
            return "MajorClaim"
        elif "Claim" in label_list:
            return "Claim"
        else:
            return label_list[0]

    def consolidate_dict_by_connected_components(self, connected_list, dict_old, label_probab_lst):
        dict_new = {}
        for i in connected_list:
            tmp, lbl_probab = [], []
            for j in sorted(i):
                tmp.extend(dict_old[j])
                lbl_probab.append(label_probab_lst[j])
            dict_new[j] = {"tokens": tmp, "label": self.get_best_label(lbl_probab)}
        return dict_new

    def map_gold_pred(self, gold_mapping_dict, pred_mapping_dict):
        matched_list, pred_ids_used = [], []
        # For each golden span, find the most matching predicted span.
        for gold_id, golden in gold_mapping_dict.items():
            max_overlap, max_idx = 0, None
            for pred_id, predicted in pred_mapping_dict.items():
                # Overlap is performed by matching set of tokens
                overlap = self.find_overlap(golden["tokens"], predicted["tokens"])
                if overlap >= max_overlap:
                    max_overlap = overlap
                    max_idx = pred_id

            if max_idx is not None:
                pred_ids_used.append(max_idx)
            matched_list.append([gold_id, max_idx, max_overlap])

        max_map = {}
        for ix, i in enumerate(matched_list):
            if max_map.get(i[1], None) is None:
                max_map[i[1]] = {"max_val": i[-1], "index": ix}
            else:
                if max_map[i[1]]["max_val"] < i[-1]:
                    max_map[i[1]]["max_val"] = i[-1]
                    max_map[i[1]]["index"] = ix

        for ix, i in enumerate(matched_list):
            d = max_map.get(i[1])
            if d["index"] != ix:
                matched_list[ix] = i[:1] + [-1, -1]

        # Additional false positives that did not match at all with the golden spans
        false_positives = [pred_id for pred_id, predicted in pred_mapping_dict.items() if pred_id not in pred_ids_used]
        matched_list.extend([[-1, i, -1] for i in false_positives])
        return matched_list

    def rel_logic_check(self, src_adu, tgt_adu, rel_type):
        if src_adu == "Claim" and tgt_adu == "Premise":
            if rel_type in [3, 4]:
                return True, True
            else:
                return True, False
        elif src_adu == "MajorClaim" and tgt_adu == "Claim":
            if rel_type in [1, 2]:
                return True, True
            else:
                return True, False
        elif src_adu == "Premise" and tgt_adu == "Premise":
            if rel_type in [3]:
                return True, True
            else:
                return True, False
        elif src_adu == "ROOT" and tgt_adu == "Claim":
            if rel_type in [1, 2]:
                return True, True
            else:
                return True, False
        elif src_adu == "ROOT" and tgt_adu == "MajorClaim":
            if rel_type in [6]:
                return True, True
            else:
                return True, False
        else:
            return False, False

    def pairwise_span_match(self, essay_id, para_id, edu_toks_enc, span_gold, span_pred, arr_gold, arr_pred,
                            label_probab_lst_gold, label_probab_lst_pred):
        # Step1: Get dictionary of ADU spans. Spans without any ADU are []
        gold_dict = self.get_adu_span_dict(edu_toks_enc, span_gold)
        pred_dict = self.get_adu_span_dict(edu_toks_enc, span_pred)

        from_list_gold, to_list_gold = np.where(np.isin(arr_gold, [1, 2, 3, 4, 6]))
        from_list_pred, to_list_pred = np.where(np.isin(arr_pred, [1, 2, 3, 4, 6]))

        # Step2: Find connected ADUs
        gold_connected, pred_connected = self.find_connected_segments(arr_gold), self.find_connected_segments(arr_pred)
        gold_connected, pred_connected = self.fix_connected_component(gold_connected, gold_dict), \
                                            self.fix_connected_component(pred_connected, pred_dict)
        gold_connected_dict = {max(i): i for i in gold_connected}
        gold_connected_dict_unrolled = {i: k for k, v in gold_connected_dict.items() for i in v}
        pred_connected_dict = {max(i): i for i in pred_connected}
        pred_connected_dict_unrolled = {i: k for k, v in pred_connected_dict.items() for i in v}
        assert len(gold_dict) == len(pred_dict)

        # Step3: Update dictionary of ADU spans by combining connected ADU tokens
        overlap_lst, pred_ids_used = [], []
        gold_dict_new = self.consolidate_dict_by_connected_components(gold_connected, gold_dict, label_probab_lst_gold)
        pred_dict_new = self.consolidate_dict_by_connected_components(pred_connected, pred_dict, label_probab_lst_pred)

        # Step4: Compute golden and pred span overlaps
        gold_pred_mapping = self.map_gold_pred(gold_dict_new, pred_dict_new)

        gold_dict_new[-1] = {"tokens": [], "label": -1}
        pred_dict_new[-1] = {"tokens": [], "label": -1}

        # Step5: Save overlap result in overlap_lst
        for i in gold_pred_mapping:
            overlap_lst.append([essay_id, para_id, i[0], i[1],
                                gold_dict_new[i[0]]["tokens"], pred_dict_new[i[1]]["tokens"], i[2],
                                gold_dict_new[i[0]]["label"], pred_dict_new[i[1]]["label"]])

        # Step6: Create mapping from golden adu id to pred adu id
        gold_mapping, pred_mapping, gold2pred = {}, {}, {-1: {'pred_adu_id': -1, 'match_pct': 1.0}}  # {-1:-1}
        for i in overlap_lst:
            if i[2] != -1:
                gold_mapping[i[2]] = {"tokens": i[4], "type": i[7]}
            if i[3] != -1:
                pred_mapping[i[3]] = {"tokens": i[5], "type": i[8]}

            if i[2] != -1 and i[3] != -1:
                gold2pred[i[2]] = {"pred_adu_id": i[3], "match_pct": i[-3]}

        # Step7: Identify golden and predicted (from:to) relationship tuples
        from_list_gold, to_list_gold = from_list_gold - 1, to_list_gold - 1
        from_list_gold = [gold_connected_dict_unrolled.get(i, -1) for i in from_list_gold]
        to_list_gold = [gold_connected_dict_unrolled.get(i, -1) for i in to_list_gold]

        from_list_pred, to_list_pred = from_list_pred - 1, to_list_pred - 1
        from_list_pred = [pred_connected_dict_unrolled.get(i, -1) for i in from_list_pred]
        to_list_pred = [pred_connected_dict_unrolled.get(i, -1) for i in to_list_pred]

        # Step8: Preserve only deduplicated tuples, and add metadata containing edge label of the tuples
        gold_rels, pred_rels = list(set(zip(from_list_gold, to_list_gold))), list(
            set(zip(from_list_pred, to_list_pred)))
        gold_rels = [[i, arr_gold[i[0] + 1, i[1] + 1]] for i in gold_rels]
        pred_rels_map = {}
        pred_mapping[-1] = {"tokens": [self.tokenizer.cls_token_id], "type": "ROOT"}
        for i in pred_rels:
            is_valid = self.rel_logic_check(pred_mapping[i[0]]["type"], pred_mapping[i[1]]["type"],
                                            arr_pred[i[0] + 1, i[1] + 1])
            pred_rels_map[i] = {"is_valid": is_valid,
                                "rel": arr_pred[i[0] + 1, i[1] + 1],
                                "adu_types": (pred_mapping[i[0]]["type"], pred_mapping[i[1]]["type"])}

        # Step9: Calculate the expected prediction tuples from the golden tuples, using gold2pred mapping
        expected_mapping = []  # gold from:to pair, pred from:to pair, relationship
        for i in gold_rels:
            f, t = gold2pred.get(i[0][0], {}), gold2pred.get(i[0][1], {})
            f_id = f["pred_adu_id"] if f.get("pred_adu_id", None) is not None else None
            t_id = t["pred_adu_id"] if t.get("pred_adu_id", None) is not None else None
            f_pct = f["match_pct"] if f.get("match_pct", None) is not None else 0
            t_pct = t["match_pct"] if t.get("match_pct", None) is not None else 0
            expected_mapping.append([i[0], (f_id, t_id), (f_pct, t_pct), i[1]])

        # gold from:to pair, pred from:to pair, pred from:to pair match %, gold from:to pair relationship,
        # pred from:to pair exists, pred from:to pair relationship matches, pred from:to pair relationship

        # Step10: For each expected tuple, check if it exists in the pred tuple map.Assign "tp", "fn" labels accordingly
        pairs_done = []
        for expected_lst in expected_mapping:
            gold_mp, expected_mp, expected_mp_pct, expected_rl = expected_lst
            pred_rl = pred_rels_map.get(expected_mp, None)
            pairs_done.append(expected_mp)
            if None in expected_mp or pred_rl is None:
                tmp = ["fn", "fn", None]
            else:
                if pred_rl["rel"] == expected_rl:
                    tmp = ["tp", "tp", expected_rl]
                else:
                    tmp = ["tp", "fn", expected_rl]
            expected_lst.extend(tmp)

        # Step11: Check for remaining valid predicted tuples, and assign "fp" labels
        for k, i in pred_rels_map.items():
            if k not in pairs_done:
                if i["is_valid"][0]:
                    if i["is_valid"][1]:
                        expected_mapping.append([(None, None), k, (0, 0), None, "fp", "fp", i["rel"]])
        # Step12: Return span prediction and relation prediction results
        expected_mapping = [[essay_id, para_id] + i for i in expected_mapping]
        return overlap_lst, expected_mapping

    def get_segment_type_label(self, essay_id, para_id, arr, edu_labels):
        """ Assign each edu to it's most probable label, depending on it's own label and related units"""
        segments = self.find_connected_segments(arr)
        segment_typ = [[edu_labels[j] for j in i] for i in segments]
        segments_map = []
        for ix, i in enumerate(segments):
            for j in i:
                segments_map.append([essay_id, para_id, j, self.majority_from_list(segment_typ[ix])])
        return segments_map

    def pairwise_span_label_match(self, essay_id, para_id, arr_pred, edu_labels_gold, edu_labels_pred):
        gold_segments_map_df = pd.DataFrame(edu_labels_gold).reset_index()
        gold_segments_map_df.columns = ["edu_id", "gold_type"]
        gold_segments_map_df["essay_id"] = essay_id
        gold_segments_map_df["para_id"] = para_id

        pred_segments_map_df = pd.DataFrame(edu_labels_pred).reset_index()
        pred_segments_map_df.columns = ["edu_id", "pred_type"]
        pred_segments_map_df["essay_id"] = essay_id
        pred_segments_map_df["para_id"] = para_id

        pred_segments_map_df2 = pd.DataFrame(self.get_segment_type_label(essay_id, para_id, arr_pred, edu_labels_pred),
                                             columns=["essay_id", "para_id", "edu_id", "pred_type_2"])
        pred_segments_map_df = pred_segments_map_df2.merge(pred_segments_map_df,
                                                           how="inner", on=["essay_id", "para_id", "edu_id"])
        df = gold_segments_map_df.merge(pred_segments_map_df, how="inner", on=["essay_id", "para_id", "edu_id"])
        return df

    def get_pairwise_edu_relationships(self, arr, adu_list):
        pair_rels = list({(i[0], i[1]): i for i in self.get_relationships(arr)}.values()) # Randomly remove conflicting predictions
        dct = {(i[0], i[1]): i[2] for i in pair_rels}

        for i in list(adu_list):
            for j in list(adu_list):
                rel = dct.get((i, j), "NA")
                pair_rels.append((i, j, rel))
        pair_rels = list(set(pair_rels))
        return pair_rels

    def pairwise_edu_relationship_match(self, essay_id, para_id, arr_gold, arr_pred, edu_list):
        edu_list = [-1] + edu_list
        golden_pairs = self.get_pairwise_edu_relationships(arr_gold, edu_list)
        pred_pairs = self.get_pairwise_edu_relationships(arr_pred, edu_list)
        if len(golden_pairs) != len(pred_pairs):
            print("Essay:",essay_id,"Para:",para_id,"arr_gold:",arr_gold, "arr_pred:",arr_pred,"edu_list:",edu_list)
        assert len(golden_pairs) == len(pred_pairs)

        golden_pairs_df = pd.DataFrame(golden_pairs, columns=["edu1", "edu2", "golden_rel"])
        pred_pairs_df = pd.DataFrame(pred_pairs, columns=["edu1", "edu2", "pred_rel"])
        pairs_df = golden_pairs_df.merge(pred_pairs_df, how="inner", on=["edu1", "edu2"])

        assert len(pairs_df) == len(pred_pairs_df) == len(golden_pairs_df)
        pairs_df["essay_id"] = essay_id
        pairs_df["para_id"] = para_id
        pairs_df = pairs_df[["essay_id", "para_id", "edu1", "edu2", "golden_rel", "pred_rel"]]
        return pairs_df

    def get_triples_text(self, edu_toks_enc, span, arr):
        token_dict = self.get_adu_span_dict(edu_toks_enc, span)
        triples = self.get_relationships(arr)

        graph = {}
        for i in triples:
            if graph.get(i[0], None) is not None:
                graph[i[0]].append((i[1], i[2]))
            else:
                graph[i[0]] = [(i[1], i[2])]

        segments = self.find_connected_segments(arr)
        segment_tokens = {}
        # paths = self.get_paths(triples)
        for i in segments:
            tmp = []
            i = sorted(i)
            for j in i:
                if token_dict.get(j, None) is None:
                    print(token_dict, j, segments, arr)
                tmp.extend(token_dict[j])
            if len(tmp) > 0:
                segment_tokens[i[-1]] = self.tokenizer.decode(tmp).strip()
            segment_tokens[-1] = "ROOT"
        return segment_tokens, graph
