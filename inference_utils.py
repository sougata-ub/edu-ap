import torch
from collections import Counter
import random
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def get_spans(toks, span_list):
    spans, tmp = [], []
    for ix, i in enumerate(span_list):
        if ix == 0:
            tmp.append((toks[ix], i))
        else:
            if i == 0:
                spans.append(tmp)
                tmp = [(toks[ix], i)]

            else:
                tmp.append((toks[ix], i))

    if len(tmp) > 0:
        spans.append(tmp)
    return spans


def assign_max_label(lst):
    tmp = [i for i in lst if i != 0]
    if len(tmp) == 0:
        return 0
    else:
        return Counter(tmp).most_common(1)[0][0]


def get_processed_span(spans):
    spans_processed = []
    for i in spans:
        tmp_toks = [j[0] for j in i]
        label = assign_max_label([j[1] for j in i])
        #         if label != 0:
        spans_processed.append((tmp_toks, label))
    return spans_processed


def find_overlap(s1, s2):
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


def get_span_mapping(lst, tokenizer):
    mapping_lst = []
    for ix, i in enumerate(lst):
        if ix == 0:
            start = 0
        else:
            start = mapping_lst[-1][2]
        mapping_lst.append((ix, start, len(i[0]) + start, i[0], i[1]))
    assert mapping_lst[-1][2] == sum([len(i[0]) for i in lst])

    mapping_dict = {}
    for i in mapping_lst:
        mapping_dict[i[0]] = {"start": i[1], "end": i[2], "adu_tokens": i[3],
                              "adu_label": i[4], "adu_text": tokenizer.decode(i[3]) if i[3] is not None else "NA"}
    # start is inclusive, end is not inclusive
    return mapping_dict


def get_adu_span_dicts(edu_toks_enc, span_gold, span_pred, tokenizer):
    # Get argument spans by considering 0 as argument boundary
    # input => list of [tokens] & span type [0,1,2,3,4]. Output => List of [[(token, type), ..], [...], ...]
    golden_spans = get_spans(edu_toks_enc, span_gold)
    pred_spans = get_spans(edu_toks_enc, span_pred)

    # For the segmented span, assign 1 consistent label which is it's majority label
    # Input => [[(token, type), (..),..], [...], ...]; Output => [([tok1, tok2, ..], segment_type), ([...],),..]
    pred_spans_processed = get_processed_span(pred_spans)
    gold_spans_processed = get_processed_span(golden_spans)

    # Create a mapping file which identifies the start and end location of each segment, along with it's text & label
    # {segment_id: {start:xx, end:yy, adu_tokens: [11,22,33], adu_label:3, adu_text:"dd dfdffsdfs"}}
    gold_mapping_dict = get_span_mapping(gold_spans_processed, tokenizer)
    pred_mapping_dict = get_span_mapping(pred_spans_processed, tokenizer)

    return gold_mapping_dict, pred_mapping_dict


def pairwise_span_match_baseline(example_id, gold_mapping_dict, pred_mapping_dict):
    matched_list, pred_ids_used = [], []
    # For each golden span, find the most matching predicted span.
    for gold_id, golden in gold_mapping_dict.items():
        max_overlap, max_idx = 0, None
        for pred_id, predicted in pred_mapping_dict.items():
            # Overlap is performed by matching set of tokens
            overlap = find_overlap(golden["adu_tokens"], predicted["adu_tokens"])
            if overlap >= max_overlap:
                max_overlap = overlap
                max_idx = pred_id

        if max_idx is not None:
            pred_ids_used.append(max_idx)
        matched_list.append([example_id, gold_id, max_idx, max_overlap])

    max_map = {}
    for ix, i in enumerate(matched_list):
        if max_map.get(i[2], None) is None:
            max_map[i[2]] = {"max_val": i[-1], "index": ix}
        else:
            if max_map[i[2]]["max_val"] < i[-1]:
                max_map[i[2]]["max_val"] = i[-1]
                max_map[i[2]]["index"] = ix

    for ix, i in enumerate(matched_list):
        d = max_map.get(i[2])
        if d["index"] != ix:
            matched_list[ix] = i[:2] + [None, 0.0]

    #Additional false positives that did not match at all with the golden spans
    false_positives = [pred_id for pred_id, predicted in pred_mapping_dict.items() if pred_id not in pred_ids_used]
    matched_list.extend([[example_id, -1, i, -1] for i in false_positives])
    return matched_list


def get_mapping(idx, mapping_dict):
    tmp = None
    for k, v in mapping_dict.items():
        if v["start"] + 1 <= idx < v["end"] + 1:
            tmp = v
            break
    return tmp, k


def rel_logic_check(src_adu, tgt_adu, rel_type):
    if src_adu == 1 and tgt_adu == 3:
        if rel_type in [5, 6]:
            return True, True
        else:
            return True, False
    elif src_adu == 4 and tgt_adu == 1:
        if rel_type in [7, 8]:
            return True, True
        else:
            return True, False
    elif src_adu == 3 and tgt_adu == 3:
        if rel_type in [5]:
            return True, True
        else:
            return True, False
    elif src_adu == -1 and tgt_adu == 1:
        if rel_type in [7, 8]:
            return True, True
        else:
            return True, False
    elif src_adu == -1 and tgt_adu == 4:
        if rel_type in [9]:
            return True, True
        else:
            return True, False
    else:
        return False, False


def is_rel_valid(src, tgt, mapping_dict, rel_type):
    tgt_map, tgt_id = get_mapping(tgt, mapping_dict)  # get the segment id, and metadata of the target
    if src == 0:
        src_adu, src_id, tgt_adu = -1, -1, tgt_map["adu_label"]  # 0 is always root
        src_txt, tgt_txt = "<ROOT>", tgt_map["adu_text"]
    else:
        src_map, src_id = get_mapping(src, mapping_dict)  # get the segment id, and metadata of the source
        src_adu, tgt_adu = src_map["adu_label"], tgt_map["adu_label"]
        src_txt, tgt_txt = src_map["adu_text"], tgt_map["adu_text"]
    dct = {}
    # check if
    # (i) a relationship can exist between the source & target
    # (ii) the relationship that is predicted to exist is valid
    rel_check = rel_logic_check(src_adu, tgt_adu, rel_type)
    if rel_check[0]:
        return rel_check, (src_id, tgt_id), (src_adu, tgt_adu, src_txt, tgt_txt)
    else:
        return rel_check, (src_id, tgt_id), (None, None, None, None)


def extract_relations(ix, from_list, to_list, mapping_dict, arr, arr_prob):
    extracted_rels = []
    for rel in list(zip(from_list, to_list)):
        is_valid, src_tgt, additional = is_rel_valid(rel[0], rel[1], mapping_dict, arr[rel])
        extracted_rels.append([ix, src_tgt, arr[rel], is_valid, additional, rel, arr_prob[rel]])
        # example_id, src_tgt, src_tgt_rel, (is_valid, is_valid_type),
        # (src_adu, tgt_adu, src_txt, tgt_txt), rel, probability of the rel existing
    return extracted_rels


def label_record(overlap, thresh=0.5):
    if overlap >= thresh:
        return "tp"
    elif overlap < thresh and overlap != -1:
        return "fn"
    elif overlap < thresh:  # == -1:
        return "fp"


def df2pct(cnt_df):
    tmp_df = pd.DataFrame(cnt_df).reset_index()
    tmp_df.columns = ["labels", "freq"]
    tmp_df["pct"] = tmp_df["freq"] / tmp_df["freq"].sum()
    tmp_df = tmp_df.set_index("labels").to_dict(orient="index")
    return tmp_df


def df2f1(cnt_dict):
    return cnt_dict["tp"] / (cnt_dict["tp"] + 0.5 * (cnt_dict["fp"] + cnt_dict["fn"]))
    # tp/(tp + 0.5*(fp+fn))


def df2precision(cnt_dict):
    return cnt_dict["tp"] / (cnt_dict["tp"] + cnt_dict["fp"])


def df2recall(cnt_dict):
    return cnt_dict["tp"] / (cnt_dict["tp"] + cnt_dict["fn"])


def get_value(dct, example_id, span_id, field):
    try:
        return dct[example_id][span_id][field]
    except:
        return -1


def get_f1(tp, fp, fn):
    # tp/(tp + 0.5*(fp+fn))
    return len(tp)/(len(tp) + 0.5 * (len(fp) + len(fn)))


def get_precision(tp, fp):
    return len(tp)/(len(tp) + len(fp))


def get_recall(tp, fn):
    return len(tp)/(len(tp) + len(fn))


def compute_span_matches(all_matched_list):
    all_matched_df = pd.DataFrame(all_matched_list, columns=["example_id", "gold_id", "pred_id", "overlap"])
    all_matched_df["partial_matched"] = all_matched_df.apply(lambda x: label_record(x["overlap"], 0.5), 1)
    all_matched_df["full_matched"] = all_matched_df.apply(lambda x: label_record(x["overlap"], 1.0), 1)

    partial_dict = all_matched_df.partial_matched.value_counts().to_dict()
    full_dict = all_matched_df.full_matched.value_counts().to_dict()

    pm_f1, pm_p, pm_r = df2f1(partial_dict), df2precision(partial_dict), df2recall(partial_dict)
    pm_n = partial_dict["tp"]
    fm_f1, fm_p, fm_r = df2f1(full_dict), df2precision(full_dict), df2recall(full_dict)
    fm_n = full_dict["tp"]

    return all_matched_df, pm_f1, pm_p, pm_r, fm_f1, fm_p, fm_r, pm_n, fm_n


def get_report_metrics(typ, report, mapping):
    lst = []
    for k, v in report.items():
        if mapping.get(k, None) is not None:
            lst.append([typ, "f1", mapping[k], v["support"], v["f1-score"]])
    lst.append([typ, "f1", "macro avg", report["macro avg"]["support"], report["macro avg"]["f1-score"]])
    return lst


def compute_span_label_matches(all_matched_df, all_gold_mapping_dict, all_pred_mapping_dict, label_mapping):
    label_mapping = {str(k): str(v) for k,v in label_mapping.items()}
    all_matched_df["gold_adu_label"] = all_matched_df.apply(lambda x: get_value(all_gold_mapping_dict, x["example_id"],
                                                                                x["gold_id"], "adu_label"), 1)
    all_matched_df["pred_adu_label"] = all_matched_df.apply(lambda x: get_value(all_pred_mapping_dict, x["example_id"],
                                                                                x["pred_id"], "adu_label"), 1)

    df_partial = all_matched_df[all_matched_df["partial_matched"] == "tp"].reset_index(drop=True)
    df_full = all_matched_df[all_matched_df["full_matched"] == "tp"].reset_index(drop=True)
    span_label_class_report_partial = classification_report(list(df_partial["gold_adu_label"]),
                                                            list(df_partial["pred_adu_label"]),
                                                            zero_division=0, output_dict=True)
    span_label_class_report_full = classification_report(list(df_full["gold_adu_label"]),
                                                         list(df_full["pred_adu_label"]),
                                                         zero_division=0, output_dict=True)

    span_label_partial_report = get_report_metrics("partial-match:span-label", span_label_class_report_partial,
                                                   label_mapping)
    span_label_full_report = get_report_metrics("full-match:span-label", span_label_class_report_full, label_mapping)

    return all_matched_df, span_label_partial_report + span_label_full_report


def match_relation_triples(all_matched_df, all_gold_rel_dict, all_pred_rel_dict, label_mapping):
    label_mapping = {str(k): str(v) for k, v in label_mapping.items()}
    partial_fn, partial_tp, partial_fp = [], [], []
    full_fn, full_tp, full_fp = [], [], []

    partial_tp_rel, partial_fp_rel = [], []
    full_tp_rel, full_fp_rel = [], []

    for example_id, v in all_gold_rel_dict.items():
        partial_expected, full_expected = [], []
        not_matched_partial, not_matched_full = [], []  # false negatives
        partial_matched, full_matched = [], []  # True positives
        partial_not_matched, full_not_matched = [], []  # False Positives

        partial_matched_rel, full_matched_rel = [], []  # True positives
        partial_not_matched_rel, full_not_matched_rel = [], []  # False Positives

        # Generate mapping from golden span id to predicted span id
        # print(all_matched_df[(all_matched_df["example_id"] == example_id) & (all_matched_df["gold_id"] != -1)])
        tmp_map = all_matched_df[(all_matched_df["example_id"] == example_id) & (all_matched_df["gold_id"] != -1)] \
            [["gold_id", "pred_id", "partial_matched", "full_matched"]].set_index("gold_id").to_dict(orient="index")
        for k2, v2 in tmp_map.items():
            if v2["partial_matched"] == "tp":
                v2["partial_matched"] = True
            else:
                v2["partial_matched"] = False

            if v2["full_matched"] == "tp":
                v2["full_matched"] = True
            else:
                v2["full_matched"] = False
        tmp_map[-1] = {'pred_id': -1, 'partial_matched': True, 'full_matched': True}

        for record in v:
            if tmp_map[record[1][0]]["partial_matched"] and tmp_map[record[1][1]]["partial_matched"]:
                # If span partial matched, then the relation is expected
                partial_expected.append((tmp_map[record[1][0]]["pred_id"], tmp_map[record[1][1]]["pred_id"]))
            else:
                # If span DIDN'T partial matched, then the relation is not existant, hence not matched & False -ve
                not_matched_partial.append([record[2], 0])

            if tmp_map[record[1][0]]["full_matched"] and tmp_map[record[1][1]]["full_matched"]:
                full_expected.append((tmp_map[record[1][0]]["pred_id"], tmp_map[record[1][1]]["pred_id"]))
            else:
                not_matched_full.append([record[2], 0])

        for i in all_pred_rel_dict[example_id]:
            if i[3][0]:  # If valid relationship exists
                if i[1] in partial_expected:  # If relationship is expected then it's a True +ve
                    partial_matched.append([i[1], i[2], i[3][0]])
                else:
                    partial_not_matched.append(i[1])  # If it is NOT expected then it's a False +ve

                if i[1] in full_expected:
                    full_matched.append([i[1], i[2], i[3][0]])
                else:
                    full_not_matched.append(i[1])
            else:  # If relationship DOESN'T exist, it's a False +ve
                partial_not_matched.append(i[1])
                full_not_matched.append(i[1])

            if i[3][0] and i[3][1]:  # If valid relationship exists and relation type is valid
                if i[1] in partial_expected:  # If relationship is expected then it's a True +ve
                    partial_matched_rel.append([i[2], i[2]])
                else:
                    partial_not_matched_rel.append([0, i[2]])  # If it is NOT expected then it's a False +ve

                if i[1] in full_expected:
                    full_matched_rel.append([i[2], i[2]])
                else:
                    full_not_matched_rel.append([0, i[2]])
            else:
                partial_not_matched_rel.append([0, i[2]])
                full_not_matched_rel.append([0, i[2]])

        partial_fn.extend(not_matched_partial)
        partial_tp.extend(partial_matched)
        partial_fp.extend(partial_not_matched)

        full_fn.extend(not_matched_full)
        full_tp.extend(full_matched)
        full_fp.extend(full_not_matched)

        partial_tp_rel.extend(partial_matched_rel)
        partial_fp_rel.extend(partial_not_matched_rel)

        full_tp_rel.extend(full_matched_rel)
        full_fp_rel.extend(full_not_matched_rel)

    partial_rel_f1, full_rel_f1 = get_f1(partial_tp, partial_fp, partial_fn), get_f1(full_tp, full_fp, full_fn)
    partial_rel_pr, full_rel_pr = get_precision(partial_tp, partial_fp), get_precision(full_tp, full_fp)
    partial_rel_rl, full_rel_rl = get_recall(partial_tp, partial_fn), get_recall(full_tp, full_fn)
    partial_rel_n, full_rel_n = len(partial_tp), len(full_tp)

    partial_rel_f1_lbl, full_rel_f1_lbl = get_f1(partial_tp_rel, partial_fp_rel, partial_fn), \
                                            get_f1(full_tp_rel, full_fp_rel, full_fn)
    partial_rel_pr_t, full_rel_pr_lbl = get_precision(partial_tp_rel, partial_fp_rel), \
                                            get_precision(full_tp_rel, full_fp_rel)
    partial_rel_rl_lbl, full_rel_rl_lbl = get_recall(partial_tp_rel, partial_fn), get_recall(full_tp_rel, full_fn)
    partial_rel_n_lbl, full_rel_n_lbl = len(partial_tp_rel), len(full_tp_rel)

    partial_stats_df = pd.DataFrame(partial_fn + partial_tp_rel + partial_fp_rel, columns=["gold", "pred"])
    full_stats_df = pd.DataFrame(full_fn + full_tp_rel + full_fp_rel, columns=["gold", "pred"])
    partial_stats_rpt = classification_report(list(partial_stats_df["gold"]), list(partial_stats_df["pred"]),
                                              zero_division=0, output_dict=True)
    full_stats_rpt = classification_report(list(full_stats_df["gold"]), list(full_stats_df["pred"]), zero_division=0,
                                           output_dict=True)

    # span_label_partial_report = get_report_metrics("full-match:relations-label", partial_stats_rpt, label_mapping)

    lst = [["partial-match:relations", "f1", "True", partial_rel_n, partial_rel_f1],
           ["partial-match:relations", "precision", "True", partial_rel_n, partial_rel_pr],
           ["partial-match:relations", "recall", "True", partial_rel_n, partial_rel_rl],
           ["full-match:relations", "f1", "True", full_rel_n, full_rel_f1],
           ["full-match:relations", "precision", "True", full_rel_n, full_rel_pr],
           ["full-match:relations", "recall", "True", full_rel_n, full_rel_rl],
           ["partial-match:relations-label", "f1", "True", partial_rel_n_lbl, partial_rel_f1_lbl],
           ["partial-match:relations-label", "precision", "True", partial_rel_n_lbl, partial_rel_pr_t],
           ["partial-match:relations-label", "recall", "True", partial_rel_n_lbl, partial_rel_rl_lbl],
           ["full-match:relations-label", "f1", "True", full_rel_n_lbl, full_rel_f1_lbl],
           ["full-match:relations-label", "precision", "True", full_rel_n_lbl, full_rel_pr_lbl],
           ["full-match:relations-label", "recall", "True", full_rel_n_lbl, full_rel_rl_lbl]]

    for k, v in partial_stats_rpt.items():
        if label_mapping.get(k, None) is not None and k in ["5", "6", "7", "8", "9"]:
            lst.append(["partial-match:relations-label", "f1", label_mapping[k], v["support"], v["f1-score"]])

    for k, v in full_stats_rpt.items():
        if label_mapping.get(k, None) is not None and k in ["5", "6", "7", "8", "9"]:
            lst.append(["full-match:relations-label", "f1", label_mapping[k], v["support"], v["f1-score"]])

    return lst


def label_match(gold_toks, pred_toks, match_pct, thresh=0.5):
    if match_pct >= thresh:
        if len(gold_toks) == len(pred_toks) == 0:
            return "tn"
        else:
            return "tp"
    else:
        if len(gold_toks) > len(pred_toks):
            return "fn"
        else:
            return "fp"


def mark_false_negative(orig_label, pct_tuple, thresh=0.5):
    if pct_tuple[0] >= thresh and pct_tuple[1] >= thresh:
        return orig_label
    else:
        if orig_label == "tp":
            return "fn"
        else:
            return orig_label


def change_rel_label(orig_label, pct_tuple, thresh=0.5):
    if pct_tuple[0] >= thresh and pct_tuple[1] >= thresh:
        return orig_label
    else:
        return -1
