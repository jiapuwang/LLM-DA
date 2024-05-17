import json
import os.path
import numpy as np
import torch

from grapher import Grapher
from TADistmult import init_model, get_scores_with_candicates
from params import get_params
from temporal_walk import store_edges
from baseline import baseline_candidates, calculate_obj_distribution
from utils import save_json_data, search_candidates_for_no_cand, filter_candidates, calculate_rank, \
    get_top_k_with_index, load_json_data, get_candicates_by_timestamp, \
    expand_candidates_auto, get_candicates_by_source_with_timestamp, \
    expand_candidates_with_source, expand_candidates_with_relation, \
    merge_scores_optimized, get_candicates_auto, expand_candidates_with_freq_weight, stat_ranks

parsed = get_params()

is_known = parsed.is_known
frequency_type = parsed.calc_type_with_no_cand
interval = parsed.interval
freq_weight = parsed.freq_weight
group_index = parsed.index
group_size = parsed.group_size

dataset = parsed.dataset
candidates_file = parsed.candidates
timestamp_file = parsed.timestamp
dir_path = "./ranked_rules/" + dataset + "/"
dataset_dir = "./datasets/" + dataset + "/"
similiary_dir = "./eva/" + dataset + "/"
mrr_file_path = os.path.join(dataset_dir, 'mrr_filter.json')

data = Grapher(dataset_dir)

num_entities = len(data.id2entity)
num_rels = len(data.relation2id)
test_data = data.test_idx if (parsed.test_data == "test") else data.valid_idx
learn_edges = store_edges(data.train_idx)
obj_dist, rel_obj_dist = calculate_obj_distribution(data.train_idx, learn_edges)

all_candidates = json.load(open(dir_path + candidates_file))
all_candidates = {int(k): v for k, v in all_candidates.items()}
for k in all_candidates:
    all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}

recent_time = max(data.valid_idx[:, 3]) - 0
test_timestamp = set(data.test_idx[:, 3])
test_interval = {}
for idx, timestamp in enumerate(test_timestamp):
    test_interval[timestamp] = timestamp - recent_time

hits_1 = 0
hits_3 = 0
hits_10 = 0
mrr = 0

num_no_cand = 0
num_out_of_scope = 0
num_no_exist = 0
num_has_no_neighbors = 0
num_target_no_exist_neighbor = 0
num_more_than_zero = 0

train_n = data.train_idx
valid_n = data.valid_idx
analysis_bkg_all = np.vstack((data.train_idx, data.valid_idx, data.test_idx))
analysis_bkg_with_train_valid = np.vstack((data.train_idx, data.valid_idx))

if parsed.calc_type_with_no_cand == 'transformer':
    similiary_file_path = os.path.join(similiary_dir, 'similiary.pkl')
    queryid2idx_file_path = os.path.join(similiary_dir, 'queryid2idx.json')
    search_candidates_for_no_cand(data, all_candidates, test_data, similiary_file_path, queryid2idx_file_path)

    top_k = num_entities
    result_dict = get_top_k_with_index(similiary_file_path, top_k)
    queryid2idx = load_json_data(queryid2idx_file_path)

model = None
if parsed.calc_type_with_no_cand == 'TADistmult' or parsed.calc_type_with_no_cand == ['TADistmult_with_recent']:
    model = init_model(os.path.join(dir_path, parsed.checkpoint))

elif parsed.calc_type_with_no_cand == 'TiRGN' or parsed.calc_type_with_no_cand == 'REGCN':

    test_numpy_file_path = os.path.join(dataset_dir, 'predict', 'test.npy')
    test_numpy = np.load(test_numpy_file_path)
    if dataset == 'tigcn_icews18':
       test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)


    score_numpy_file_path = os.path.join(dataset_dir, 'predict', 'score.npy')
    score_numpy = np.load(score_numpy_file_path)

num_in_scope = []
out_scope = []
out_scope_index = []
no_cand = []
num_samples = len(test_data)
print("Evaluating " + candidates_file + ":")

print(f'test_data:{len(test_data)}')
print(f'all_candidates:{len(all_candidates)}')

if (parsed.calc_type_with_no_cand == 'transformer') or (parsed.calc_type_with_no_cand == 'timestamp') or (
        parsed.calc_type_with_no_cand == 'based_source_with_timestamp') or (
        parsed.calc_type_with_no_cand == 'origin'):

    for i in range(num_samples):
        is_frequency = False
        test_query = test_data[i]
        candidates = all_candidates[i]

        if len(candidates) == 0:
            is_frequency = True

        if is_frequency:
            if parsed.calc_type_with_no_cand == 'transformer':
                candidates = result_dict[queryid2idx[str(i)]]
            elif parsed.calc_type_with_no_cand == 'timestamp':
                candidates = get_candicates_by_timestamp(test_query, analysis_bkg_with_train_valid, 70)
            elif parsed.calc_type_with_no_cand == 'based_source_with_timestamp':
                candidates = get_candicates_by_source_with_timestamp(test_query, analysis_bkg_all, interval)
            elif parsed.calc_type_with_no_cand == 'origin':
                candidates = baseline_candidates(test_query[1], learn_edges, obj_dist, rel_obj_dist)

            num_no_cand = num_no_cand + 1

            no_cand.append(test_query.tolist())

        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        if rank == num_entities:
            num_out_of_scope = num_out_of_scope + 1
            out_scope.append(test_query.tolist())
            out_scope_index.extend([i])
        else:
            num_in_scope.extend([rank])

        if rank:
            if rank <= 10:
                hits_10 += 1
                if rank <= 3:
                    hits_3 += 1
                    if rank == 1:
                        hits_1 += 1
            mrr += 1 / rank

data_dict = {}

if parsed.calc_type_with_no_cand == 'TiRGN' or parsed.calc_type_with_no_cand == 'REGCN':
    for i in range(num_samples):
        temp_candidates = {k: 0 for k in range(num_entities)}

        test_query = test_data[i]
        rule_candidates = all_candidates[i]
        rule_candidates = {**temp_candidates, **rule_candidates}

        indices = np.where((test_numpy == test_query).all(axis=1))[0]
        score = score_numpy[indices[0]]
        regcn_candidates = {index: value for index, value in enumerate(score)}

        candidates = {k: freq_weight * regcn_candidates[k] + (1 - freq_weight) * rule_candidates[k] for k in rule_candidates}


        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        if rank == num_entities:
            num_out_of_scope = num_out_of_scope + 1
        else:
            num_in_scope.extend([rank])

        if rank:
            if rank <= 10:
                hits_10 += 1
                if rank <= 3:
                    hits_3 += 1
                    if rank == 1:
                        hits_1 += 1
            mrr += 1 / rank

        if test_query[3] not in data_dict:
            # 如果键不存在，初始化一个空列表
            data_dict[test_query[3]] = []
            # 将数据项添加到对应键的列表中
        data_dict[test_query[3]].append(rank)

ranks_filter = []
sorted_data_dict = {key: data_dict[key] for key in sorted(data_dict)}
for key, value in sorted_data_dict.items():
    tensor_value = torch.LongTensor(value)
    ranks_filter.append(tensor_value)

mrr_filter_list = []
for idx in range(len(ranks_filter)):
    mrr_filter = stat_ranks(ranks_filter[idx:(idx+1)], "filter_ent")
    mrr_filter_list.append(round(mrr_filter.tolist(), 4))


save_json_data(mrr_filter_list, mrr_file_path)

if (parsed.calc_type_with_no_cand == 'fusion') or (parsed.calc_type_with_no_cand == 'fusion_with_weight') or (
        parsed.calc_type_with_no_cand == 'fusion_with_source') or (
        parsed.calc_type_with_no_cand == 'fusion_with_relation'):

    for i in range(num_samples):
        test_query = test_data[i]
        candidates = all_candidates[i]
        is_has_neighbor = True
        exist_in_neighbors = True
        if parsed.calc_type_with_no_cand == 'fusion':
            candidates, is_exist = expand_candidates_auto(candidates, analysis_bkg_with_train_valid, 70, test_query)
        elif parsed.calc_type_with_no_cand == 'fusion_with_weight':
            candidates, is_exist = expand_candidates_with_freq_weight(candidates, analysis_bkg_all,
                                                                      interval, test_query, parsed.freq_weight)
        elif parsed.calc_type_with_no_cand == 'fusion_with_source':
            candidates, is_exist, is_has_neighbor, exist_in_neighbors = expand_candidates_with_source(candidates,
                                                                                                      analysis_bkg_with_train_valid,
                                                                                                      parsed[
                                                                                                          "interval"],
                                                                                                      test_query,
                                                                                                      parsed[
                                                                                                          'freq_weight'])
        elif parsed.calc_type_with_no_cand == 'fusion_with_relation':
            candidates, is_exist, is_has_neighbor, exist_in_neighbors = expand_candidates_with_relation(candidates,
                                                                                                        analysis_bkg_all,
                                                                                                        parsed[
                                                                                                            "interval"],
                                                                                                        test_query,
                                                                                                        parsed[
                                                                                                            'freq_weight'])

        if is_exist is False:
            num_no_cand = num_no_cand + 1

        if is_has_neighbor is False:
            num_has_no_neighbors = num_has_no_neighbors + 1

        if exist_in_neighbors is False:
            num_target_no_exist_neighbor = num_target_no_exist_neighbor + 1

        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        if rank == num_entities:
            num_out_of_scope = num_out_of_scope + 1
        else:
            num_in_scope.extend([rank])

        if rank:
            if rank <= 10:
                hits_10 += 1
                if rank <= 3:
                    hits_3 += 1
                    if rank == 1:
                        hits_1 += 1
            mrr += 1 / rank

if (parsed.calc_type_with_no_cand == 'TADistmult') or (
        parsed.calc_type_with_no_cand == 'TADistmult_with_recent'):

    for i in range(num_samples):
        test_query = test_data[i]
        rule_candidates = all_candidates[i]

        if len(rule_candidates) == 0:
            num_no_cand = num_no_cand + 1

        interval = test_interval[test_query[3]]
        candicates_list = get_candicates_auto(test_query[3], interval, analysis_bkg_with_train_valid)

        if parsed.calc_type_with_no_cand == 'TADistmult' and len(candicates_list) > 0:
            dist = get_scores_with_candicates(model, test_query, data.id2ts, candicates_list)
            candidates = merge_scores_optimized(rule_candidates, dist, parsed['model_weight'])
        elif parsed.calc_type_with_no_cand == 'TADistmult_with_recent' and len(candicates_list) > 0:
            pass
        else:
            candidates = rule_candidates

        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        if rank == num_entities:
            num_out_of_scope = num_out_of_scope + 1
        else:
            num_in_scope.extend([rank])

        if rank:
            if rank <= 10:
                hits_10 += 1
                if rank <= 3:
                    hits_3 += 1
                    if rank == 1:
                        hits_1 += 1
            mrr += 1 / rank

hits_1 /= num_samples
hits_3 /= num_samples
hits_10 /= num_samples
mrr /= num_samples

print('Num of more than zero:', num_more_than_zero)
print('Num of no cand:', num_no_cand)
print('num_out_of_scope:', num_out_of_scope)
print("Hits@1: ", round(hits_1, 6))
print("Hits@3: ", round(hits_3, 6))
print("Hits@10: ", round(hits_10, 6))
print("MRR: ", round(mrr, 6))

filename = candidates_file[:-5] + "_eval.txt"
with open(dir_path + filename, "w", encoding="utf-8") as fout:
    fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
    fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
    fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
    fout.write("MRR: " + str(round(mrr, 6)))
