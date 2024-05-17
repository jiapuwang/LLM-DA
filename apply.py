import json
import os.path
import time
import argparse
import itertools
import numpy as np
import torch
import gc
import math
from joblib import Parallel, delayed
import multiprocessing as mp

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12, score_13, score_14
from utils import get_win_subgraph, save_json_data, load_json_data


# from sentence_transformers import SentenceTransformer
# # 加载预训练的句子嵌入模型
# model = SentenceTransformer('bert-base-nli-mean-tokens')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--rule_files", "-f", default="", type=str)
parser.add_argument("--confidence_type", default="TLogic", type=str,
                    choices=['TLogic', 'LLM', 'And', 'Or'])
parser.add_argument("--weight", default=0.0, type=float)
parser.add_argument("--weight_0", default=0.5, type=float)
parser.add_argument("--min_conf", default=0.01, type=float)
parser.add_argument("--coor_weight", default=0, type=float)
parser.add_argument("--lmbda", default=0.1, type=float)
parser.add_argument("--bgkg", default="all", type=str,
                      choices=['all', 'train', 'valid', 'test', 'train_valid', 'train_test', 'valid_test'])
parser.add_argument("--score_type", default="noisy-or", type=str,
                      choices=['noisy-or', 'sum', 'mean', 'min', 'max'])
parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
parser.add_argument("--is_sorted", default='no', type=str_to_bool)
parser.add_argument("--is_return_timestamp", default='no', type=str_to_bool)
parser.add_argument("--num_batch", default=0, type=int)

parser.add_argument('--evaluation_type', type=str,  choices=['transformer', 'origin', 'end'])

parser.add_argument("--index", default=0, type=int)
parser.add_argument("--group_size", default=0, type=int)
parser.add_argument("--win_start", default=0, type=int)
parser.add_argument("--is_sampled", default='no', type=str_to_bool)
parser.add_argument("--is_rule_priority", default='no', type=str_to_bool)

parsed = vars(parser.parse_args())

win_start = parsed["win_start"]
group_size = parsed["group_size"]
dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "./datasets/" + dataset + "/"
dir_path = "./ranked_rules/" + dataset + "/"
data = Grapher(dataset_dir, parsed)
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)

if parsed['evaluation_type'] == 'origin':
    score_func = score_12
    parsed['coor_weight'] = 0
elif parsed['evaluation_type'] == 'transformer':
    score_func = score_13
elif parsed['evaluation_type'] == 'end':
    score_func = score_14
    parsed['coor_weight'] = 0

unique_test_data, inverse = np.unique(test_data[:,[0,1,3]],axis=0, return_inverse=True)
unique_test_shape = unique_test_data.shape

# 创建一列全是0的数据，长度为N
zeros_column = np.zeros(unique_test_shape[0])

# 在第三列（索引为2）插入全0的数据列
# axis=1 表示按列插入
new_test_data = np.insert(unique_test_data, 2, zeros_column, axis=1)


new_rules_dict, sort_rules_dict = ra.filter_rules(
    rules_dict, min_conf=parsed['min_conf'], min_body_supp=2, rule_lengths=rule_lengths, confidence_type=parsed["confidence_type"]
)

if parsed["is_sorted"] is False:
    rules_dict = new_rules_dict
else:
    rules_dict = sort_rules_dict

print("Rules statistics after pruning:")
rules_statistics(rules_dict)
learn_edges = store_edges(data.train_idx)


windown_subgraph = get_win_subgraph(test_data, data, learn_edges, window, win_start=win_start)


# It is possible to specify a list of list of arguments for tuning
args = [[parsed['lmbda'], parsed['weight_0'], parsed['confidence_type'], parsed['weight'], parsed['min_conf'], parsed['coor_weight']]]


def process_data_with_timeout(mp_args):
    # 这里是处理数据的函数，根据索引处理数据
    # print(f"{process_id}_{start_index}_{end_index-1}")
    process_id, start_index, end_index, parsed = mp_args

    return apply_rules_with_index(process_id, start_index, end_index,parsed)

def calculate_correlation(test_query, rule, data):
    # head_entity = datatest_query[0]
    query_head = data.id2entity[test_query[0]]
    query_relation = data.id2relation[test_query[1]]

    boby_rels_name = []
    for body_rel_id in rule['body_rels']:
        boby_rels_name.append(data.id2relation[body_rel_id])

    body_name = ' '.join(boby_rels_name)

    # 定义两个句子
    sentences = [
        f'{query_head} {query_relation}',
        body_name
    ]

    # 计算句子嵌入
    embeddings = model.encode(sentences)

    embeddings = np.array(embeddings)

    # 使用 PyTorch 计算余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0].reshape(1, -1)),
        torch.tensor(embeddings[1].reshape(1, -1))
    )

    return round(float(cosine_similarity),4)


def apply_rules_with_index(process_id, start_index, end_index, parsed):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", process_id, "...")
    torch.cuda.set_device(parsed['gpu'])
    all_candidates = [dict() for _ in range(len(args))]
    all_timestamp = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    cur_ts = new_test_data[start_index][3]
    edges = windown_subgraph[cur_ts]
    # edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    for j in range(start_index, end_index):
        # j_start = time.time()

        test_query = new_test_data[j]
        cands_dict = [dict() for _ in range(len(args))]
        timestamp_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            # edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)
            edges = windown_subgraph[cur_ts]

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, test_query)

                if parsed['evaluation_type'] == 'transformer':
                   corre = calculate_correlation(test_query, rule, data)
                else:
                   corre = 0

                if 0 not in [len(x) for x in walk_edges]:
                    if parsed['evaluation_type'] != 'end':
                       rule_walks = ra.get_walks(rule, walk_edges, parsed["is_relax_time"])
                    else:
                       rule_walks = ra.get_walks_end(rule, walk_edges, parsed["is_relax_time"])

                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict, timestamp_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                            corre,
                            parsed['is_return_timestamp'],
                            parsed['evaluation_type'],
                            timestamp_dict,
                        )
                        for s in dicts_idx:
                            if parsed['is_return_timestamp'] is True:
                                for x in cands_dict[s].keys():
                                    # 获取 cands_dict 中排序后的列表及其索引
                                    sorted_indices, sorted_cands = zip(
                                        *sorted(enumerate(cands_dict[s][x]), key=lambda pair: pair[1],
                                                reverse=True))

                                    sorted_indices = list(sorted_indices)
                                    sorted_cands = list(sorted_cands)

                                    # 重新排列 timestamp_dict 中对应的列表
                                    timestamp_dict[s][x] = [timestamp_dict[s][x][i] for i in sorted_indices]

                                    # 更新 cands_dict 中的列表为排序后的列表
                                    cands_dict[s][x] = sorted_cands

                                # 对 cands_dict[s] 进行排序
                                sorted_items = sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True)

                                # 更新 cands_dict[s]
                                cands_dict[s] = dict(sorted_items)

                                # 使用相同的顺序更新 timestamp_dict[s]
                                timestamp_dict[s] = {k: timestamp_dict[s][k] for k, _ in sorted_items}


                            else:
                                cands_dict[s] = {
                                    x: sorted(cands_dict[s][x], reverse=True)
                                    for x in cands_dict[s].keys()
                                }

                                cands_dict[s] = dict(sorted(cands_dict[s].items(),key=lambda item: item[1], reverse=True))


                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)

                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    if parsed["score_type"] == 'noisy-or':
                        scores = list(
                            map(
                                lambda x: 1 - np.product(1 - np.array(x)),
                                cands_dict[s].values(),
                            )
                        )
                    elif  parsed['score_type'] == 'sum':
                        scores = [sum(sublist) for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'mean':
                        scores = [sum(sublist) / len(sublist) for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'min':
                        scores = [min(sublist)  for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'max':
                        scores = [max(sublist)  for sublist in list(cands_dict[s].values())]

                    scores = [np.float64(x) for x in scores]
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )

                    temp_timestamp = {}
                    if parsed['is_return_timestamp'] is True:
                        for time_key, time_value in timestamp_dict[s].items():
                            temp_timestamp[time_key] = max(time_value)
                        all_timestamp[s][j] = temp_timestamp

                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()
                    if parsed['is_return_timestamp'] is True:
                       all_timestamp[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()
                if parsed['is_return_timestamp'] is True:
                    all_timestamp[s][j] = dict()

        if not (j - start_index + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    process_id, j - start_index + 1, end_index - start_index, it_time
                )
            )
            it_start = time.time()

        # j_end = time.time()
        # print(f'{round(j_end - j_start, 6)} sec')

    return all_candidates, no_cands_counter, all_timestamp

def apply_rules(i, num_queries, parsed):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", i, "...")
    torch.cuda.set_device(parsed['gpu'])
    all_candidates = [dict() for _ in range(len(args))]
    all_timestamp = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    if i < num_processes - 1:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))
        # test_queries_idx = range(i * num_queries, 2)

    if len(test_queries_idx) == 0:
        return  all_candidates, no_cands_counter, all_timestamp

    cur_ts = test_data[test_queries_idx[0]][3]
    edges = windown_subgraph[cur_ts]

    it_start = time.time()
    for j in test_queries_idx:
        # j_start = time.time()

        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]
        timestamp_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = windown_subgraph[cur_ts]

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule_idx, rule in enumerate(rules_dict[test_query[1]]):
                walk_edges = ra.match_body_relations(rule, edges, test_query, is_sample=parsed["is_sampled"])

                if parsed['evaluation_type'] == 'transformer':
                   corre = calculate_correlation(test_query, rule, data)
                else:
                   corre = 0

                if 0 not in [len(x) for x in walk_edges]:
                    if parsed['evaluation_type'] != 'end':
                       rule_walks = ra.get_walks(rule, walk_edges, parsed["is_relax_time"])
                    else:
                       rule_walks = ra.get_walks_end(rule, walk_edges, parsed["is_relax_time"])

                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict, timestamp_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                            corre,
                            parsed['is_return_timestamp'],
                            parsed['evaluation_type'],
                            timestamp_dict,
                        )
                        for s in dicts_idx:
                            if parsed['is_return_timestamp'] is True:
                                for x in cands_dict[s].keys():
                                    # 获取 cands_dict 中排序后的列表及其索引
                                    sorted_indices, sorted_cands = zip(
                                        *sorted(enumerate(cands_dict[s][x]), key=lambda pair: pair[1],
                                                reverse=True))

                                    sorted_indices = list(sorted_indices)
                                    sorted_cands = list(sorted_cands)

                                    # 重新排列 timestamp_dict 中对应的列表
                                    timestamp_dict[s][x] = [timestamp_dict[s][x][i] for i in sorted_indices]

                                    # 更新 cands_dict 中的列表为排序后的列表
                                    cands_dict[s][x] = sorted_cands

                                # 对 cands_dict[s] 进行排序
                                sorted_items = sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True)

                                # 更新 cands_dict[s]
                                cands_dict[s] = dict(sorted_items)

                                # 使用相同的顺序更新 timestamp_dict[s]
                                timestamp_dict[s] = {k: timestamp_dict[s][k] for k, _ in sorted_items}


                            else:
                                cands_dict[s] = {
                                    x: sorted(cands_dict[s][x], reverse=True)
                                    for x in cands_dict[s].keys()
                                }

                                cands_dict[s] = dict(sorted(cands_dict[s].items(),key=lambda item: item[1], reverse=True))

                            if parsed['is_rule_priority'] is False:
                                top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                                unique_scores = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores)
                                )
                                if len(unique_scores) >= top_k:
                                    dicts_idx.remove(s)
                            else:
                                if rule_idx >= top_k:
                                    dicts_idx.remove(s)

                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    if parsed["score_type"] == 'noisy-or':
                        scores = list(
                            map(
                                lambda x: 1 - np.product(1 - np.array(x)),
                                cands_dict[s].values(),
                            )
                        )
                    elif  parsed['score_type'] == 'sum':
                        scores = [sum(sublist) for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'mean':
                        scores = [sum(sublist) / len(sublist) for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'min':
                        scores = [min(sublist)  for sublist in list(cands_dict[s].values())]
                    elif parsed['score_type'] == 'max':
                        scores = [max(sublist)  for sublist in list(cands_dict[s].values())]

                    scores = [np.float64(x) for x in scores]
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )

                    temp_timestamp = {}
                    if parsed['is_return_timestamp'] is True:
                        for time_key, time_value in timestamp_dict[s].items():
                            temp_timestamp[time_key] = max(time_value)
                        all_timestamp[s][j] = temp_timestamp

                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()
                    if parsed['is_return_timestamp'] is True:
                       all_timestamp[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()
                if parsed['is_return_timestamp'] is True:
                    all_timestamp[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

        # j_end = time.time()
        # print(f'{round(j_end - j_start, 6)} sec')

    return all_candidates, no_cands_counter, all_timestamp

final_all_candidates = [dict() for _ in range(len(args))]
final_all_timestamp = [dict() for _ in range(len(args))]
final_no_cands_counter = 0

start = time.time()

if parsed['num_batch'] == 0:
    num_queries = len(test_data) // num_processes
    # num_queries = 1 // num_processes
    output = Parallel(n_jobs=num_processes)(
        delayed(apply_rules)(i, num_queries, parsed) for i in range(num_processes)
    )

    for s in range(len(args)):
        for i in range(num_processes):
            final_all_candidates[s].update(output[i][0][s])
            output[i][0][s].clear()

            final_all_timestamp[s].update(output[i][2][s])
            output[i][2][s].clear()

    for i in range(num_processes):
        final_no_cands_counter += output[i][1]
else:
    N = parsed['num_batch']
    batch_size = math.ceil(len(new_test_data) / N)
    # N = 2
    # batch_size = 11 // N
    output = []

    error_list = []
    # for i in range(N):
    #     start_index = i * batch_size
    #     end_index = 11 if i == N - 1 else min((i + 1) * batch_size, len(new_test_data))

    for i in range(N):
        start_index = i * batch_size
        end_index = len(new_test_data) if i == N - 1 else min((i + 1) * batch_size, len(new_test_data))
        mp_start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            jobs = []
            for j in range(num_processes):
                mp_args = (
                    j,
                    j * (batch_size // num_processes) + start_index,
                    ((j + 1) * (batch_size // num_processes) + start_index) if j < num_processes - 1 else end_index,
                    parsed
                )
                job = pool.apply_async(process_data_with_timeout, (mp_args,))
                jobs.append((job, j))  # 保存 job 和对应的进程编号

            try:
                # 尝试获取每个任务的结果
                for job, process_id in jobs:
                    try:
                        result = job.get(timeout=1500)  # 设置合理的超时时间
                        output.append(result)
                    except mp.TimeoutError:
                        error_list.append([start_index, end_index])
                        print(f"Task execution timed out, process {process_id} may be blocked.")
                        # 在这里处理超时的情况，比如重新提交任务或记录日志等
                        pool.terminate()
                        break
                    except Exception as e:
                        error_list.append([start_index, end_index])
                        print(f"Task execution error: {e}")
                        # 在这里处理其它异常
                        pool.terminate()
                        break  # 退出循环
            finally:
                # 确保池最终关闭
                pool.close()  # 关闭进程池，不再接受新的任务
                pool.join()  # 等待所有子进程结束

        mp_end_time = time.time()
        print(f"The time spent on each batch:{mp_end_time - mp_start_time} sec")

    print(f'error index:{error_list}')
    save_json_data(error_list, os.path.join(dir_path, 'error_list.json'))

    for s in range(len(args)):
        for idx in range(len(output)):
            final_all_candidates[s].update(output[idx][0][s])
            output[idx][0][s].clear()

            final_all_timestamp[s].update(output[i][2][s])
            output[i][2][s].clear()

    for idx in range(len(output)):
        final_no_cands_counter += output[idx][1]

end = time.time()
total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")

for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    score_func_str = f'{score_func_str}_rule_{parsed["is_rule_priority"]}_top_{parsed["top_k"]}_et_{parsed["evaluation_type"]}_sorted_{parsed["is_sorted"]}_index_{parsed["index"]}_bgkg_{parsed["bgkg"]}_start_{parsed["win_start"]}_relax_{parsed["is_relax_time"]}_sample_{parsed["is_sampled"]}'
    ra.save_candidates(
        rules_file,
        dir_path,
        final_all_candidates[s],
        rule_lengths,
        window,
        score_func_str,
        final_all_timestamp[s],
    )
