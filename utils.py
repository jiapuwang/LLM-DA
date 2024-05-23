import random
import torch
import os
import numpy as np
import re
import json
import scipy.sparse as ssp
import shutil
import pickle
import argparse
import math
from collections import Counter
import pandas as pd

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import rule_application as ra
from baseline import baseline_candidates


# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")
# os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'


def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length * "#" + msg + length * "#")


def camel_to_normal(camel_string):
    # 使用正则表达式将驼峰字符串转换为正常字符串
    normal_string = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_string).lower()
    return normal_string


def clean_symbol_in_rel(rel):
    '''
    clean symbol in relation

    Args:
        rel (str): relation name
    '''

    rel = rel.strip("_")  # Remove heading
    # Replace inv_ with inverse
    # rel = rel.replace("inv_", "inverse ")
    if "/" in rel:
        if "inverse" in rel:
            rel = rel.replace("inverse ", "")
            rel = "inverse " + fb15k_rel_map[rel]
        else:
            rel = fb15k_rel_map[rel]
    # WN-18RR
    elif "_" in rel:
        rel = rel.replace("_", " ")  # Replace _ with space
    # UMLS
    elif "&" in rel:
        rel = rel.replace("&", " ")  # Replace & with space
    # YAGO 
    else:
        rel = camel_to_normal(rel)
    return rel


def query(message, llm_model):
    '''
    Query ChatGPT API
    :param message:·
    :return:
    '''

    return llm_model.generate_sentence(message)


def unknown_check_prompt_length(prompt, condicate_list, return_rules, model):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_condicate = ";".join(condicate_list)
    return_rules = return_rules.format(candidate_rels=all_condicate)
    all_tokens = prompt + return_rules
    maximun_token = model.maximum_token
    if model.token_len(all_tokens) < maximun_token:
        return all_condicate
    else:
        # Shuffle the paths
        random.shuffle(condicate_list)
        new_list_candcate = []
        # check the length of the prompt
        for p in condicate_list:
            tmp_all_paths = ";".join(new_list_candcate + [p])
            return_rules = return_rules.format(candidate_rels=tmp_all_paths)
            tmp_all_tokens = prompt + return_rules
            if model.token_len(tmp_all_tokens) > maximun_token:
                return ";".join(new_list_candcate)
            new_list_candcate.append(p)

def iteration_check_prompt_length(prompt, condicate_list, return_rules, model):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_condicate = ";".join(condicate_list)
    return_rules = return_rules.format(candidate_rels=all_condicate)
    all_tokens = prompt + return_rules
    maximun_token = model.maximum_token
    if model.token_len(all_tokens) < maximun_token:
        return all_condicate
    else:
        # Shuffle the paths
        random.shuffle(condicate_list)
        new_list_candcate = []
        # check the length of the prompt
        for p in condicate_list:
            tmp_all_paths = ";".join(new_list_candcate + [p])
            return_rules = return_rules.format(candidate_rels=tmp_all_paths)
            tmp_all_tokens = prompt + return_rules
            if model.token_len(tmp_all_tokens) > maximun_token:
                return ";".join(new_list_candcate)
            new_list_candcate.append(p)

def check_prompt_length(prompt, list_of_paths, model):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    maximun_token = model.maximum_token
    if model.token_len(all_tokens) < maximun_token:
        return all_paths
    else:
        # Shuffle the paths
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        # check the length of the prompt
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if model.token_len(tmp_all_tokens) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)


def num_tokens_from_message(path_string, model):
    """Returns the number of tokens used by a list of messages."""
    messages = [{"role": "user", "content": path_string}]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-3.5-turbo", 'gpt-3.5-turbo-16k']:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    elif model == "gpt-4":
        tokens_per_message = 3
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    tokenizer = tiktoken.encoding_for_model(model)
    return num_tokens_limit, tokenizer


def split_path_list(path_list, token_limit, model):
    """
    Split the path list into several lists, each list can be fed into the model.
    """
    output_list = []
    current_list = []
    current_token_count = 4

    for path in path_list:
        path += '\n'
        path_token_count = num_tokens_from_message(path, model) - 4
        if current_token_count + path_token_count > token_limit:  # If the path makes the current list exceed the token limit
            output_list.append(current_list)
            current_list = [path]  # Start a new list.
            current_token_count = path_token_count + 4
        else:  # The new path fits into the current list without exceeding the limit
            current_list.append(path)  # Just add it there.
            current_token_count += path_token_count
    # Add the last list of tokens, if it's non-empty.
    if current_list:  # The last list not exceed the limit but no more paths
        output_list.append(current_list)
    return output_list


def shuffle_split_path_list(path_content_list, prompt_len, model):
    """
    First shuffle the path_content list, then split the path list into a list of several lists
    Each list can be directly fed into the model
    """
    token_limitation, tokenizer = get_token_limit(model)  # Get input token limitation for current model
    token_limitation -= prompt_len + 4  # minus prompt length for path length
    all_path_content = '\n'.join(path_content_list)
    token_num_all_path = num_tokens_from_message(all_path_content, model)
    random.shuffle(path_content_list)
    if token_num_all_path > token_limitation:
        list_of_paths = split_path_list(path_content_list, token_limitation, model)
    else:
        list_of_paths = [[path + '\n' for path in path_content_list]]
    return list_of_paths


def ill_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


def harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


def balance_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    if pred[ent2idx[q_t]] != 0:
        pred_ranks = np.argsort(pred)[::-1]

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t != ent2idx[q_t]]

        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth:
                filtered_ranks.append(idx)

        rank = filtered_ranks.index(ent2idx[q_t]) + 1
    else:
        truth = gt[(q_h, q_r)]

        filtered_pred = []

        for i in range(len(pred)):
            if i not in truth:
                filtered_pred.append(pred[i])
        n_non_zero = np.count_nonzero(filtered_pred)
        rank = n_non_zero + 1
    return rank


def random_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    truth.append(ent2idx[q_t])
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            if (pred[idx] == pred[ent2idx[q_t]]) and (np.random.uniform() < 0.5):
                filtered_ranks.append(idx)
            else:
                filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


def load_json_data(file_path, default=None):
    """从文件加载JSON数据，如果文件不存在则返回默认值。"""
    try:
        if os.path.exists(file_path):
            print(f"Use cache from: {file_path}")
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            print(f"File not found: {file_path}")
            # 在这里添加你想要执行的操作，比如创建一个空的JSON对象并返回
            return default
    except Exception as e:
        print(f"Error loading JSON data from {file_path}: {e}")
    return default


def save_json_data(data, file_path):
    """将数据保存到JSON文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data has been converted to JSON and saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON data to {file_path}: {e}")


def write_to_file(content, path):
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(content)

def stat_ranks(rank_list, method='filter'):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr


def construct_adjacency_list_and_index(triples, relation_id_list, num_entities):
    """构造邻接矩阵列表"""
    adj_list = []
    relation_index = {}
    triples = np.array(triples)
    for i, relation_id in enumerate(relation_id_list):
        idx = np.argwhere(triples[:, 1] == relation_id)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triples[:, 0][idx].squeeze(1), triples[:, 2][idx].squeeze(1))),
                                       shape=(num_entities, num_entities)))
        relation_index[relation_id] = i

    return adj_list, relation_index


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def extract_neighbors(adj, roots, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def get_subgraph_nodes(root1_nei, root2_nei, ind, kind):
    # 根据'kind'获取子图节点
    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    if ind[0] in subgraph_nei_nodes_int:
        subgraph_nei_nodes_int.remove(ind[0])
    if ind[1] in subgraph_nei_nodes_int:
        subgraph_nei_nodes_int.remove(ind[1])
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    if ind[0] in subgraph_nei_nodes_un:
        subgraph_nei_nodes_un.remove(ind[0])
    if ind[1] in subgraph_nei_nodes_un:
        subgraph_nei_nodes_un.remove(ind[1])
    if kind == "intersection":
        subgraph_nodes = set(list(ind) + list(subgraph_nei_nodes_int))
    else:
        subgraph_nodes = set(list(ind) + list(subgraph_nei_nodes_un))

    return list(subgraph_nodes)


def subgraph_extraction_labeling(ind, A_list, kind, h=1, max_nodes_per_hop=None):
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = extract_neighbors(A_incidence, set([ind[0]]), h, max_nodes_per_hop)
    root2_nei = extract_neighbors(A_incidence, set([ind[1]]), h, max_nodes_per_hop)

    temp_entity_id = {}
    subgraph_nodes = get_subgraph_nodes(root1_nei, root2_nei, ind, kind)
    for idx, entity in enumerate(subgraph_nodes):
        temp_entity_id[idx] = entity

    subject_object_list = []
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    nonzero_row_indices, nonzero_col_indices = incidence_matrix(subgraph).nonzero()
    for idx in range(len(nonzero_row_indices)):
        subject = temp_entity_id[nonzero_row_indices[idx]]
        object = temp_entity_id[nonzero_col_indices[idx]]
        subject_object_list.append([subject, object])

    return subject_object_list


def copy_folder_contents(source_folder, destination_folder):
    # 创建目标文件夹，如果它不存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制每个文件和子文件夹
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)
        else:
            shutil.copy2(source_item, destination_item)

    print(f"Contents of '{source_folder}' have been copied to '{destination_folder}'")



def search_candidates_for_no_cand(data, all_candidates, test_data, similiary_file_path, queryid2idx_file_path):
    query_id_without_candidates = []
    query_name_withoud_candidates = []
    for key, value in all_candidates.items():
        if len(value) == 0:
            query_id_without_candidates.extend([key])

    candidates_list = list(data.entity2id.keys())

    queryid2idx = {}
    for idx, query_id in enumerate(query_id_without_candidates):
        query_name = test_data[query_id]
        head_name = data.id2entity[query_name[0]]
        rel_name = data.id2relation[query_name[1]]

        query_head_rel = f'{head_name} {rel_name}'
        query_name_withoud_candidates.append(query_head_rel)
        queryid2idx[query_id] = idx


    # 加载预训练的模型
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # 定义句子
    sentences_A = query_name_withoud_candidates
    sentences_B = candidates_list

    # 使用模型为句子编码
    embeddings_A = model.encode(sentences_A)
    embeddings_B = model.encode(sentences_B)

    # 计算句子之间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)

    # 打印相似度矩阵
    with open(similiary_file_path, 'wb') as f:
        pickle.dump(similarity_matrix, f)

    save_json_data(queryid2idx, queryid2idx_file_path)


def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates


def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = num_entities
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = list(candidates.values())
        all_confs = sorted(all_confs, reverse=True)
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]

        try:

            if setting == "average":
                rank = (ranks[0] + ranks[-1]) // 2 + 1
            elif setting == "best":
                rank = ranks[0] + 1
            elif setting == "worst":
                rank = ranks[-1] + 1
        except Exception as e:
            ranks

    return rank

def get_top_k_with_index(similarity_file_path, top_k):
    with open(similarity_file_path, 'rb') as f:
        loaded_arr = pickle.load(f)
    # 获取每一行的最大 top_k 个值的索引
    top_k_indices = np.argsort(loaded_arr, axis=1)[:, -top_k:][:, ::-1]
    # 利用高级索引获取对应的值
    rows = np.arange(loaded_arr.shape[0])[:, None]  # 生成行索引
    top_k_values = loaded_arr[rows, top_k_indices]

    # 将结果以字典形式存储
    result_dict = {}
    for i in range(loaded_arr.shape[0]):
        result_dict[i] = {index: value for index, value in zip(top_k_indices[i], top_k_values[i])}

    return result_dict

def dataset_analysis(data, all_candidates):
    query_id_without_candidates = []
    for key, value in all_candidates.items():
        if len(value) == 0:
            query_id_without_candidates.extend([key])

    test_ = data.test_idx
    train_ = data.train_idx
    valid_ = data.valid_idx

    test_n = np.array(test_)
    test_n = test_n[query_id_without_candidates]

    train_n = np.array(train_)
    valid_n = np.array(valid_)

    analysis_bkg = np.vstack((valid_n, train_n))

    temp_dict = {}
    num_miss = 0
    for head_id, rel_id, answer_id, timestamp_id in test_n:
        temp_key = f'{head_id}_{rel_id}_{answer_id}_{timestamp_id}'
        mask = analysis_bkg[:, 2] == answer_id
        answer_timestamp = analysis_bkg[mask][:, 3]

        try:
          temp_dict[temp_key] = int(timestamp_id - max(answer_timestamp))
        except Exception as e:
            num_miss = num_miss + 1

    print(num_miss)

    interval_list = list(temp_dict.values())

    save_json_data(interval_list, '/mnt/sda/sk/project/LLM_Temporal/eva/icews14/interval_list.json')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_candicates_by_timestamp(test_query, bkg, interval):
    timestamp_id = test_query[3]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    candidates = select_canicates_based_timestamp(time_filtered, timestamp_id)
    return candidates


def select_canicates_based_timestamp_normal(time_filtered, target_timestamp_id, min_score, max_score):
    # 按照第四列（时间戳）降序排列
    time_filtered_sorted = time_filtered[time_filtered[:, 3].argsort()[::-1]]

    # 获取每个目标实体的第一个（最大）时间戳的索引
    unique_targets, indices = np.unique(time_filtered_sorted[:, 2], return_index=True)
    max_timestamps = time_filtered_sorted[indices, 3]

    array = 1.0 / (target_timestamp_id - max_timestamps)
    a = min_score
    b = max_score

    # 找到数组的最小值和最大值
    min_val = min(array)
    max_val = max(array)

    # 使用 Min-Max 归一化将数组归一化到 [a, b] 区间
    normalized_array = [
        a + (x - min_val) * (b - a) / (max_val - min_val) if max_val != min_val else a
        for x in array
    ]

    # 创建一个字典，将每个目标实体映射到其最大时间戳
    candidates = dict(zip(unique_targets, normalized_array))

    return candidates

def select_canicates_based_timestamp(time_filtered, target_timestamp_id):
    # 按照第四列（时间戳）降序排列
    time_filtered_sorted = time_filtered[time_filtered[:, 3].argsort()[::-1]]

    # 获取每个目标实体的第一个（最大）时间戳的索引
    unique_targets, indices = np.unique(time_filtered_sorted[:, 2], return_index=True)
    max_timestamps = time_filtered_sorted[indices, 3]

    # 创建一个字典，将每个目标实体映射到其最大时间戳
    candidates = dict(zip(unique_targets, 1.0 / (target_timestamp_id - max_timestamps)))

    return candidates


def get_candicates_by_source_with_timestamp(test_query, bkg, interval):
    source_id = test_query[0]
    timestamp_id = test_query[3]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]

    # 进行源实体的匹配
    target_mask = time_filtered[:, 0] == source_id
    source_mask = time_filtered[:, 2] == source_id

    candicates_target = time_filtered[target_mask][:, 2]
    candicates_source = time_filtered[source_mask][:, 0]

    # 合并候选实体数组
    candicates = np.hstack((candicates_target, candicates_source))
    if len(candicates) == 0:
        candicates = select_canicates_based_timestamp(time_filtered, timestamp_id)
    else:
        counter = Counter(candicates)
        candicates = dict(counter)

    return candicates


def expand_candidates(candidates, data, interval, target_timestamp_id):
    min_timestamp_id = target_timestamp_id - interval

    train_n = np.array(data.train_idx)
    valid_n = np.array(data.valid_idx)
    analysis_bkg = np.vstack((valid_n, train_n))

    mask = analysis_bkg[:, 3] >= min_timestamp_id
    temp_dict = {}
    for bkg_target_id, bkg_timestamp_id in analysis_bkg[mask][:, 2:4]:
        if bkg_target_id in candidates:
            continue

        if bkg_target_id not in temp_dict:
            temp_dict[bkg_target_id] = bkg_timestamp_id

        curr_timestamp_id = temp_dict[bkg_target_id]
        if curr_timestamp_id < bkg_timestamp_id:
            temp_dict[bkg_target_id] = bkg_timestamp_id


    pro = 0.0
    for value in temp_dict.values():
        pro = pro + value

    temp_temp_dict = {}
    for key, value in temp_dict.items():
        temp_temp_dict[key] = value / pro

    if len(temp_temp_dict.values()) == 0:
        return candidates

    X_min = min(list(temp_temp_dict.values()))
    X_max = max(list(temp_temp_dict.values()))

    if X_max == X_min:
        return  candidates

    b = max(list(candidates.values()))
    a = min(list(candidates.values()))

    temp_3_dict = {}
    for key, value in temp_temp_dict.items():
        # temp_3_dict[key] = (a + (b - a) * (value - X_min) / (X_max - X_min))
        temp_3_dict[key] = 0.2*(a + (b - a) * (math.log(value) - math.log(X_min)) / (math.log(X_max) - math.log(X_min)))

    merged_dict = {**candidates, **temp_3_dict}

    return merged_dict

def expand_candidates_auto(candidates, bkg, interval, test_query):
    is_exist = True
    if len(candidates) == 0:
        is_exist = False

    timestamp_id = test_query[3]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    candidates_with_max_timestamp = select_canicates_based_timestamp(time_filtered, timestamp_id)

    exist_candidates_set = set(list(candidates.keys()))
    added_candidates_set = set(list(candidates_with_max_timestamp.keys()))

    share_candidates = exist_candidates_set.intersection(added_candidates_set)

    if not share_candidates:
        benchmark_rate = 1.0
    else:
        benchmark_rate = min(candidates[share] / candidates_with_max_timestamp[share] for share in share_candidates)

    candidates_with_max_timestamp = {key: value * benchmark_rate for key, value in candidates_with_max_timestamp.items()}

    merge_dict = {**candidates_with_max_timestamp, **candidates}

    return merge_dict, is_exist


def expand_candidates_with_freq_weight(candidates, bkg, interval, test_query, freq_weight):
    if len(candidates) == 1:
        return candidates, True

    source_id = test_query[0]
    target_timestamp_id = test_query[3]
    if interval == 0:
        time_mask = bkg[:, 3] < target_timestamp_id
    else:
        min_timestamp_id = target_timestamp_id - interval
        time_mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < target_timestamp_id)

    target_mask = bkg[:, 0] == source_id
    source_mask = bkg[:, 2] == source_id
    combined_mask = target_mask | source_mask
    mask = combined_mask * time_mask
    subgraph = bkg[mask]

    if len(subgraph) != 0:
        if len(candidates) == 0:
            min_score = 0
            max_score = 1
        else:
            min_score = min(list(candidates.values()))
            max_score = max(list(candidates.values()))

        candidates_with_max_timestamp_id = select_canicates_based_timestamp_normal(subgraph, target_timestamp_id,
                                                                                   min_score,
                                                                                   max_score)
    else:
        if len(candidates) == 0:
            mask = bkg[:, 3] < target_timestamp_id
            time_filtered = bkg[mask]
            candidates_with_max_timestamp_id = select_canicates_based_timestamp(time_filtered, target_timestamp_id)

            return candidates_with_max_timestamp_id, False
        else:
            return candidates, True

    # 获取两个字典的全部键集合
    all_keys = set(candidates_with_max_timestamp_id.keys()).union(set(candidates.keys()))

    # 使用字典的 get 方法设置默认值为 0
    merge_dict = {
        key: (1 - freq_weight) * candidates_with_max_timestamp_id.get(key, 0) + freq_weight * candidates.get(key, 0)
        for key in all_keys
    }

    return merge_dict, True



def expand_candidates_auto_with_freq_weight(candidates, bkg, interval, test_query, freq_weight):
    is_exist = True
    if len(candidates) == 0:
        is_exist = False

    timestamp_id = test_query[3]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    candidates_with_max_timestamp = select_canicates_based_timestamp(time_filtered, timestamp_id)

    exist_candidates_set = set(list(candidates.keys()))
    added_candidates_set = set(list(candidates_with_max_timestamp.keys()))

    share_candidates = exist_candidates_set.intersection(added_candidates_set)
    if not share_candidates:
        benchmark_rate = 1.0
        candidates_with_max_timestamp = {key: value * benchmark_rate for key, value in
                                         candidates_with_max_timestamp.items()}
        merge_dict = candidates_with_max_timestamp
    else:


        merge_dict = {**candidates_with_max_timestamp, **candidates}
        for cand_id in share_candidates:
            # merge_dict[cand_id] = (1 - freq_weight) * candidates[cand_id] + freq_weight * candidates_with_max_timestamp[
            #     cand_id]

            merge_dict[cand_id] = candidates[cand_id] + candidates_with_max_timestamp[
                cand_id]

    return merge_dict, is_exist

def expand_candidates_with_source(candidates, bkg, interval, test_query, freq_weight):
    exist_in_neighbors = True
    is_has_neighbors = True
    is_exist = True
    if len(candidates) == 0:
        is_exist = False

    timestamp_id = test_query[3]
    source_id = test_query[0]
    target_id = test_query[2]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    # 进行源实体的匹配
    target_mask = time_filtered[:, 0] == source_id
    source_mask = time_filtered[:, 2] == source_id
    candicates_target_with_timestamp = time_filtered[target_mask][:, [2,3]]
    candicates_source_with_timestamp = time_filtered[source_mask][:, [0,3]]
    # 合并候选实体数组
    candicates_with_neighbor = np.hstack((candicates_target_with_timestamp[:,0], candicates_source_with_timestamp[:,0]))
    if len(candicates_with_neighbor) == 0:
        result_dict = select_canicates_based_timestamp(time_filtered, timestamp_id)
        is_has_neighbors = False
        exist_in_neighbors = False
    else:
        unique_neighbors = np.unique(candicates_with_neighbor)
        if target_id not in unique_neighbors:
            exist_in_neighbors = False


        exist_candidates_set = set(list(candidates.keys()))
        added_candidates_set = set(unique_neighbors.tolist())

        share_candidates = exist_candidates_set.intersection(added_candidates_set)

        # 创建一个包含初始数据的DataFrame
        data = np.vstack((candicates_target_with_timestamp,candicates_source_with_timestamp))
        df = pd.DataFrame(data, columns=['neighbor', 'timestamp'])

        # 计算最大值10000 - 第二列的数值 的倒数
        df['timestamp'] = freq_weight * (1 / (timestamp_id - df['timestamp']))

        # 使用groupby根据第一列进行分组，并找到每个分组中第二列的最大值
        result = df.loc[df.groupby('neighbor')['timestamp'].idxmax()].reset_index(drop=True)
        result_dict = result.set_index('neighbor')['timestamp'].to_dict()

        # if len(share_candidates) == 0:
        #     benchmark_rate = 1.0
        # else:
        #     benchmark_rate = max(candidates[share] / result_dict[share] for share in share_candidates)
        #     benchmark_rate = sum(candidates[share] / result_dict[share] for share in share_candidates)/len(share_candidates)
        #
        # result_dict = {key: value * benchmark_rate for key, value in result_dict.items()}


    # merge_dict = {k: freq_weight * result_dict.get(k, 0) + candidates.get(k, 0) for k in set(result_dict) | set(candidates)}
    merge_dict = {**result_dict, **candidates}
    # merge_dict = {**result_dict}

    return merge_dict, is_exist, is_has_neighbors, exist_in_neighbors

def expand_candidates_with_relation(candidates, bkg, interval, test_query, freq_weight):
    exist_in_neighbors = True
    is_has_neighbors = True
    is_exist = True
    if len(candidates) == 0:
        is_exist = False

    timestamp_id = test_query[3]
    source_id = test_query[0]
    relation_id = test_query[1]
    target_id = test_query[2]

    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    # 进行源实体的匹配
    target_mask = time_filtered[:, 1] == relation_id
    source_mask = time_filtered[:, 1] == relation_id
    candicates_target_with_timestamp = time_filtered[target_mask][:, [2,3]]
    candicates_source_with_timestamp = time_filtered[source_mask][:, [0,3]]
    # 合并候选实体数组
    candicates_with_neighbor = np.hstack((candicates_target_with_timestamp[:,0], candicates_source_with_timestamp[:,0]))
    if len(candicates_with_neighbor) == 0:
        result_dict = select_canicates_based_timestamp(time_filtered, timestamp_id)
        is_has_neighbors = False
        exist_in_neighbors = False
    else:
        unique_neighbors = np.unique(candicates_with_neighbor)
        if target_id not in unique_neighbors:
            exist_in_neighbors = False

        # 创建一个包含初始数据的DataFrame
        data = np.vstack((candicates_target_with_timestamp,candicates_source_with_timestamp))
        df = pd.DataFrame(data, columns=['neighbor', 'timestamp'])

        # 计算最大值10000 - 第二列的数值 的倒数
        df['timestamp'] = freq_weight * (1 / (timestamp_id - df['timestamp']))

        # 使用groupby根据第一列进行分组，并找到每个分组中第二列的最大值
        result = df.loc[df.groupby('neighbor')['timestamp'].idxmax()].reset_index(drop=True)
        result_dict = result.set_index('neighbor')['timestamp'].to_dict()


    # merge_dict = {k: freq_weight * result_dict.get(k, 0) + candidates.get(k, 0) for k in set(result_dict) | set(candidates)}
    merge_dict = {**result_dict, **candidates}
    # merge_dict = {**result_dict}

    return merge_dict, is_exist, is_has_neighbors, exist_in_neighbors

def remove_candidates(candidates, data, interval, target_timestamp_id):
    min_timestamp_id = target_timestamp_id - interval

    train_n = np.array(data.train_idx)
    valid_n = np.array(data.valid_idx)
    analysis_bkg = np.vstack((valid_n, train_n))

    mask = analysis_bkg[:, 3] >= min_timestamp_id

    candidates_id = analysis_bkg[mask][:, 2]

    temp_dict = candidates.copy()
    for key in candidates.keys():
        if key not in candidates_id:
            del temp_dict[key]

    return temp_dict

def data_analysis(test_query, analysis_bkg_all):
    # Source Entity Existence Analysis
    source_id = test_query[0]
    

    # Use boolean indexing for a quick check without creating a new array
    source_exists_in_bkg = np.any(analysis_bkg_all[:, 0] == source_id)
    target_exists_in_bkg = np.any(analysis_bkg_all[:, 2] == source_id)

    # If source entity does not exist in either train or validation set
    if not (source_exists_in_bkg or target_exists_in_bkg):
        return 1

    return 0

def get_win_subgraph(test_data, data, learn_edges, window, win_start=0):
    unique_timestamp_id = np.unique(test_data[:,3])
    win_subgraph = {}
    for timestamp_id in unique_timestamp_id:
        subgraph = ra.get_window_edges(data.all_idx, timestamp_id - win_start, learn_edges, window)
        win_subgraph[timestamp_id] = subgraph

    return win_subgraph



def calculate_hours_between_dates_pandas(start_dates, end_dates):
    """
    使用 pandas 计算多对日期之间的小时差。

    参数:
        start_dates (list of str): 开始日期，格式为 "YYYY-MM-DD" 的字符串列表。
        end_dates (list of str): 结束日期，格式为 "YYYY-MM-DD" 的字符串列表。

    返回:
        list of float: 日期对之间的小时差列表。
    """
    # 将列表转换为 pandas Series
    start_series = pd.to_datetime(start_dates)
    end_series = pd.to_datetime(end_dates)

    # 计算小时差
    difference = (end_series - start_series).astype('timedelta64[h]').astype(int)

    return difference.tolist()


def merge_scores_optimized(dict_A, dict_B, model_weight):
    """
    Normalize the scores in two dictionaries to a 0-1 range and merge the scores for all keys.

    Args:
        dict_A (dict): Scores from the first dictionary.
        dict_B (dict): Scores from the second dictionary.

    Returns:
        dict: Merged scores dictionary for all keys from both dictionaries.
    """
    # Normalize the scores in both dictionaries
    normalized_A = normalize_scores(dict_A)
    normalized_B = normalize_scores(dict_B)

    # Get the union of keys in both dictionaries
    all_keys = set(dict_A.keys()) | set(dict_B.keys())

    # Initialize merged scores dictionary with zeros
    merged_scores = {key: 0 for key in all_keys}

    # Update merged scores dictionary with normalized values from both dictionaries
    for key in all_keys:
        score_A = normalized_A.get(key, 0)
        score_B = normalized_B.get(key, 0)
        merged_scores[key] = (score_A + score_B * model_weight) if key in dict_A and key in dict_B else score_A or (
                    score_B * model_weight)

    # normalized_B = {key: value * model_weight for key, value in normalized_B.items()}
    # merged_scores = {**normalized_B, **normalized_A}


    return merged_scores


# Helper function to normalize scores
def normalize_scores(score_dict):
    """
    Normalize the scores in the dictionary to a 0-1 range.

    Args:
        score_dict (dict): Dictionary with keys and scores.

    Returns:
        dict: Dictionary with normalized scores.
    """
    if not score_dict:
        return {}
    min_score = min(score_dict.values())
    max_score = max(score_dict.values())
    score_range = max_score - min_score if max_score != min_score else 1
    return {key: (value - min_score) / score_range for key, value in score_dict.items()}

def get_candicates_within_interval(timestamp_id, interval, bkg, return_recent=False):
    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    head_id = time_filtered[:,0]
    target_id = time_filtered[:,2]

    if return_recent is False:
       return list(set(head_id).union(set(target_id)))

    candicates_target_with_timestamp = time_filtered[:, [2, 3]]
    candicates_source_with_timestamp = time_filtered[:, [0, 3]]
    # 合并候选实体数组
    candicates_with_recent = np.hstack(
        (candicates_target_with_timestamp[:, 0], candicates_source_with_timestamp[:, 0]))
    if len(candicates_with_recent) == 0:
        return list(set(head_id).union(set(target_id))), {}
    else:
        # 创建一个包含初始数据的DataFrame
        data = np.vstack((candicates_target_with_timestamp, candicates_source_with_timestamp))
        df = pd.DataFrame(data, columns=['neighbor', 'timestamp'])

        # 计算最大值10000 - 第二列的数值 的倒数
        df['timestamp'] = (1 / (timestamp_id - df['timestamp']))

        # 使用groupby根据第一列进行分组，并找到每个分组中第二列的最大值
        result = df.loc[df.groupby('neighbor')['timestamp'].idxmax()].reset_index(drop=True)
        result_dict = result.set_index('neighbor')['timestamp'].to_dict()

def process_candidates_based_frequency(candidates:dict, test_query, interval,  bkg, learn_edges, obj_dist,
                                                            rel_obj_dist, freq_weight):
    # 判断是否存在候选项
    is_exist = bool(candidates)  # 使用bool直接转换，更简洁

    # 获取最大时间戳对应的候选项
    cand_with_max_timestamp = get_candicates_by_timestamp(test_query, bkg, interval)

    # 如果没有候选项直接返回
    if not candidates:
        candidates = baseline_candidates(test_query[1], learn_edges, obj_dist, rel_obj_dist)

    normalize_candidates = normalize_scores(candidates)

    # 计算共有键
    common_keys = set(normalize_candidates.keys()) & set(cand_with_max_timestamp.keys())

    # 如果没有共有键，直接返回原始候选项
    if not common_keys:
        return normalize_candidates, is_exist

    # 如果存在共有键，更新candidates中的分数
    for key in common_keys:
        score_A = normalize_candidates[key]  # 直接访问，因为key一定存在
        score_B = cand_with_max_timestamp.get(key, 0)
        normalize_candidates[key] = score_A + freq_weight * score_B

    return normalize_candidates, is_exist

def get_candicates_auto(timestamp_id, interval, bkg, return_recent=False):
    min_timestamp_id = timestamp_id - interval

    mask = (bkg[:, 3] >= min_timestamp_id) * (bkg[:, 3] < timestamp_id)

    # 先筛选出时间戳符合条件的条目
    time_filtered = bkg[mask]
    target_id = time_filtered[:,2]

    if return_recent is False:
       return list(set(target_id))