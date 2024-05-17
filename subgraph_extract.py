import argparse
import time
import os
import rule_application as ra
import numpy as np

from grapher import Grapher
from joblib import Parallel, delayed

from utils import load_json_data, save_json_data


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
parser.add_argument("--dataset", "-d", default="icews14", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--rule_files", "-f", default="", type=str)
parser.add_argument("--confidence_type", default="TLogic", type=str,
                    choices=['TLogic', 'LLM', 'And', 'Or'])
parser.add_argument("--weight", default=0.0, type=float)
parser.add_argument("--weight_0", default=0.5, type=float)
parser.add_argument("--min_conf", default=0.01, type=float)
parser.add_argument("--bgkg", default="all", type=str,
                    choices=['all', 'train', 'valid', 'test', 'train_valid', 'train_test', 'valid_test'])
parser.add_argument("--score_type", default="noisy-or", type=str,
                    choices=['noisy-or', 'sum', 'mean', 'min', 'max'])
parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
# parser.add_argument("--subgraph_dir", default="./subgraph/", type=str)

args = parser.parse_args()
parsed = vars(parser.parse_args())

dataset = args.dataset
num_processes = args.num_processes

args.subgraph_dir = "./subgraph/" + dataset + "/"

dataset_dir = "./datasets/" + dataset + "/"
dir_path = "./ranked_rules/" + dataset + "/"
data = Grapher(dataset_dir, parsed)
test_data = data.test_idx if (args.test_data == "test") else data.valid_idx
unique_head_relation = np.unique(test_data[:,0:2],axis=0)

def extract_subgraph_for_mulitis_thread(i, num_queries, args, grapher):
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

    if i < num_processes - 1:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(unique_head_relation))

    similary_matrix = np.load(os.path.join(args.subgraph_dir, "matrix.npy"))
    transformers_rel2id = load_json_data(os.path.join(args.subgraph_dir, "transfomers_rel2id.json"))


    all_neighbors = {}
    for j in test_queries_idx:
        all_neighbors_list = []
        query_head_relation = unique_head_relation[j]
        query_head = query_head_relation[0]
        query_relation = query_head_relation[1]
        neighbors = grapher.all_idx[query_head==grapher.all_idx[:,0]]
        neighbors_relation_names = np.array(list(grapher.id2relation.values()))[neighbors[:,1]]
        neighbors_rel_id_list = []

        transformers_query_id = transformers_rel2id[grapher.id2relation[query_relation]]
        for rel_name in neighbors_relation_names:
            neighbors_rel_id = transformers_rel2id[rel_name]
            neighbors_rel_id_list.append(neighbors_rel_id)

        relevance_with_query = similary_matrix[transformers_query_id][neighbors_rel_id_list]
        first_order_neighbors = neighbors[np.argsort(relevance_with_query)[-args.top_k:][::-1]]

        first_order_nei = get_unique_list(first_order_neighbors.tolist())
        all_neighbors_list.extend(first_order_nei)

        second_order_neighbors_list = []
        for target in set(first_order_neighbors[:,2].tolist()):
            neighbors = grapher.all_idx[target==grapher.all_idx[:,0]]
            neighbors_relation_names = np.array(list(grapher.id2relation.values()))[neighbors[:, 1]]
            neighbors_rel_id_list = []

            transformers_query_id = transformers_rel2id[grapher.id2relation[query_relation]]
            for rel_name in neighbors_relation_names:
                neighbors_rel_id = transformers_rel2id[rel_name]
                neighbors_rel_id_list.append(neighbors_rel_id)

            relevance_with_query = similary_matrix[transformers_query_id][neighbors_rel_id_list]
            second_order_neighbors = neighbors[np.argsort(relevance_with_query)[-args.top_k:][::-1]]
            second_order_neighbors_list.extend(second_order_neighbors.tolist())

        second_order_nei = get_unique_list(second_order_neighbors_list)

        all_neighbors_list.extend(second_order_nei)

        all_neighbors['_'.join(map(str, query_head_relation.tolist()))] = get_unique_list(all_neighbors_list)

    return all_neighbors


def get_unique_list(second_order_neighbors_list):
    # 将子列表转换为 NumPy 数组
    array_list = np.array(second_order_neighbors_list)
    # 使用 NumPy 的 unique 函数去重，并保留原始顺序
    _, idx = np.unique(array_list, axis=0, return_index=True)
    unique_list = array_list[np.sort(idx)].tolist()

    return unique_list


start = time.time()
num_queries = len(unique_head_relation) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(extract_subgraph_for_mulitis_thread)(i, num_queries, args, data) for i in range(num_processes)
)
end = time.time()
total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))

merged_dict = {}
for d in output:
    merged_dict.update(d)

subgraph_file_path = args.subgraph_dir + f'top_k_{args.top_k}_2_order.json'
save_json_data(merged_dict, subgraph_file_path)



