import json
import os
import numpy as np

from grapher import Grapher
from params import get_params
from utils import filter_candidates, calculate_rank

def main():
    parsed = get_params()

    dataset = parsed.dataset
    candidates_file = parsed.candidates

    dataset_dir = os.path.join("./datasets", dataset)
    ranked_rules_dir = os.path.join("./ranked_rules", dataset)

    data = Grapher(dataset_dir)
    num_entities = len(data.id2entity)
    test_data = data.test_idx if (parsed.test_data == "test") else data.valid_idx

    all_rule_candidates = load_candidates(ranked_rules_dir, candidates_file)

    if parsed.graph_reasoning_type in ['TiRGN', 'REGCN']:
        test_numpy, score_numpy = load_test_and_score_data(dataset, dataset_dir, parsed.graph_reasoning_type)
    else:
        test_numpy, score_numpy = None, None

    results = evaluate(parsed, test_data, all_rule_candidates, num_entities, test_numpy, score_numpy)
    hits_1, hits_3, hits_10, mrr = results

    hits_1 /= len(test_data)
    hits_3 /= len(test_data)
    hits_10 /= len(test_data)
    mrr /= len(test_data)

    print_results(hits_1, hits_3, hits_10, mrr)

    save_evaluation_results(ranked_rules_dir, candidates_file, hits_1, hits_3, hits_10, mrr)

def load_candidates(ranked_rules_dir, candidates_file):
    with open(os.path.join(ranked_rules_dir, candidates_file), 'r') as f:
        candidates = json.load(f)
    return {int(k): {int(cand): v for cand, v in v.items()} for k, v in candidates.items()}

def calculate_test_interval(data):
    recent_time = max(data.valid_idx[:, 3])
    test_timestamp = set(data.test_idx[:, 3])
    return {timestamp: timestamp - recent_time for timestamp in test_timestamp}

def load_test_and_score_data(dataset, dataset_dir, graph_reasoning_type):
    test_numpy = np.load(os.path.join(dataset_dir, graph_reasoning_type, 'test.npy'))
    if dataset == 'icews18':
        test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)
    score_numpy = np.load(os.path.join(dataset_dir, graph_reasoning_type, 'score.npy'))
    return test_numpy, score_numpy

def evaluate(parsed, test_data, all_rule_candidates, num_entities, test_numpy, score_numpy):
    hits_1 = hits_3 = hits_10 = mrr = 0
    num_samples = len(test_data)

    for i in range(num_samples):
        test_query = test_data[i]
        candidates = get_final_candidates(parsed, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy)
        candidates = filter_candidates(test_query, candidates, test_data)
        rank = calculate_rank(test_query[2], candidates, num_entities)

        hits_1, hits_3, hits_10, mrr = update_metrics(hits_1, hits_3, hits_10, mrr, rank)

    return hits_1, hits_3, hits_10, mrr

def get_final_candidates(parsed, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy):
    if parsed.graph_reasoning_type in ['TiRGN', 'REGCN']:
        return get_candidates(parsed, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy)
    else:
        return all_rule_candidates[i]


def get_candidates(parsed, test_query, all_rule_candidates, i, num_entities, test_numpy, score_numpy):
    temp_candidates = {k: 0 for k in range(num_entities)}
    rule_candidates = all_rule_candidates[i]
    rule_candidates = {**temp_candidates, **rule_candidates}

    indices = np.where((test_numpy == test_query).all(axis=1))[0]
    score = score_numpy[indices[0]]
    regcn_candidates = {index: value for index, value in enumerate(score)}

    candidates = {k: (1 - parsed.rule_weight) * regcn_candidates[k] + parsed.rule_weight * rule_candidates[k] for k in
                  rule_candidates}
    return candidates

def update_metrics(hits_1, hits_3, hits_10, mrr, rank):
    if rank <= 10:
        hits_10 += 1
        if rank <= 3:
            hits_3 += 1
            if rank == 1:
                hits_1 += 1
    mrr += 1 / rank
    return hits_1, hits_3, hits_10, mrr

def print_results(hits_1, hits_3, hits_10, mrr):
    print("Hits@1: ", round(hits_1, 6))
    print("Hits@3: ", round(hits_3, 6))
    print("Hits@10: ", round(hits_10, 6))
    print("MRR: ", round(mrr, 6))

def save_evaluation_results(ranked_rules_dir, candidates_file, hits_1, hits_3, hits_10, mrr):
    filename = candidates_file[:-5] + "_eval.txt"
    with open(os.path.join(ranked_rules_dir, filename), "w", encoding="utf-8") as fout:
        fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
        fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
        fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
        fout.write("MRR: " + str(round(mrr, 6)) + "\n")

if __name__ == "__main__":
    main()
