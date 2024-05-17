import torch
from model import TADistmultModel
from sklearn.metrics.pairwise import linear_kernel
from typing import Dict


tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18, '10m': 19, '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
}

def init_model(path_name: str) -> TADistmultModel:
    distmult_model = torch.load(path_name, map_location=torch.device('cpu'))
    return distmult_model

def get_scores_with_candicates(distmult_model: TADistmultModel, test_query, id2ts : dict, candicates_list: list) -> Dict[int, float]:
    ent_embeddings = distmult_model.ent_embeddings.weight.data.cpu().numpy()
    head_id = test_query[0]
    rel_id = test_query[1]
    timestamp_id = test_query[3]
    timestamp_name  = id2ts[timestamp_id]
    timestamp_name = get_tem_list(timestamp_name)

    test_r_batch = torch.LongTensor([rel_id])
    test_time_batch = torch.LongTensor(timestamp_name)
    rseq_e = distmult_model.get_rseq(test_r_batch, test_time_batch).data.cpu().numpy()
    head_embeddings = ent_embeddings[head_id]
    candicates_embeddings = ent_embeddings[candicates_list]
    c_t_e = head_embeddings * rseq_e
    # 使用linear_kernel得到head和每个候选项之间的分数
    scores = linear_kernel(c_t_e.reshape(1, -1), candicates_embeddings).flatten()

    # 构建一个字典，键为候选项ID，值为相应的分数
    dist = {candicate: score for candicate, score in zip(candicates_list, scores)}

    return dist

def get_tem_list(timestamp_name):
    tem = []
    year, month, day = timestamp_name.split("-")
    tem_id_list = []
    for j in range(len(year)):
        token = year[j:j + 1] + 'y'
        tem_id_list.append(tem_dict[token])

    for j in range(1):
        token = month + 'm'
        tem_id_list.append(tem_dict[token])

    for j in range(len(day)):
        token = day[j:j + 1] + 'd'
        tem_id_list.append(tem_dict[token])

    tem.append(tem_id_list)

    return tem

def get_scores_with_recent(distmult_model: TADistmultModel, test_query, id2ts : dict, candicates_list: list) -> Dict[int, float]:
    ent_embeddings = distmult_model.ent_embeddings.weight.data.cpu().numpy()
    head_id = test_query[0]
    rel_id = test_query[1]
    timestamp_id = test_query[3]
    timestamp_name  = id2ts[timestamp_id]
    timestamp_name = get_tem_list(timestamp_name)

    test_r_batch = torch.LongTensor([rel_id])
    test_time_batch = torch.LongTensor(timestamp_name)
    rseq_e = distmult_model.get_rseq(test_r_batch, test_time_batch).data.cpu().numpy()
    head_embeddings = ent_embeddings[head_id]
    candicates_embeddings = ent_embeddings[candicates_list]
    c_t_e = head_embeddings * rseq_e
    # 使用linear_kernel得到head和每个候选项之间的分数
    scores = linear_kernel(c_t_e.reshape(1, -1), candicates_embeddings).flatten()

    # 构建一个字典，键为候选项ID，值为相应的分数
    dist = {candicate: score for candicate, score in zip(candicates_list, scores)}

    return dist



