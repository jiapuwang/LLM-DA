import scipy.sparse as sp
from rrgcn import RecurrentRGCN

import os
import numpy as np
import torch
from rgcn.utils import split_by_time, build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from tqdm import tqdm


def initialization(parsed, dataset_dir, num_entities, num_rels, use_cuda):
    train_quadruple_path = os.path.join(dataset_dir, 'train_.txt')
    valid_quadruple_path = os.path.join(dataset_dir, 'valid_.txt')
    test_quadruple_path = os.path.join(dataset_dir, 'test_.txt')

    # 使用 numpy 的 loadtxt 方法读取数据
    train_quadruple = np.loadtxt(train_quadruple_path, dtype=int)
    valid_quadruple = np.loadtxt(valid_quadruple_path, dtype=int)
    test_quadruple = np.loadtxt(test_quadruple_path, dtype=int)

    train_list, train_times = split_by_time(train_quadruple)  # 划分为snapshots，逐时间步的数据集
    valid_list, valid_times = split_by_time(valid_quadruple)
    test_list, test_times = split_by_time(test_quadruple)

    time_interval = train_times[1] - train_times[0]

    if parsed.dataset == "tigcn_icews14":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)

    if parsed.add_static_graph:
        static_triples_path = os.path.join(dataset_dir, "e-w-graph.txt")
        static_triples = np.array(_read_triplets_as_list(static_triples_path, {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_entities
        static_node_id = torch.from_numpy(np.arange(num_words + num_entities)).view(-1, 1).long().cuda(parsed.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + num_entities)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_node_id = 0, 0, [], None

    model = create_recurrent_rgcn(parsed, num_entities, int(num_rels / 2), num_static_rels, num_words, num_times,
                                  time_interval, use_cuda)

    if use_cuda:
        torch.cuda.set_device(parsed.gpu)
        model.cuda()

    static_graph = None
    if parsed.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, parsed.gpu)

    data_tuple = (train_list, valid_list, test_list, train_times, valid_times, test_times)

    return model, static_graph, data_tuple


def create_recurrent_rgcn(args, num_nodes, num_rels, num_static_rels, num_words, num_times, time_interval, use_cuda):
    model = RecurrentRGCN(
        decoder_name=args.decoder,
        encoder_name=args.encoder,
        num_ents=num_nodes,
        num_rels=num_rels,
        num_static_rels=num_static_rels,
        num_words=num_words,
        num_times=num_times,
        time_interval=time_interval,
        h_dim=args.n_hidden,
        opn=args.opn,
        history_rate=args.history_rate,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis
    )
    return model


def load_model_and_evaluate(model, use_cuda, parsed, model_name):
    if use_cuda:
        checkpoint = torch.load(model_name, map_location=torch.device(parsed.gpu))
    else:
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    print("Load Model name: {}. Using best epoch : {}".format(model_name,
                                                              checkpoint['epoch']))  # use best stat checkpoint
    print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


def perform_model_prediction_with_history(model, parsed, data_tuple, dataset_dir, static_graph, use_cuda, num_nodes,
                                          num_rels):
    train_list = data_tuple[0]
    valid_list = data_tuple[1]
    test_list = data_tuple[2]
    train_times = data_tuple[3]
    valid_times = data_tuple[4]
    test_times = data_tuple[5]

    history_list = train_list + valid_list
    history_time_nogt = test_times[0]

    input_list = [snap for snap in history_list[-parsed.test_history_len:]]

    if parsed.multi_step:
        all_tail_seq_file = f'tail_history_{history_time_nogt}.npz'
        all_tail_seq_path = os.path.join(dataset_dir, 'history', all_tail_seq_file)
        all_tail_seq = sp.load_npz(all_tail_seq_path)

        all_rel_seq_file = f'rel_history_{history_time_nogt}.npz'
        all_rel_seq_path = os.path.join(dataset_dir, 'history', all_rel_seq_file)
        all_rel_seq = sp.load_npz(all_rel_seq_path)

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, parsed.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(parsed.gpu)

        # get history
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()
        if parsed.multi_step:
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq_file = f'tail_history_{test_times[time_idx]}.npz'
            all_tail_seq_path = os.path.join(dataset_dir, 'history', all_tail_seq_file)
            all_tail_seq = sp.load_npz(all_tail_seq_path)

            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            all_rel_seq_file = f'rel_history_{test_times[time_idx]}.npz'
            all_rel_seq_path = os.path.join(dataset_dir, 'history', all_rel_seq_file)
            all_rel_seq = sp.load_npz(all_rel_seq_path)

            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda()
            one_hot_rel_seq = one_hot_rel_seq.cuda()

        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
                                                                 test_triples_input, one_hot_tail_seq, one_hot_rel_seq,
                                                                 use_cuda)
