import argparse
import json
import os
import shutil
import glob
from tqdm import tqdm
from functools import partial
from data import *
from multiprocessing.pool import ThreadPool
import random
import ast
from utils import *
from llms import get_registed_model

prompt_dict = {}

prompt_dict['chain_defination'] = (
    'The defination of Temporal Logical Rules:\n Temporal Logical Rules "{head}(X,Y,T(l+1))<-R1(X,X1,T1)&...&Rl(X(l-1),Y,Tl)" are rules used in temporal knowledge graph reasoning to predict relations between entities over time. They describe how the relation "{head}" between entities "X" and "Y" evolves from past time steps "Ti (i={{1,...,l}})" (rule body) to the next "T(l+1)" (rule head), strictly following the constraint "T1 <= ··· <= Tl < T(l+1)".\n\n')

prompt_dict['subgraph_defination'] = (
    'Each rule head "{head}(X,Y,T)" is associated with multiple subgraphs and each subgraph consists of multiple quadruplets "[[X,Ri,Y,Ti]]", where "Ri" represents the relation between entities "X" and "Y", and "Ti" represents the timestamp.\n\n')


prompt_dict['context'] = (
    'You are an automated reasoning engine capable of generating as many most relevant temporal logic rules corresponding to the "{head}(X,Y,T)" based on chains and subgraphs, ensuring they conform to the definition of temporal logic rules.\n\n')

# prompt_dict['rel_id']['answer']='Please answer\n'
prompt_dict['rel_id'] = {}
prompt_dict['rel_id']['Few_context_for_chain'] = "chain rules:\n"
prompt_dict['rel_id']['Few_context_for_subgraph'] = "subgraphs:\n"

prompt_dict['rel_name'] = {}
prompt_dict['rel_name']['Few_context_for_chain'] = "chain rules:\n"
prompt_dict['rel_name']['Few_context_for_subgraph'] = "subgraphs:\n"

prompt_dict['Final_predict'] = (
    '\n\nLet\'s think step-by-step, and based on the above chain rules and subgraphs, please generate as many as possible most relevant temporal rules that are relative to "{head}(X,Y,T)".\n\n')
prompt_dict[
    'return'] = 'For the relations in rule body, you are going to choose from the following candidates: {candidate_rels}, where each candidate relation is represented by a natural language name and a corresponding ID.\n\n Return the rules only without any explanations.'


def read_paths(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def build_prompt(head, candidate_rels, is_zero, args, prompt_dict):
    k = args.k
    if args.is_rel_name is True:
        type = 'rel_name'
    else:
        type = 'rel_id'

    # head = clean_symbol_in_rel(head)
    chain_defination = prompt_dict['chain_defination'].format(head=head)
    subgraph_defination = prompt_dict['subgraph_defination'].format(head=head)
    # chain_defination = prompt_dict['chain_defination']
    # subgraph_defination = prompt_dict['subgraph_defination']

    if is_zero and args.k != 0:  # Zero-shot
        context = prompt_dict[type]['Zero_context'].format(head=head)
        predict = prompt_dict[type]['Zero_predict'].format(head=head, k=k)
        # predict = f'\nGiven a rule head: "{head}(X,Y, T)", please thoroughly understand it, contemplate their feasibility, and generate {k} rules that are the most important and relevant to the rule head and their associated confidence score values. Ensure each rule is accompanied by its corresponding confidence score without the need for a “confidence” label.'
    else:  # Few-shot
        context = prompt_dict['context'].format(head=head)
        few_context_for_chain = prompt_dict['rel_name']['Few_context_for_chain']
        few_context_for_subgraph = prompt_dict['rel_name']['Few_context_for_subgraph']

    predict = prompt_dict['Final_predict'].format(head=head)
    return_rules = prompt_dict['return']
    return chain_defination + subgraph_defination + context, few_context_for_chain, few_context_for_subgraph, predict, return_rules


def get_rule_format(head, path, kg_rules_path):
    kg_rules_dict = load_json_data(kg_rules_path)
    if kg_rules_dict is None:
        path_list = []
        # head = clean_symbol_in_rel(head)
        for p in path:
            context = f"{head}(X,Y) <-- "
            for i, r in enumerate(p.split("|")):
                # r = clean_symbol_in_rel(r)
                if i == 0:
                    first = "X"
                else:
                    first = f"Z_{i}"
                if i == len(p.split("|")) - 1:
                    last = "Y"
                else:
                    last = f"Z_{i + 1}"
                context += f"{r}({first}, {last}) & "
            context = context.strip(" & ")
            path_list.append(context)
        return path_list
    else:
        return kg_rules_dict[head]


def get_subgraph(relation_subgraph, head_id, fixed_character, rules, model):
    subgraphes = relation_subgraph[str(head_id)]

    max_promt = 1000000
    min_idx = 0
    for idx, subgraph in enumerate(subgraphes):
        # subgraph = len(list(subgraph.values())[0])
        temp_list = list(subgraph.values())
        unique_list = [list(t) for t in set(tuple(sublist) for sublist in temp_list[0])]
        sorted_list = sorted(unique_list, key=lambda x: x[3])
        list_str = str(sorted_list)
        before_all_tokens = fixed_character + "\n".join(rules)
        before_length = model.token_len(before_all_tokens)
        after_all_tokens = fixed_character + list_str + "\n".join(rules)
        # maximun_token = model.maximum_token
        maximun_token = before_length + 2000
        tokens_length = model.token_len(after_all_tokens)
        if tokens_length < maximun_token:
            return list_str
        else:
            if tokens_length < max_promt:
                max_promt = tokens_length
                min_idx = idx

    min_length_subgraph = subgraphes[min_idx]
    my_list = list(min_length_subgraph.values())[0]

    my_list_array = np.array(my_list)
    timestamps = my_list_array[:, 3]
    unique_array = np.unique(timestamps)
    for idx, time in enumerate(unique_array):
        prume_subgraph = my_list_array[timestamps > time].tolist()
        unique_list = [list(t) for t in set(tuple(sublist) for sublist in prume_subgraph)]
        sorted_list = sorted(unique_list, key=lambda x: x[3])
        list_str = str(sorted_list)
        before_all_tokens = fixed_character + "\n".join(rules)
        before_length = model.token_len(before_all_tokens)
        after_all_tokens = fixed_character + list_str + "\n".join(rules)
        # maximun_token = model.maximum_token
        maximun_token = before_length + 2000
        tokens_length = model.token_len(after_all_tokens)
        if tokens_length < maximun_token:
            return list_str


def generate_rule(row, rdict, rule_path, kg_rules_path, model, args, relation_subgraph, relation_regex,
                  similiary_rel_dict):
    relation2id = rdict.rel2idx
    head = row["head"]
    paths = row["paths"]
    head_id = relation2id[head]
    # print("Head: ", head)

    head_formate = head
    if args.is_rel_name is True:
        all_rels = list(relation2id.keys())
        candidate_rels = ", ".join(all_rels)
        head_formate = head
    else:
        all_rels = list(relation2id.values())
        str_list = [str(item) for item in all_rels]
        candidate_rels = ", ".join(str_list)
        head_formate = head_id

    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )
    # Build prompt excluding rules
    fixed_context, few_context_for_chain, few_context_for_subgraph, predict, return_rules = build_prompt(
        head_formate, candidate_rels, args.is_zero, args, prompt_dict
    )
    current_prompt = fixed_context + few_context_for_chain + few_context_for_subgraph + predict + return_rules

    if args.is_zero:  # For zero-shot setting
        with open(os.path.join(rule_path, f"{head}_zero_shot.query"), "w") as f:
            f.write(current_prompt + "\n")
            f.close()
        if not args.dry_run:
            response = query(current_prompt, model)
            with open(os.path.join(rule_path, f"{head}_zero_shot.txt"), "w") as f:
                f.write(response + "\n")
                f.close()
    else:  # For few-shot setting
        path_content_list = get_rule_format(head, paths, kg_rules_path)
        file_name = head.replace("/", "-")
        with open(os.path.join(rule_path, f"{file_name}.txt"), "w") as rule_file, open(
                os.path.join(rule_path, f"{file_name}.query"), "w") as query_file:
            rule_file.write(f"Rule_head: {head}\n")
            for i in range(args.l):

                if args.select_with_confidence is True:
                    sorted_list = sorted(path_content_list, key=lambda x: float(x.split('&')[-1]), reverse=True)
                    # few_shot_samples = sorted_list[:args.f]
                    new_shot_samples = [item for item in sorted_list if float(item.split('&')[-1]) > 0.01]
                    if len(new_shot_samples) >= args.f:
                        few_shot_samples = new_shot_samples
                    else:
                        few_shot_samples = sorted_list[:args.f]
                else:
                    few_shot_samples = random.sample(
                        path_content_list, min(args.f, len(path_content_list))
                    )
                    relation_set = set()
                    for rule in few_shot_samples:
                        rule_body = rule.split('<-')[-1]
                        matches = re.findall(relation_regex, rule_body)
                        for match in matches:
                            relation = match[0]
                            relation_set.update([relation])

                    similiary_rel_set = set()
                    for rel_name in relation_set:
                        similiary_rel_set.update(similiary_rel_dict[rel_name])

                    condicate = similiary_rel_set.union(relation_set)

                    # Convert list elements to the desired string format
                    formatted_string = ';'.join([f'{relation2id[name]}:{name}' for name in condicate])

                # fixed_context, few_context_for_chain, few_context_for_subgraph, predict, return_rules

                return_rules = return_rules.format(candidate_rels=formatted_string)

                temp_current_prompt = fixed_context + few_context_for_chain + few_context_for_subgraph + predict + return_rules

                few_shot_subgraph = get_subgraph(relation_subgraph, head_id, temp_current_prompt, few_shot_samples, model)

                few_shot_paths = check_prompt_length(
                    temp_current_prompt + few_shot_subgraph,
                    few_shot_samples, model
                )

                if not few_shot_paths:
                    raise ValueError("few_shot_paths is empty, head:{}".format(head))

                few_shot_paths = few_shot_paths + "\n\n"

                return_rules = "\n\n" + return_rules

                prompt = fixed_context + few_context_for_chain + few_shot_paths + few_context_for_subgraph + few_shot_subgraph + predict + return_rules
                model.token_len(prompt)
                # tqdm.write("Prompt: \n{}".format(prompt))
                query_file.write(f"Sample {i + 1} time: \n")
                query_file.write(prompt + "\n")
                if not args.dry_run:
                    response = model.generate_sentence(prompt)
                    if response is not None:
                        # tqdm.write("Response: \n{}".format(response))
                        rule_file.write(f"Sample {i + 1} time: \n")
                        rule_file.write(response + "\n")
                    else:
                        with open(os.path.join(rule_path, f"fail_{file_name}.txt"), "w") as fail_rule_file:
                            fail_rule_file.write(prompt + "\n")
                        break


def copy_files(source_dir, destination_dir, file_extension):
    # 创建目标文件夹
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 遍历源文件夹中的文件
    for filename in os.listdir(source_dir):
        # 检查文件类型是否符合要求
        if filename.endswith(file_extension):
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename)
            # 复制文件
            shutil.copyfile(source_file, destination_file)


def process_rules_files(input_dir, output_dir, rdict, relation_regex, error_file_path):
    sum = 0
    with open(error_file_path, 'w') as f_error_out:
        for input_filepath in glob.glob(os.path.join(input_dir, "*.txt")):
            file_name = input_filepath.split("/")[-1]
            if file_name.startswith('fail'):
                continue
            else:
                with open(input_filepath, 'r') as fin, open(os.path.join(output_dir, file_name), 'w') as fout:
                    rules = fin.readlines()
                    for idx, rule in enumerate(rules):
                        is_save = True
                        if rule.startswith('Rule_head:'):
                            continue
                        elif rule.startswith('Sample'):
                            continue
                        else:
                            rule_by_name = ""
                            temp_rule = re.sub(r'\s*<-\s*', '&', rule)
                            regrex_list = re.split(r'\s*&\s*|\t', temp_rule)
                            confidence = regrex_list[-1].strip()
                            for id, regrex in enumerate(regrex_list[:-1]):
                                match = re.search(relation_regex, regrex)
                                if match:
                                    if match[1].strip().isdigit():
                                        rel_id = int(match[1].strip())
                                        if rel_id not in list(rdict.idx2rel):
                                            print(f"Error relation id:{rel_id}, rule:{rule}")
                                            f_error_out.write(f"Error relation id:{rel_id}, rule:{rule}")
                                            sum = sum + 1
                                            is_save = False
                                            break

                                        relation_name = rdict.idx2rel[rel_id]
                                        subject = match[2].strip()
                                        object = match[3].strip()
                                        timestamp = match[4].strip()
                                        regrex_name = f"{relation_name}({subject},{object},{timestamp})"
                                        if id == 0:
                                            regrex_name += '<-'
                                        else:
                                            regrex_name += '&'
                                        rule_by_name += regrex_name
                                    else:
                                        print(f"Error relation id:{match[1].strip()}, rule:{rule}")
                                        f_error_out.write(f"Error relation id:{match[1].strip()}, rule:{rule}")
                                        sum = sum + 1
                                        is_save = False
                                        break

                                else:
                                    print(f"Error rule:{rule}, rule:{rule}")
                                    f_error_out.write(f"Error rule:{rule}, rule:{rule}")
                                    sum = sum + 1
                                    is_save = False
                                    break
                            if is_save:
                                rule_by_name += confidence
                                fout.write(rule_by_name + '\n')
        f_error_out.write(f"The number of error during id maps name is:{sum}")


def clear_folder(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        return

    # 遍历文件夹中的所有文件和文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 如果是文件，则直接删除
        if os.path.isfile(file_path):
            os.remove(file_path)
        # 如果是文件夹，则递归清空文件夹
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_topk_similiary_rel(topk, similary_matrix, transformers_id2rel, transformers_rel2id):
    # 计算每一行中数值最大的前 topk 个元素的索引
    topk = -topk
    top_k_indices = np.argsort(similary_matrix, axis=1)[:, topk:]
    similiary_rel_dict = {}
    for idx, similary_rels in enumerate(top_k_indices):
        rel_name = transformers_id2rel[str(idx)]
        similary_rel_name = [transformers_id2rel[str(i)] for i in similary_rels]
        similiary_rel_dict[rel_name] = similary_rel_name

    return similiary_rel_dict


def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)
    sampled_path_dir = os.path.join(args.sampled_paths, args.dataset)
    sampled_path = read_paths(os.path.join(sampled_path_dir, "closed_rel_paths.jsonl"))
    if args.is_rel_name is True:
        kg_rules_path = os.path.join(sampled_path_dir, "rules_name.json")
    else:
        kg_rules_path = os.path.join(sampled_path_dir, "rules_id.json")

    constant_config = load_json_data('./Config/constant.json')
    relation_regex = constant_config['relation_regex'][args.dataset]

    relation_subgraph_path = os.path.join(sampled_path_dir, "relation_subgraph.json")
    relation_subgraph = load_json_data(relation_subgraph_path)

    rdict = dataset.get_relation_dict()

    similary_matrix = np.load(os.path.join(sampled_path_dir, "matrix.npy"))
    transformers_id2rel = load_json_data(os.path.join(sampled_path_dir, "transfomers_id2rel.json"))
    transformers_rel2id = load_json_data(os.path.join(sampled_path_dir, "transfomers_rel2id.json"))

    similiary_rel_dict = get_topk_similiary_rel(args.topk, similary_matrix, transformers_id2rel, transformers_rel2id)

    # Save paths
    rule_path = os.path.join(
        args.rule_path,
        args.dataset,
        f"{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}",
    )
    if not os.path.exists(rule_path):
        os.makedirs(rule_path)

    filter_rule_path = os.path.join(
        args.rule_path,
        args.dataset,
        f"copy_{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}",
    )
    if not os.path.exists(filter_rule_path):
        os.makedirs(filter_rule_path)
    else:
        clear_folder(filter_rule_path)

    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    # Generate rules
    with ThreadPool(args.n) as p:
        for _ in tqdm(
                p.imap_unordered(
                    partial(
                        generate_rule,
                        rdict=rdict,
                        rule_path=rule_path,
                        kg_rules_path=kg_rules_path,
                        model=model,
                        args=args,
                        relation_subgraph=relation_subgraph,
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict
                    ),
                    sampled_path,
                ),
                total=len(sampled_path),
        ):
            pass

    exit(0)

    for input_filepath in glob.glob(os.path.join(rule_path, "fail_*.txt")):
        filename = input_filepath.split('/')[-1].split('fail_')[-1]
        with open(input_filepath, 'r') as fin, open(os.path.join(rule_path, filename), 'w') as fout:
            content = fin.read()
            response = model.generate_sentence(content)
            if response is not None:
                fout.write(response + "\n")
            else:
                print(f"Error:{filename}")

    statistics_dir = os.path.join(
        args.rule_path,
        args.dataset,
        "statistics",
    )

    if not os.path.exists(statistics_dir):
        os.makedirs(statistics_dir)
    else:
        clear_folder(statistics_dir)

    statistics_file_path = os.path.join(statistics_dir, 'statistics.txt')
    error_file_path = os.path.join(statistics_dir, 'error.txt')

    if args.is_rel_name is True:
        copy_files(rule_path, filter_rule_path, 'txt')
    else:
        constant_config = load_json_data('./Config/constant.json')
        relation_regex = constant_config['relation_regex'][args.dataset]
        process_rules_files(rule_path, filter_rule_path, rdict, relation_regex, error_file_path)

    model.gen_rule_statistic(rule_path, statistics_file_path)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="datasets", help="data directory"
    )
    parser.add_argument("--dataset", type=str, default="family", help="dataset")
    parser.add_argument(
        "--sampled_paths", type=str, default="sampled_path", help="sampled path dir"
    )
    parser.add_argument(
        "--rule_path", type=str, default="gen_rules_based_subgraph", help="path to rule file"
    )
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument(
        "--is_zero",
        action="store_true",
        help="Enable this for zero-shot rule generation",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=0,
        help="Number of generated rules, 0 denotes as much as possible",
    )
    parser.add_argument("-f", type=int, default=5, help="Few-shot number")
    parser.add_argument("-topk", type=int, default=20, help="topk")
    parser.add_argument("-n", type=int, default=5, help="multi thread number")
    parser.add_argument(
        "-l", type=int, default=3, help="sample l times for generating k rules"
    )
    parser.add_argument("--prefix", type=str, default="", help="prefix")
    parser.add_argument("--dry_run", action="store_true", help="dry run")
    parser.add_argument("--is_rel_name", default='yes', type=str_to_bool)
    parser.add_argument("--select_with_confidence", default='no', type=str_to_bool)

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
