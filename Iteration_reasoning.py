import argparse
import glob
import os.path
import traceback
from difflib import get_close_matches

from tqdm import tqdm
from functools import partial

from data import *
from multiprocessing.pool import ThreadPool
from utils import *
from llms import get_registed_model
from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics

from concurrent.futures import ThreadPoolExecutor
from params import str_to_bool

def read_paths(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def build_prompt(head, prompt_dict):
    definition = prompt_dict['common']['definition'].format(head=head)
    context = prompt_dict['common']['context'].format(head=head)
    chain = prompt_dict['common']['chain']
    predict = prompt_dict['common']['predict'].format(head=head)
    return_rules = prompt_dict['common']['return']
    return definition + context + chain, predict, return_rules

def build_prompt_for_zero(head, prompt_dict):
    context = prompt_dict['zero']['context'].format(head=head)
    predict = prompt_dict['zero']['predict'].format(head=head)
    return_rules = prompt_dict['zero']['return']
    return context, predict, return_rules


def build_prompt_based_high(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    chain_defination = prompt_dict['chain_defination_for_high'].format(head=head)

    context = prompt_dict['iteration_context_for_high'].format(head=head)

    high_quality_context = prompt_dict['example_for_high']
    # predict = prompt_dict['interaction_finale_predict_for_high'].format(head=head, k=20)
    predict = prompt_dict['interaction_finale_predict_for_high'].format(head=head)
    return_rules = prompt_dict['return_for_high']

    return chain_defination + context + high_quality_context, predict, return_rules

def build_prompt_based_low(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    # chain_defination = prompt_dict['iteration']['defination']['chain'].format(head=head)

    context = prompt_dict['iteration']['context'].format(head=head)
    low_quality_context = prompt_dict['iteration']['low_quality_example']
    sampled_rules = prompt_dict['iteration']['sampled_rules']
    predict = prompt_dict['iteration']['predict'].format(head=head)
    return_rules = prompt_dict['iteration']['return']
    return  context + low_quality_context, sampled_rules,predict, return_rules

def build_prompt_for_unknown(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    chain_defination = prompt_dict['chain_defination'].format(head=head)

    context = prompt_dict['unknown_relation_context'].format(head=head)

    predict = prompt_dict['unknown_relation_final_predict'].format(head=head)
    return_rules = prompt_dict['unknown_relation_return']
    return chain_defination + context, predict, return_rules


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


def generate_rule(row, rdict, rule_path, kg_rules_path, model, args, relation_regex,
                  similiary_rel_dict, prompt_info_dict):
    relation2id = rdict.rel2idx
    head = row["head"]
    paths = row["paths"]
    head_id = relation2id[head]

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
    fixed_context, predict, return_rules = build_prompt(head_formate, prompt_info_dict)
    current_prompt = fixed_context + predict + return_rules

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

                    formatted_string = ';'.join([f'{name}' for name in condicate])

                return_rules = return_rules.format(candidate_rels=formatted_string)

                temp_current_prompt = fixed_context + predict + return_rules

                few_shot_paths = check_prompt_length(
                    temp_current_prompt,
                    few_shot_samples, model
                )

                if not few_shot_paths:
                    raise ValueError("few_shot_paths is empty, head:{}".format(head))

                few_shot_paths = few_shot_paths + "\n\n"

                return_rules = "\n\n" + return_rules

                prompt = fixed_context + few_shot_paths + predict + return_rules
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

def generate_rule_for_zero(head, rdict, rule_path, model, args, prompt_info_zero_dict):
    relation2id = rdict.rel2idx
    all_rels = list(relation2id.keys())

    fixed_context, predict, return_rules_template = build_prompt_for_zero(head, prompt_info_zero_dict)
    return_rules = return_rules_template.format(candidate_rels=all_rels)
    current_prompt = fixed_context + predict + return_rules

    # 定义文件路径
    query_file_path = os.path.join(rule_path, f"{head}.query")
    txt_file_path = os.path.join(rule_path, f"{head}.txt")
    fail_file_path = os.path.join(rule_path, f"fail_{head}.txt")

    try:
        with open(query_file_path, "w") as fout_zero_query, open(txt_file_path, "w") as fout_zero_txt:
            for i in range(args.l):
                entry = f"Sample {i + 1} time:\n"
                fout_zero_query.write(entry + current_prompt + "\n")

                response = query(current_prompt, model)
                if response:
                    fout_zero_txt.write(entry + response + "\n")
                else:
                    raise ValueError("Failed to generate response.")
    except ValueError as e:
        with open(fail_file_path, "w") as fail_rule_file:
            fail_rule_file.write(current_prompt + "\n")
        print(e)  # Optional: Handle the exception as needed


def extract_and_expand_relations(args, path_content_list, similiary_rel_dict, relation_regex):
    """
    从提供的规则样本中随机抽取一定数量的规则，并扩展这些规则中的关系集合。

    :param args: 命名空间，包含参数f，表示要抽取的规则数量。
    :param path_content_list: 包含规则的列表。
    :param similiary_rel_dict: 一个字典，键为关系名，值为与该关系相似的关系集合。
    :param relation_regex: 用于从规则中提取关系的正则表达式。
    :return: 包含原始关系和相似关系的集合。
    """
    # 随机抽取样本
    few_shot_samples = random.sample(
        path_content_list, min(args.f, len(path_content_list))
    )

    # 从抽取的样本中提取关系
    relation_set = set()
    for rule in few_shot_samples:
        rule_body = rule.split('<-')[-1]
        matches = re.findall(relation_regex, rule_body)
        for match in matches:
            relation = match[0]
            relation_set.update([relation])

    # 扩展找到的关系集合，包括相似的关系
    similiary_rel_set = set()
    for rel_name in relation_set:
        similiary_rel_set.update(similiary_rel_dict[rel_name])

    # 合并原始关系集和相似关系集
    condicate = similiary_rel_set.union(relation_set)

    return condicate


def generate_rule_for_iteration_by_multi_thread(row, rdict, rule_path, kg_rules_path, model, args,
                                                relation_regex,
                                                similiary_rel_dict, kg_rules_path_with_valid):
    relation2id = rdict.rel2idx
    head = row["head"]
    rules = row["rules"]

    head_id = relation2id[head]

    valid_rules_name = load_json_data(kg_rules_path_with_valid)

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

    if args.based_rule_type ==  'low':
        # Build prompt excluding rules
        fixed_context, sampled_rules, predict, return_rules = build_prompt_based_low(
            head_formate, candidate_rels, args.is_zero, args, prompt_dict
        )
    else:
        fixed_context, predict, return_rules = build_prompt_based_high(
            head_formate, candidate_rels, args.is_zero, args, prompt_dict
        )

    kg_rules_dict = load_json_data(kg_rules_path)
    path_content_list = kg_rules_dict.get(head, None)
    file_name = head.replace("/", "-")
    with open(os.path.join(rule_path, f"{file_name}.txt"), "w") as rule_file, open(
            os.path.join(rule_path, f"{file_name}.query"), "w") as query_file:
        rule_file.write(f"Rule_head: {head}\n")
        for i in range(args.second):

            if path_content_list is not None:
                condicate = extract_and_expand_relations(args, path_content_list, similiary_rel_dict, relation_regex)
            else:
                condicate = set(all_rels)

            temp_valid_rules = valid_rules_name.get(head, '')
            valid_rules_with_head = random.sample(temp_valid_rules,  min(20, len(temp_valid_rules)))

            temp_rules = random.sample(rules, min(20, len(rules)))
            quitity_string = ''.join(temp_rules)
            quitity_string = quitity_string + '\n'

            valid_rules_string = ''.join(valid_rules_with_head)
            valid_rules_string = valid_rules_string + '\n'


            temp_current_prompt = fixed_context + quitity_string + valid_rules_string + predict

            formatted_string = iteration_check_prompt_length(
                temp_current_prompt,
                list(condicate), return_rules, model
            )

            return_rules = return_rules.format(candidate_rels=formatted_string)

            prompt = temp_current_prompt + return_rules
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

def generate_rule_for_unknown_relation_by_multi_thread(row, rdict, rule_path, kg_rules_path, model, args, relation_subgraph,
                                                relation_regex,
                                                similiary_rel_dict):
    relation2id = rdict.rel2idx
    head = row

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

    # Build prompt excluding rules
    fixed_context, predict, return_rules = build_prompt_for_unknown(
        head_formate, candidate_rels, args.is_zero, args, prompt_dict
    )

    file_name = head.strip()
    with open(os.path.join(rule_path, f"{file_name}.txt"), "w") as rule_file, open(
            os.path.join(rule_path, f"{file_name}.query"), "w") as query_file:
        rule_file.write(f"Rule_head: {head}\n")
        for i in range(args.second):
            # # Convert list elements to the desired string format
            # formatted_string = ';'.join([f'{name}' for name in condicate])

            formatted_string = unknown_check_prompt_length(
                fixed_context + predict, all_rels, return_rules, model
            )

            return_rules = return_rules.format(candidate_rels=formatted_string)


            prompt = fixed_context + predict + return_rules
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


def get_low_conf(low_conf_file_path, relation_regex, rdict):
    rule_dict = {}
    with open(low_conf_file_path, 'r') as fin_low:
        rules = fin_low.readlines()
        for rule in rules:
            if 'index' in rule:
                continue
            regrex_list = rule.split('<-')
            match = re.search(relation_regex, regrex_list[0])
            if match:
                head = match[1].strip()
                if head not in list(rdict.rel2idx.keys()):
                    raise ValueError(f"Not exist relation:{head}")

                if head not in rule_dict:
                    rule_dict[head] = []
                rule_dict[head].append(rule)

    rule_list = []
    for key, value in rule_dict.items():
        rule_list.append({'head': key, 'rules': value})

    return rule_list

def get_high_conf(high_conf_file_path, relation_regex, rdict):
    rule_dict = {}
    with open(high_conf_file_path, 'r') as fin_low:
        rules = fin_low.readlines()
        for rule in rules:
            if 'index' in rule:
                continue
            regrex_list = rule.split('<-')
            match = re.search(relation_regex, regrex_list[0])
            if match:
                head = match[1].strip()
                if head not in list(rdict.rel2idx.keys()):
                    raise ValueError(f"Not exist relation:{head}")

                if head not in rule_dict:
                    rule_dict[head] = []
                rule_dict[head].append(rule)

    rule_list = []
    for key, value in rule_dict.items():
        rule_list.append({'head': key, 'rules': value})

    return rule_list




def analysis_data(confidence_folder, kg_rules_path):
    with open(os.path.join(confidence_folder, 'hight_conf.txt'), 'r') as fin_hight, open(
            os.path.join(confidence_folder, 'low_conf.txt'), 'r') as fin_low:
        hight_rule_set = set()
        rules = fin_hight.readlines()
        for rule in rules:
            if "index" in rule:
                continue
            hight_rule_set.update([rule.strip()])

        low_rule_set = set()
        rules = fin_low.readlines()
        for rule in rules:
            if "index" in rule:
                continue
            low_rule_set.update([rule.strip()])

    rules_dict = load_json_data(kg_rules_path)

    all_rules = [item.strip() for sublist in rules_dict.values() for item in sublist]
    all_rules_set = set(all_rules)

    with open(os.path.join(confidence_folder, 'statistic.txt'), 'w') as fout_state:
        fout_state.write(f'valid_high:{len(hight_rule_set-all_rules_set)}\n')
        fout_state.write(f'valid_low:{len(low_rule_set-all_rules_set)}\n')


def load_data_and_paths(args):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)

    sampled_path_with_valid_dir = os.path.join(args.sampled_paths, args.dataset + '_valid')
    sampled_path_dir = os.path.join(args.sampled_paths, args.dataset)
    sampled_path = read_paths(os.path.join(sampled_path_dir, "closed_rel_paths.jsonl"))

    prompt_path = os.path.join(args.prompt_paths, 'common.json')
    prompt_path_for_zero = os.path.join(args.prompt_paths, 'zero.json')

    return dataset, sampled_path, sampled_path_with_valid_dir, sampled_path_dir, prompt_path, prompt_path_for_zero


def prepare_rule_heads(dataset, sampled_path):
    rule_head_without_zero = {rule['head'] for rule in sampled_path}
    rule_head_with_zero = set(dataset.rdict.rel2idx.keys()) - rule_head_without_zero
    return rule_head_without_zero, rule_head_with_zero


def determine_kg_rules_path(args, sampled_path_dir):
    if args.is_rel_name:
        return os.path.join(sampled_path_dir, "rules_name.json")
    else:
        return os.path.join(sampled_path_dir, "rules_id.json")


def load_configuration(dataset, sampled_path_dir, args):
    constant_config = load_json_data('./Config/constant.json')
    relation_regex = constant_config['relation_regex'][args.dataset]

    rdict = dataset.get_relation_dict()
    similarity_matrix = np.load(os.path.join(sampled_path_dir, "matrix.npy"))
    transformers_id2rel = load_json_data(os.path.join(sampled_path_dir, "transfomers_id2rel.json"))
    transformers_rel2id = load_json_data(os.path.join(sampled_path_dir, "transfomers_rel2id.json"))

    similar_rel_dict = get_topk_similiary_rel(args.topk, similarity_matrix, transformers_id2rel, transformers_rel2id)

    return rdict, relation_regex, similar_rel_dict


def create_directories(args):
    rule_path = os.path.join(
        args.rule_path, args.dataset,
        f"{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}"
    )
    os.makedirs(rule_path, exist_ok=True)

    filter_rule_path = os.path.join(
        args.rule_path, args.dataset,
        f"copy_{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}"
    )
    if not os.path.exists(filter_rule_path):
        os.makedirs(filter_rule_path)
    else:
        clear_folder(filter_rule_path)

    return rule_path, filter_rule_path


def main(args, LLM):
    dataset, sampled_path, sampled_path_with_valid_dir, sampled_path_dir, prompt_path, prompt_path_for_zero = load_data_and_paths(args)
    rule_head_without_zero, rule_head_with_zero = prepare_rule_heads(dataset, sampled_path)
    kg_rules_path = determine_kg_rules_path(args, sampled_path_dir)
    rdict, relation_regex, similar_rel_dict = load_configuration(dataset, sampled_path_dir, args)
    rule_path, filter_rule_path = create_directories(args)

    prompt_info_dict = load_json_data(prompt_path)
    prompt_info_zero_dict = load_json_data(prompt_path_for_zero)

    model = LLM(args)
    print("Preparing pipeline for inference...")
    model.prepare_for_inference()

    llm_rule_generate(
        args, filter_rule_path, kg_rules_path, model, rdict, relation_regex, rule_path,
        sampled_path, similar_rel_dict, sampled_path_with_valid_dir, rule_head_with_zero, prompt_info_dict, prompt_info_zero_dict
    )

def llm_rule_generate(args, filter_rule_path, kg_rules_path, model, rdict, relation_regex, rule_path,
                      sampled_path, similiary_rel_dict, sampled_path_with_valid_dir, rule_head_with_zero, prompt_info_dict, prompt_info_zero_dict):
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
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict,
                        prompt_info_dict=prompt_info_dict
                    ),
                    sampled_path,
                ),
                total=len(sampled_path),
        ):
            pass

    with ThreadPool(args.n) as p:
        for _ in tqdm(
                p.imap_unordered(
                    partial(
                        generate_rule_for_zero,
                        rdict=rdict,
                        rule_path=rule_path,
                        model=model,
                        args=args,
                        prompt_info_zero_dict=prompt_info_zero_dict
                    ),
                    rule_head_with_zero,
                ),
                total=len(rule_head_with_zero),
        ):
            pass

    for input_filepath in glob.glob(os.path.join(rule_path, "fail_*.txt")):
        filename = input_filepath.split('/')[-1].split('fail_')[-1]
        with open(input_filepath, 'r') as fin, open(os.path.join(rule_path, filename), 'w') as fout:
            content = fin.read()
            response = model.generate_sentence(content)
            if response is not None:
                fout.write(response + "\n")
            else:
                print(f"Error:{filename}")

    #valid dataset中的rule
    kg_rules_path_with_valid = os.path.join(sampled_path_with_valid_dir, "rules_name.json")

    #分析LLM第一次生成规则的情况
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

    constant_config = load_json_data('./Config/constant.json')
    relation_regex = constant_config['relation_regex'][args.dataset]
    if args.is_rel_name is True:
        copy_files(rule_path, filter_rule_path, 'txt')
    else:
        process_rules_files(rule_path, filter_rule_path, rdict, relation_regex, error_file_path)

    model.gen_rule_statistic(rule_path, statistics_file_path)

    output_clean_folder = os.path.join(args.rule_path, args.dataset, 'clean')
    if not os.path.exists(output_clean_folder):
        os.makedirs(output_clean_folder)
    else:
        clear_folder(output_clean_folder)

    output_filter_train_folder = os.path.join(args.rule_path, args.dataset, 'filter', 'train')
    if not os.path.exists(output_filter_train_folder):
        os.makedirs(output_filter_train_folder)
    else:
        clear_folder(output_filter_train_folder)

    #filter文件夹保存历次迭代生成的rule，包括high conf和low conf文件
    output_filter_eva_folder = os.path.join(args.rule_path, args.dataset, 'filter', 'eva')
    if not os.path.exists(output_filter_eva_folder):
        os.makedirs(output_filter_eva_folder)
    else:
        clear_folder(output_filter_eva_folder)

    output_filter_train_eva_folder = os.path.join(args.rule_path, args.dataset, 'filter', 'train_eva')
    if not os.path.exists(output_filter_train_eva_folder):
        os.makedirs(output_filter_train_eva_folder)
    else:
        clear_folder(output_filter_train_eva_folder)

    # evaluation文件夹记录每次迭代时，中间生成的confidence.json
    output_eva_train_folder = os.path.join(args.rule_path, args.dataset, 'evaluation', 'train')
    if not os.path.exists(output_eva_train_folder):
        os.makedirs(output_eva_train_folder)
    else:
        clear_folder(output_eva_train_folder)

    output_eva_eva_folder = os.path.join(args.rule_path, args.dataset, 'evaluation', 'eva')
    if not os.path.exists(output_eva_eva_folder):
        os.makedirs(output_eva_eva_folder)
    else:
        clear_folder(output_eva_eva_folder)

    output_eva_train_eva_folder = os.path.join(args.rule_path, args.dataset, 'evaluation', 'train_eva')
    if not os.path.exists(output_eva_train_eva_folder):
        os.makedirs(output_eva_train_eva_folder)
    else:
        clear_folder(output_eva_train_eva_folder)

    #临时文件夹，保存关于LLM的request和response
    iteration_rule_file_path = os.path.join(args.rule_path, args.dataset, 'iteration')
    if not os.path.exists(iteration_rule_file_path):
        os.makedirs(iteration_rule_file_path)
    else:
        clear_folder(iteration_rule_file_path)

    #保存每次迭代时，LLM的response
    for i in range(args.num_iter + 1):
        temp_only_txt_file_path = os.path.join(args.rule_path, args.dataset, f'only_txt_{i}')
        clear_folder(temp_only_txt_file_path)

    #临时文件夹，保存关于LLM的response
    iteration_only_txt_file_path = os.path.join(args.rule_path, args.dataset, 'only_txt')
    if not os.path.exists(iteration_only_txt_file_path):
        os.makedirs(iteration_only_txt_file_path)
    else:
        clear_folder(iteration_only_txt_file_path)

    copy_folder_contents(filter_rule_path, iteration_only_txt_file_path)

    for i in range(args.num_iter):
        temp_only_txt_file_path = os.path.join(args.rule_path, args.dataset, f'only_txt_{i}')
        copy_folder_contents(iteration_only_txt_file_path, temp_only_txt_file_path)

        start_time = time.time()

        output_rules_folder_dir = clean(args, model, iteration_only_txt_file_path, output_clean_folder)
        conf_folder = None
        if args.bgkg == 'train':
            train_rule_set = evaluation(args, output_rules_folder_dir, output_eva_train_folder, 'train', index=i)
            filter_rules_based_confidence(train_rule_set, args.min_conf, output_filter_train_folder, i)
            conf_folder = output_filter_train_folder
        elif args.bgkg == 'valid':
            env_rule_set = evaluation(args, output_rules_folder_dir, output_eva_eva_folder, 'eva', index=i)
            filter_rules_based_confidence(env_rule_set, args.min_conf, output_filter_eva_folder, i)
            conf_folder = output_filter_eva_folder
        elif args.bgkg == 'train_valid':
            train_env_rule_set = evaluation(args, output_rules_folder_dir, output_eva_train_eva_folder, 'train_eva',
                                            index=i)
            filter_rules_based_confidence(train_env_rule_set, args.min_conf, output_filter_train_eva_folder, i)
            conf_folder = output_filter_train_eva_folder

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60

        print(f"程序运行时间：{elapsed_minutes}分钟")

        if args.is_high is False:
           conf = get_low_conf(os.path.join(conf_folder, 'temp_low_conf.txt'), relation_regex, rdict)
        else:
           conf = get_high_conf(os.path.join(conf_folder, 'temp_hight_conf.txt'), relation_regex, rdict)

        clear_folder(iteration_only_txt_file_path)
        clear_folder(iteration_rule_file_path)
        gen_rules_iteration(args, kg_rules_path, model, rdict, relation_regex,
                            iteration_rule_file_path,
                            conf, similiary_rel_dict, kg_rules_path_with_valid)
        copy_files(iteration_rule_file_path, iteration_only_txt_file_path, 'txt')

    temp_only_txt_file_path = os.path.join(args.rule_path, args.dataset, f'only_txt_{args.num_iter}')
    copy_folder_contents(iteration_only_txt_file_path, temp_only_txt_file_path)

    output_rules_folder_dir = clean(args, model, iteration_only_txt_file_path, output_clean_folder)

    source_rule_path = None
    if args.bgkg == 'train':
        train_rule_set = evaluation(args, output_rules_folder_dir, output_eva_train_folder, 'train', index=args.num_iter)
        filter_rules_based_confidence(train_rule_set, args.min_conf, output_filter_train_folder, args.num_iter)
        analysis_data(output_filter_train_folder, kg_rules_path)
        source_rule_path = output_filter_train_folder
    elif args.bgkg == 'valid':
        env_rule_set = evaluation(args, output_rules_folder_dir, output_eva_eva_folder, 'eva', index=args.num_iter)
        filter_rules_based_confidence(env_rule_set, args.min_conf, output_filter_eva_folder, args.num_iter)
        analysis_data(output_filter_eva_folder, kg_rules_path)
        source_rule_path = output_filter_eva_folder
    elif args.bgkg == 'train_valid':
        train_env_rule_set = evaluation(args, output_rules_folder_dir, output_eva_train_eva_folder, 'train_eva',
                                        index=args.num_iter)
        filter_rules_based_confidence(train_env_rule_set, args.min_conf, output_filter_train_eva_folder, args.num_iter)
        analysis_data(output_filter_train_eva_folder, kg_rules_path)
        source_rule_path = output_filter_train_eva_folder

    final_sumary_file_path = os.path.join('gen_rules_iteration', args.dataset, 'final_summary')
    os.makedirs(final_sumary_file_path, exist_ok=True)
    if args.rule_domain == 'high':
        high_train_eva_file_path = os.path.join(source_rule_path, 'hight_conf.txt')
        with open(high_train_eva_file_path, 'r') as fin_high:
            high_unique_strings = set(fin_high.read().split())

        unique_strings = high_unique_strings

    elif args.rule_domain == 'iteration':
        high_train_eva_file_path = os.path.join(source_rule_path, 'hight_conf.txt')
        with open(high_train_eva_file_path, 'r') as fin_high:
            high_unique_strings = set(fin_high.read().split())

        low_train_eva_file_path = os.path.join(source_rule_path, 'low_conf.txt')
        with open(low_train_eva_file_path, 'r') as fin_low:
            low_unique_strings = set(fin_low.read().split())

        unique_strings = low_unique_strings.union(high_unique_strings)

    else:
        pass

    with open(os.path.join(final_sumary_file_path, 'rules.txt'), 'w') as fout_final:
        for rule in unique_strings:
            fout_final.write(f'{rule}\n')

def gen_rules_iteration(args, kg_rules_path, model, rdict, relation_regex, rule_path, conf,
                        similiary_rel_dict, kg_rules_path_with_valid):
    with ThreadPool(args.n) as p:
        for _ in tqdm(
                p.imap_unordered(
                    partial(
                        generate_rule_for_iteration_by_multi_thread,
                        rdict=rdict,
                        rule_path=rule_path,
                        kg_rules_path=kg_rules_path,
                        model=model,
                        args=args,
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict,
                        kg_rules_path_with_valid = kg_rules_path_with_valid
                    ),
                    conf,
                ),
                total=len(conf),
        ):
            pass

def gen_rules_for_unknown_relation(args, kg_rules_path, model, rdict, relation_regex, relation_subgraph, rule_path, low_conf,
                        similiary_rel_dict):
    with ThreadPool(args.n) as p:
        for _ in tqdm(
                p.imap_unordered(
                    partial(
                        generate_rule_for_unknown_relation_by_multi_thread,
                        rdict=rdict,
                        rule_path=rule_path,
                        kg_rules_path=kg_rules_path,
                        model=model,
                        args=args,
                        relation_subgraph=relation_subgraph,
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict
                    ),
                    low_conf,
                ),
                total=len(low_conf),
        ):
            pass


def filter_rules_based_confidence(rule_set, min_conf, output_folder, index):
    with open(os.path.join(output_folder, 'hight_conf.txt'), 'a') as fout_hight, open(
            os.path.join(output_folder, 'low_conf.txt'), 'a') as fout_low, open(
        os.path.join(output_folder, 'temp_hight_conf.txt'), 'w') as fout_temp_hight, open(
        os.path.join(output_folder, 'temp_low_conf.txt'), 'w') as fout_temp_low:
        fout_hight.write(f"index:{index}\n")
        fout_low.write(f"index:{index}\n")
        for rule in rule_set:
            confidence = float(rule.split('&')[-1].strip())
            temp_rule = rule.split('&')[:-1]
            rule_without_confidence = '&'.join(temp_rule)
            if confidence > min_conf:
                fout_hight.write(rule_without_confidence + '\n')
                fout_temp_hight.write(rule_without_confidence + '\n')
            else:
                fout_low.write(rule_without_confidence + '\n')
                fout_temp_low.write(rule_without_confidence + '\n')


def evaluation(args, output_rules_folder_dir, output_evaluation_folder, dataset_type, index=0):
    is_merge = args.is_merge
    dataset_dir = "./datasets/" + args.dataset + "/"
    data = Grapher(dataset_dir)

    if dataset_type == 'train':
        temporal_walk = Temporal_Walk(np.array(data.train_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)
    elif dataset_type == 'eva':
        temporal_walk = Temporal_Walk(np.array(data.valid_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)
    else:
        temporal_walk = Temporal_Walk(np.array(data.valid_idx.tolist() + data.train_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)

    rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, args.dataset)
    rule_path = output_rules_folder_dir
    constant_config = load_json_data('./Config/constant.json')
    relation_regex = constant_config['relation_regex'][args.dataset]

    rules_var_path = os.path.join("sampled_path", args.dataset, "original", "rules_var.json")
    rules_var_dict = load_json_data(rules_var_path)

    if args.is_only_with_original_rules:
        for key, value in rules_var_dict.items():
            temp_var = {}
            temp_var['head_rel'] = value['head_rel']
            temp_var['body_rels'] = value['body_rels']
            temp_var["var_constraints"] = value["var_constraints"]
            if temp_var not in rl.original_found_rules:
                rl.original_found_rules.append(temp_var.copy())
                rl.update_rules_dict(value)
                rl.num_original += 1
    else:
        llm_gen_rules_list, fail_calc_confidence = calculate_confidence(rule_path, data.relation2id, data.inv_relation_id, rl, relation_regex,
                                                  rules_var_dict, is_merge, is_has_confidence=False)

    rules_statistics(rl.rules_dict)

    if args.is_only_with_original_rules:
        dir_path = output_evaluation_folder
        confidence_file_path = os.path.join(dir_path, 'original_confidence.json')
        save_json_data(rl.rules_dict, confidence_file_path)
    else:
        if is_merge is True:
            original_rules_set = set(list(rules_var_dict.keys()))
            llm_gen_rules_set = set(llm_gen_rules_list)
            for idx, rule_chain in enumerate(original_rules_set - llm_gen_rules_set):
                rule = rules_var_dict[rule_chain]
                rl.update_rules_dict(rule)

            rules_statistics(rl.rules_dict)

            dir_path = output_evaluation_folder
            confidence_file_path = os.path.join(dir_path, 'merge_confidence.json')
            save_json_data(rl.rules_dict, confidence_file_path)
        else:
            dir_path = output_evaluation_folder
            confidence_file_path = os.path.join(dir_path, f'{index}_confidence.json')
            save_json_data(rl.rules_dict, confidence_file_path)

            fail_confidence_file_path = os.path.join(dir_path, 'fail_confidence.txt')
            with open(fail_confidence_file_path, 'a') as fout:
                for fail_rule in fail_calc_confidence:
                    fout.write(f'{fail_rule}\n')

    return set(llm_gen_rules_list)


def calculate_confidence(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge,
                         is_has_confidence=False):
    llm_gen_rules_list = []
    fail_calc_confidence = []
    for input_filepath in glob.glob(os.path.join(rule_path, "*_cleaned_rules.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):
                try:
                    if is_has_confidence:
                        try:
                            confidence = float(rule.split('&')[-1].strip())
                            temp_rule = rule.split('&')[:-1]
                            rule_without_confidence = '&'.join(temp_rule)
                            rule_without_confidence = rule_without_confidence.strip()
                            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)

                            rule_with_confidence = rl.create_rule_for_merge_for_iteration(walk, confidence,
                                                                                              rule_without_confidence,
                                                                                              rules_var_dict,
                                                                                              is_merge)
                            llm_gen_rules_list.append(rule_with_confidence + "\n")
                        except Exception as e:
                            print(e)
                            fail_calc_confidence.append(rule + "\n")
                    else:
                        try:
                            confidence = 0
                            temp_rule = rule.split('&')
                            rule_without_confidence = '&'.join(temp_rule)
                            rule_without_confidence = rule_without_confidence.strip()
                            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
                            rule_with_confidence = rl.create_rule_for_merge_for_iteration(walk, confidence,
                                                                                          rule_without_confidence,
                                                                                          rules_var_dict,
                                                                                          is_merge)
                            llm_gen_rules_list.append(rule_with_confidence + "\n")
                        except Exception as e:
                            print(e)
                            fail_calc_confidence.append(rule + "\n")



                except Exception as e:
                    print(f"Error processing rule: {rule}")
                    traceback.print_exc()  # 打印异常的详细信息和调用栈

    return llm_gen_rules_list, fail_calc_confidence


def process_rule(rule, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge, is_has_confidence):
    try:
        if is_has_confidence:
            confidence = float(rule.split('&')[-1].strip())
            temp_rule = rule.split('&')[:-1]
            rule_without_confidence = '&'.join(temp_rule).strip()
            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
            rule_with_confidence = rl.create_rule_for_merge_for_iteration(walk, confidence,
                                                                          rule_without_confidence,
                                                                          rules_var_dict, is_merge)
            return rule_with_confidence + "\n", None
        else:
            confidence = 0
            temp_rule = rule.split('&')
            rule_without_confidence = '&'.join(temp_rule).strip()
            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
            rule_with_confidence = rl.create_rule_for_merge_for_iteration(walk, confidence,
                                                                          rule_without_confidence,
                                                                          rules_var_dict, is_merge)
            return rule_with_confidence + "\n", None
    except Exception as e:
        return None, rule + "\n"

def calculate_confidence_O(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge,
                         is_has_confidence=False):
    llm_gen_rules_list = []
    fail_calc_confidence = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for input_filepath in glob.glob(os.path.join(rule_path, "*_cleaned_rules.txt")):
            with open(input_filepath, 'r') as f:
                rules = f.readlines()
                for rule in rules:
                    future = executor.submit(process_rule, rule, relation2id, inv_relation_id, rl, relation_regex,
                                             rules_var_dict, is_merge, is_has_confidence)
                    futures.append(future)

        for future in futures:
            result, error = future.result()
            if result:
                llm_gen_rules_list.append(result)
            if error:
                fail_calc_confidence.append(error)

    return llm_gen_rules_list, fail_calc_confidence

def calculate_confidence_1(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge,
                         is_has_confidence=False, num_threads=10):
    llm_gen_rules_list = []
    fail_calc_confidence = []

    all_rules = []
    for input_filepath in glob.glob(os.path.join(rule_path, "*_cleaned_rules.txt")):
        with open(input_filepath, 'r') as f:
            all_rules.extend(f.readlines())

    with ThreadPool(num_threads) as p:
        for result, error in tqdm(p.imap_unordered(partial(process_rule, relation2id=relation2id,
                                                           inv_relation_id=inv_relation_id, rl=rl,
                                                           relation_regex=relation_regex,
                                                           rules_var_dict=rules_var_dict,
                                                           is_merge=is_merge,
                                                           is_has_confidence=is_has_confidence),
                                                   all_rules),
                                  total=len(all_rules)):
            if result:
                llm_gen_rules_list.append(result)
            if error:
                fail_calc_confidence.append(error)

    return llm_gen_rules_list, fail_calc_confidence



def get_walk(rule, relation2id, inv_relation_id, regex):
    head_body = rule.split('<-')
    rule_head_full_name = head_body[0].strip()
    condition_string = head_body[1].strip()

    # 定义正则表达式
    relation_regex = regex

    # 提取规则头的关系、主语和宾语
    match = re.search(relation_regex, rule_head_full_name)
    head_relation_name, head_subject, head_object, head_timestamp = match.groups()[:4]

    # 提取规则体的关系和实体
    matches = re.findall(relation_regex, condition_string)
    entities = [head_object] + [match[1].strip() for match in matches[:-1]] + [matches[-1][1].strip(),
                                                                               matches[-1][2].strip()]

    relation_ids = [relation2id[head_relation_name]] + [relation2id[match[0].strip()] for match in matches]

    # 反转除第一个元素外的列表
    entities = entities[:1] + entities[1:][::-1]
    relation_ids = relation_ids[:1] + [inv_relation_id[x] for x in relation_ids[:0:-1]]

    # 构造结果字典
    result = {
        'entities': entities,
        'relations': relation_ids
    }

    return result

def clean(args, llm_model, filter_rule_path, output_folder):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Dataset(data_root=data_path, inv=True)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    input_folder = filter_rule_path

    output_statistic_folder_dir = os.path.join(output_folder, 'clean_statistics')
    if not os.path.exists(output_statistic_folder_dir):
        os.makedirs(output_statistic_folder_dir)

    output_rules_folder_dir = os.path.join(output_folder, 'rules')
    if not os.path.exists(output_rules_folder_dir):
        os.makedirs(output_rules_folder_dir)
    else:
        clear_folder(output_rules_folder_dir)

    #分析clean过程中success与error的情况
    output_error_file_path = os.path.join(output_statistic_folder_dir, 'error.txt')
    output_suc_file_path = os.path.join(output_statistic_folder_dir, 'suc.txt')
    with open(output_error_file_path, 'a') as fout_error, open(output_suc_file_path, 'a') as fout_suc:
        num_error, num_suc = clean_processing(all_rels, args, fout_error, input_folder, llm_model,
                                              output_rules_folder_dir,
                                              fout_suc)
        fout_error.write(f"The number of cleaned rules is {num_error}\n")
        fout_suc.write(f"The number of retain rules is {num_suc}\n")

    return output_rules_folder_dir


def clean_processing(all_rels, args, fout_error, input_folder, llm_model, output_folder, fout_suc):
    constant_config = load_json_data('./Config/constant.json')
    rule_start_with_regex = constant_config["rule_start_with_regex"]
    replace_regex = constant_config["replace_regex"]
    relation_regex = constant_config["relation_regex"][args.dataset]
    num_error = 0
    num_suc = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt") and "query" not in filename and filename.startswith("fail") is False:
            input_filepath = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            summarized_filepath = os.path.join(output_folder, f"{name}_summarized_rules.txt")
            clean_filename = name + '_cleaned_rules.txt'
            clean_filepath = os.path.join(output_folder, clean_filename)

            if not args.clean_only:
                # Step 1: Summarize rules from the input file
                print("Start summarize: ", filename)
                # Summarize rules
                summarized_rules = summarize_rule(input_filepath, llm_model, args, rule_start_with_regex, replace_regex)
                print("write file", summarized_filepath)
                with open(summarized_filepath, "w") as f:
                    f.write('\n'.join(summarized_rules))

            # Step 2: Clean summarized rules and keep format
            print(f"Clean file {summarized_filepath} with keeping the format")
            cleaned_rules, num, num_0 = clean_rules(summarized_filepath, all_rels, relation_regex, fout_error, fout_suc)
            num_error = num_error + num
            num_suc = num_suc + num_0

            if len(cleaned_rules) != 0:
                with open(clean_filepath, "w") as f:
                    f.write('\n'.join(cleaned_rules))
    return num_error, num_suc


def extract_rules(content_list, rule_start_with_regex, replace_regex):
    """ Extract the rules in the content without any explanation and the leading number if it has."""
    rule_pattern = re.compile(rule_start_with_regex)
    extracted_rules = [s.strip() for s in content_list if rule_pattern.match(s)]
    number_pattern = re.compile(replace_regex)
    cleaned_rules = [number_pattern.sub('', s) for s in extracted_rules]
    return list(set(cleaned_rules))  # Remove duplicates by converting to set and back to list


def summarize_rules_prompt(relname, k):
    """
    Generate prompt for the relation in the content_list
    """

    if k != 0:
        prompt = f'\n\nPlease identify the most important {k} rules from the following rules for the rule head: "{relname}(X,Y,T)". '
    else:  # k ==0
        prompt = f'\n\nPlease identify as many of the most important rules for the rule head: "{relname}(X,Y,T)" as possible. '

    prompt += 'You can summarize the rules that have similar meanings as one rule, if you think they are important. ' \
              'Return the rules only without any explanations. '
    return prompt

def summarize_rule_for_unkown(file, llm_model, args, rule_start_with_regex, replace_regex):
    """
    Summarize the rules
    """
    with open(file, 'r') as f:  # Load files
        content = f.read()
        rel_name = os.path.splitext(file)[0].split('/')[-1]

    content_list = content.split('\n')
    rule_list = extract_rules(content_list, rule_start_with_regex,
                              replace_regex)  # Extract rules and remove any explanations
    if not args.force_summarize or llm_model is None:  # just return the whole rule_list
        return rule_list
    else:  # Do summarization and correct the spelling error
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(summarize_prompt, args.model_name)
        list_of_rule_lists = shuffle_split_path_list(rule_list, summarize_prompt_len, args.model_name)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            print('prompt: ', message)
            response = query(message, llm_model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list, rule_start_with_regex,
                                       replace_regex)  # Extract rules and remove any explanations from summarized response

        return response_rules

def summarize_rule(file, llm_model, args, rule_start_with_regex, replace_regex):
    """
    Summarize the rules
    """
    with open(file, 'r') as f:  # Load files
        content = f.read()
        rel_name = os.path.splitext(file)[0].split('/')[-1]

    content_list = content.split('\n')
    rule_list = extract_rules(content_list, rule_start_with_regex,
                              replace_regex)  # Extract rules and remove any explanations
    if not args.force_summarize or llm_model is None:  # just return the whole rule_list
        return rule_list
    else:  # Do summarization and correct the spelling error
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(summarize_prompt, args.model_name)
        list_of_rule_lists = shuffle_split_path_list(rule_list, summarize_prompt_len, args.model_name)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            print('prompt: ', message)
            response = query(message, llm_model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list, rule_start_with_regex,
                                       replace_regex)  # Extract rules and remove any explanations from summarized response

        return response_rules


def modify_process(temp_rule, relation_regex):
    regrex_list = temp_rule.split('&')
    for idx, regrex in enumerate(regrex_list):
        match = re.search(relation_regex, regrex)
        if match:
            relation_name = match[1].strip()
            subject = match[2].strip()
            object = match[3].strip()
            timestamp = match[4].strip()



def clean_rules_for_unknown(summarized_file_path, all_rels, relation_regex, fout_error, fout_suc):
    """
    Clean error rules and remove rules with error relation.
    """
    num_error = 0
    num_suc = 0
    with open(summarized_file_path, 'r') as f:
        input_rules = [line.strip() for line in f]
    cleaned_rules = list()
    # Correct spelling error/grammar error for the relation in the rules and Remove rules with error relation.
    for input_rule in input_rules:
        if input_rule == "":
            continue
        rule_list = []
        temp_rule = re.sub(r'\s*<-\s*', '&', input_rule)
        regrex_list = temp_rule.split('&')
        last_subject = None
        final_object = None
        time_squeque = []
        final_time = None
        is_save = True
        is_check = True
        try:
            for idx, regrex in enumerate(regrex_list):
                match = re.search(relation_regex, regrex)
                if match:
                    relation_name = match[1].strip()
                    subject = match[2].strip()
                    object = match[3].strip()
                    timestamp = match[4].strip()

                    if timestamp[1:].isdigit() is False:
                        correct_rule = modify_process(temp_rule, relation_regex)
                        is_check = False
                        break

                    if relation_name not in all_rels:
                        best_match = get_close_matches(relation_name, all_rels, n=1)
                        if not best_match:
                            print(f"Cannot correctify this rule, head not in relation:{input_rule}\n")
                            fout_error.write(f"Cannot correctify this rule, head not in relation:{input_rule}\n")
                            is_save = False
                            num_error = num_error + 1
                            break
                        relation_name = best_match[0].strip()

                    rule_list.append(f'{relation_name}({subject},{object},{timestamp})')

                    if idx == 0:
                        head_subject = subject
                        head_object = object
                        head_subject = head_subject

                        last_subject = head_subject
                        final_object = head_object

                        final_time = int(timestamp[1:])
                    else:
                        if last_subject == subject:
                            last_subject = object
                        else:
                            print(f"Error: Rule {input_rule} does not conform to the definition of chain rule.")
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                        time_squeque.append(int(timestamp[1:]))

                    if idx == len(regrex_list) - 1:
                        if last_subject != final_object:
                            print(f"Error: Rule {input_rule} does not conform to the definition of chain rule.")
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                else:
                    print(f"Error: rule {input_rule}")
                    fout_error.write(f"Error: rule {input_rule}\n")
                    num_error = num_error + 1
                    is_save = False
                    break

            if is_check is True:
                if all(time_squeque[i] <= time_squeque[i + 1] for i in range(len(time_squeque) - 1)) is False:
                    print(f"Error: Rule {input_rule} time_squeque is error.")
                    fout_error.write(f"Error: Rule {input_rule} time_squeque is error.\n")
                    num_error = num_error + 1
                    is_save = False
                elif final_time < time_squeque[-1]:
                    print(f"Error: Rule {input_rule} time_squeque is error.")
                    fout_error.write(f"Error: Rule {input_rule} time_squeque is error.\n")
                    num_error = num_error + 1
                    is_save = False

            if is_save:
                correct_rule = '&'.join(rule_list).strip().replace('&', '<-', 1)
                cleaned_rules.append(correct_rule)
                fout_suc.write(correct_rule + '\n')
                num_suc = num_suc + 1

        except Exception as e:
            print(f"Processing {input_rule} failed.\n Error: {str(e)}")
            fout_error.write(f"Processing {input_rule} failed.\n Error: {str(e)}\n")
            num_error = num_error + 1
    return cleaned_rules, num_error, num_suc

def clean_rules(summarized_file_path, all_rels, relation_regex, fout_error, fout_suc):
    """
    Clean error rules and remove rules with error relation.
    """
    num_error = 0
    num_suc = 0
    with open(summarized_file_path, 'r') as f:
        input_rules = [line.strip() for line in f]
    cleaned_rules = list()
    # Correct spelling error/grammar error for the relation in the rules and Remove rules with error relation.
    for input_rule in input_rules:
        if input_rule == "":
            continue
        rule_list = []
        temp_rule = re.sub(r'\s*<-\s*', '&', input_rule)
        regrex_list = temp_rule.split('&')
        last_subject = None
        final_object = None
        time_squeque = []
        final_time = None
        is_save = True
        try:
            for idx, regrex in enumerate(regrex_list):
                match = re.search(relation_regex, regrex)
                if match:
                    relation_name = match[1].strip()
                    subject = match[2].strip()
                    object = match[3].strip()
                    timestamp = match[4].strip()

                    if timestamp[1:].isdigit() is False:
                        print(f"Error: Rule {input_rule}:{timestamp} is not digit")
                        fout_error.write(f"Error: Rule {input_rule}:{timestamp} is not digit\n")
                        num_error = num_error + 1
                        is_save = False
                        break

                    if relation_name not in all_rels:
                        best_match = get_close_matches(relation_name, all_rels, n=1)
                        if not best_match:
                            print(f"Cannot correctify this rule, head not in relation:{input_rule}\n")
                            fout_error.write(f"Cannot correctify this rule, head not in relation:{input_rule}\n")
                            is_save = False
                            num_error = num_error + 1
                            break
                        relation_name = best_match[0].strip()

                    rule_list.append(f'{relation_name}({subject},{object},{timestamp})')

                    if idx == 0:
                        head_subject = subject
                        head_object = object
                        head_subject = head_subject

                        last_subject = head_subject
                        final_object = head_object

                        final_time = int(timestamp[1:])
                    else:
                        if last_subject == subject:
                            last_subject = object
                        else:
                            print(f"Error: Rule {input_rule} does not conform to the definition of chain rule.")
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                        time_squeque.append(int(timestamp[1:]))

                    if idx == len(regrex_list) - 1:
                        if last_subject != final_object:
                            print(f"Error: Rule {input_rule} does not conform to the definition of chain rule.")
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                else:
                    print(f"Error: rule {input_rule}")
                    fout_error.write(f"Error: rule {input_rule}\n")
                    num_error = num_error + 1
                    is_save = False
                    break

            if all(time_squeque[i] <= time_squeque[i + 1] for i in range(len(time_squeque) - 1)) is False:
                print(f"Error: Rule {input_rule} time_squeque is error.")
                fout_error.write(f"Error: Rule {input_rule} time_squeque is error.\n")
                num_error = num_error + 1
                is_save = False
            elif final_time < time_squeque[-1]:
                print(f"Error: Rule {input_rule} time_squeque is error.")
                fout_error.write(f"Error: Rule {input_rule} time_squeque is error.\n")
                num_error = num_error + 1
                is_save = False

            if is_save:
                correct_rule = '&'.join(rule_list).strip().replace('&', '<-', 1)
                cleaned_rules.append(correct_rule)
                fout_suc.write(correct_rule + '\n')
                num_suc = num_suc + 1

        except Exception as e:
            print(f"Processing {input_rule} failed.\n Error: {str(e)}")
            fout_error.write(f"Processing {input_rule} failed.\n Error: {str(e)}\n")
            num_error = num_error + 1
    return cleaned_rules, num_error, num_suc

def parse_arguments():
    parser = argparse.ArgumentParser(description="KGC rule generation parameters")
    parser.add_argument("--data_path", type=str, default="datasets", help="Data directory")
    parser.add_argument("--dataset", type=str, default="family", help="Dataset name")
    parser.add_argument("--sampled_paths", type=str, default="sampled_path", help="Sampled path directory")
    parser.add_argument("--prompt_paths", type=str, default="prompt", help="Sampled path directory")
    parser.add_argument("--rule_path", type=str, default="gen_rules_iteration", help="Path to rule file")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--is_zero", action="store_true", help="Enable zero-shot rule generation")
    parser.add_argument("-k", type=int, default=0, help="Number of generated rules, 0 denotes as much as possible")
    parser.add_argument("-f", type=int, default=5, help="Few-shot number")
    parser.add_argument("-topk", type=int, default=20, help="Top-k paths")
    parser.add_argument("-n", type=int, default=5, help="Number of threads")
    parser.add_argument("-l", type=int, default=3, help="Sample times for generating k rules")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for files")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode")
    parser.add_argument("--is_rel_name", type=str_to_bool, default='yes', help="Enable relation names")
    parser.add_argument("--select_with_confidence", type=str_to_bool, default='no', help="Select with confidence")
    parser.add_argument('--clean_only', action='store_true', help='Load summarized rules and clean rules only')
    parser.add_argument('--force_summarize', action='store_true', help='Force summarize rules')
    parser.add_argument("--is_merge", type=str_to_bool, default='no', help="Enable merge")
    parser.add_argument("--transition_distr", type=str, default="exp", help="Transition distribution")
    parser.add_argument("--is_only_with_original_rules", type=str_to_bool, default='no', help="Use only original rules")
    parser.add_argument("--is_high", type=str_to_bool, default='No', help="Enable high mode")
    parser.add_argument("--min_conf", type=float, default=0.01, help="Minimum confidence")
    parser.add_argument("--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-second", type=int, default=3, help="Second sampling times for generating k rules")
    parser.add_argument("--bgkg", type=str, default="valid", choices=['train', 'train_valid', 'valid', 'test'], help="Background knowledge graph")
    parser.add_argument("--based_rule_type", type=str, default='low', choices=['low', 'high'], help="Base rule type")
    parser.add_argument("--rule_domain", type=str, default='iteration', choices=['iteration', 'high', 'all'], help="Rule domain")

    args, _ = parser.parse_known_args()
    return args, parser

if __name__ == "__main__":
    args, parser = parse_arguments()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
