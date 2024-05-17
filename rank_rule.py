import os
import argparse
import glob
import re
import traceback
import numpy as np

from grapher import Grapher
from rule_learning import Rule_Learner, rules_statistics
from temporal_walk import Temporal_Walk
from utils import save_json_data, load_json_data


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


def calculate_confidence_original(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge):
    llm_gen_rules_list = []
    for input_filepath in glob.glob(os.path.join(rule_path, "*_cleaned_rules.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):
                try:
                    confidence = float(rule.split('&')[-1].strip())
                    temp_rule = rule.split('&')[:-1]
                    rule_without_confidence = '&'.join(temp_rule)
                    walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
                    rl.create_rule_for_merge(walk, confidence, rule_without_confidence, rules_var_dict, is_merge)
                    llm_gen_rules_list.append(rule_without_confidence)
                except Exception as e:
                    print(f"Error processing rule: {rule}")
                    traceback.print_exc()  # 打印异常的详细信息和调用栈

    return llm_gen_rules_list

def calculate_confidence(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge, is_has_confidence=False, is_relax_time=False):
    llm_gen_rules_list = []
    for input_filepath in glob.glob(os.path.join(rule_path, "rules.txt")):
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

                            rl.create_rule_for_merge(walk, confidence, rule_without_confidence, rules_var_dict,
                                                                                          is_merge, is_relax_time=is_relax_time)

                        except Exception as e:
                            print(f"Error processing rule: {rule}")
                            print(e)
                    else:
                        try:
                            confidence = 0
                            temp_rule = rule.split('&')
                            rule_without_confidence = '&'.join(temp_rule)
                            rule_without_confidence = rule_without_confidence.strip()
                            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
                            rl.create_rule_for_merge(walk, confidence, rule_without_confidence, rules_var_dict,
                                                     is_merge, is_relax_time=is_relax_time)
                            llm_gen_rules_list.append(rule_without_confidence)
                        except Exception as e:
                            print(f"Error processing rule: {rule}")
                            print(e)



                except Exception as e:
                    print(f"Error processing rule: {rule}")
                    traceback.print_exc()  # 打印异常的详细信息和调用栈

    return llm_gen_rules_list


def main(args):
    is_merge = args.is_merge
    dataset_dir = "./datasets/" + args.dataset + "/"
    data = Grapher(dataset_dir)
    if args.bgkg == 'train_valid':
        temporal_walk = Temporal_Walk(np.array(data.train_idx.tolist() + data.valid_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)
    elif args.bgkg == 'all':
        temporal_walk = Temporal_Walk(
            np.array(data.train_idx.tolist() + data.valid_idx.tolist() + data.test_idx.tolist()), data.inv_relation_id,
            args.transition_distr)
    elif args.bgkg == 'test':
        temporal_walk = Temporal_Walk(np.array(data.test_idx.tolist()), data.inv_relation_id, args.transition_distr)
    else:
        temporal_walk = Temporal_Walk(np.array(data.train_idx.tolist()), data.inv_relation_id, args.transition_distr)

    rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, args.dataset)
    if args.is_iteration is False:
        rule_path = os.path.join('clean_rules', args.dataset, args.p, args.model_name)
    else:
        rule_path = os.path.join('gen_rules_iteration', args.dataset, 'final_summary')

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
                rl.num_original  += 1
    else:
        llm_gen_rules_list = calculate_confidence(rule_path, data.relation2id, data.inv_relation_id, rl,
                                                  relation_regex, rules_var_dict, is_merge, is_relax_time=args.is_relax_time)

    save_rules(args, rules_var_dict, rl, llm_gen_rules_list, is_merge)


def save_rules(args, rules_var_dict, rl, llm_gen_rules_list, is_merge):
    # 创建文件夹路径
    dir_path = f"./ranked_rules/{args.dataset}/"
    os.makedirs(dir_path, exist_ok=True)

    # 根据条件设置文件名
    if args.is_only_with_original_rules:
        confidence_file_name = 'original_confidence.json'
    else:
        if is_merge:
            # 更新规则字典
            original_rules_set = set(rules_var_dict)
            llm_gen_rules_set = set(llm_gen_rules_list)
            for rule_chain in original_rules_set - llm_gen_rules_set:
                rule = rules_var_dict[rule_chain]
                rl.update_rules_dict(rule)

            confidence_file_name = 'merge_confidence.json'
        else:
            confidence_file_name = 'confidence.json'

    rules_statistics(rl.rules_dict)

    # 保存到json文件
    confidence_file_path = os.path.join(dir_path, confidence_file_name)
    save_json_data(rl.rules_dict, confidence_file_path)


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
    parser.add_argument("--dataset", default="family")
    parser.add_argument('--model_name', default='none', help='model name',
                        choices=['none', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    parser.add_argument("-p", default="gpt-3.5-turbo-top-0-f-5-l-3")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all",
                        help="evaluate on all or only test set")
    parser.add_argument("--input_path", default="clean_rules", type=str, help="input folder")
    parser.add_argument("--output_path", default="ranked_rules", type=str, help="path to output file")
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument("--transition_distr", default="exp", type=str)
    parser.add_argument("--is_merge", default='yes', type=str_to_bool)
    parser.add_argument("--is_only_with_original_rules", default='no', type=str_to_bool)
    parser.add_argument("--is_iteration", default='yes', type=str_to_bool)
    parser.add_argument("--bgkg", default="train", type=str,
                        choices=['train', 'train_valid', 'all', 'test'])
    parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
    args = parser.parse_args()

    main(args)
