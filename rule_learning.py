import os
import json
import itertools
import numpy as np
from collections import Counter

import copy
import re
import traceback

from utils import save_json_data


class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            id2relation (dict): mapping of index to relation
            inv_relation_id (dict): mapping of relation to inverse relation
            dataset (str): dataset name

        Returns:
            None
        """

        self.edges = edges
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.num_individual = 0
        self.num_shared = 0
        self.num_original = 0

        self.found_rules = []
        self.rule2confidence_dict = {}
        self.original_found_rules = []
        self.rules_dict = dict()
        self.output_dir = "./sampled_path/" + dataset + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_rule(self, walk, confidence=0):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)

    def create_rule_with_relax_time(self, walk, confidence=0):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule,is_relax_time=True)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)

    def create_rule_for_merge(self, walk, confidence=0, rule_without_confidence="", rules_var_dict=None,
                              is_merge=False, is_relax_time=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if is_merge is True:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"],
                    ) = self.estimate_confidence(rule)

                    rule["llm_confidence"] = confidence

                    if rule["conf"] or confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)


            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = confidence
                temp_var = {}
                temp_var['head_rel'] = rule_var['head_rel']
                temp_var['body_rels'] = rule_var['body_rels']
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"],
                ) = self.estimate_confidence(rule, is_relax_time=is_relax_time)

                # if rule["body_supp"] == 0:
                #     rule["body_supp"] = 2

                rule["llm_confidence"] = confidence

                if rule["conf"] or confidence:
                    self.update_rules_dict(rule)

    def create_rule_for_merge_for_iteration(self, walk, llm_confidence=0, rule_without_confidence="",
                                            rules_var_dict=None,
                                            is_merge=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        rule_with_confidence = ""

        if is_merge is True:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"],
                    ) = self.estimate_confidence(rule)

                    tuple_key = str(rule)
                    self.rule2confidence_dict[tuple_key] = rule["conf"]
                    rule_with_confidence = rule_without_confidence + '&' + str(rule["conf"])

                    rule["llm_confidence"] = llm_confidence

                    if rule["conf"] or llm_confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)
                else:
                    tuple_key = tuple(rule)
                    confidence = self.rule2confidence_dict[tuple_key]
                    rule_with_confidence = rule_without_confidence + '&' + confidence


            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = llm_confidence
                temp_var = {}
                temp_var['head_rel'] = rule_var['head_rel']
                temp_var['body_rels'] = rule_var['body_rels']
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                tuple_key = str(rule)
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"],
                ) = self.estimate_confidence(rule)

                self.rule2confidence_dict[tuple_key] = rule["conf"]
                rule_with_confidence = rule_without_confidence + '&' + str(rule["conf"])

                if rule["body_supp"] == 0:
                    rule["body_supp"] = 2

                rule["llm_confidence"] = llm_confidence

                if rule["conf"] or llm_confidence:
                    self.update_rules_dict(rule)
            else:
                tuple_key = str(rule)
                confidence = self.rule2confidence_dict[tuple_key]
                rule_with_confidence = rule_without_confidence + '&' + str(confidence)

        return rule_with_confidence

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        var_constraints = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)

    def estimate_confidence(self, rule, num_samples=2000, is_relax_time=False):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.

        Parameters:
            rule (dict): rule
                         {"head_rel": int, "body_rels": list, "var_constraints": list}
            num_samples (int): number of samples

        Returns:
            confidence (float): confidence of the rule, rule_support/body_support
            rule_support (int): rule support
            body_support (int): body support
        """

        if any(body_rel not in self.edges for body_rel in rule["body_rels"]):
            return 0, 0, 0

        if rule['head_rel'] not in self.edges:
            return 0, 0, 0

        all_bodies = []
        for _ in range(num_samples):

            if is_relax_time is False:
                sample_successful, body_ents_tss = self.sample_body(
                    rule["body_rels"], rule["var_constraints"]
                )
                if sample_successful:
                    all_bodies.append(body_ents_tss)
            else:
                sample_successful, body_ents_tss = self.sample_body_with_relax_time(
                    rule["body_rels"], rule["var_constraints"]
                )
                if sample_successful:
                    all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support

    def sample_body(self, body_rels, var_constraints):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss


    def sample_body_with_relax_time(self, body_rels, var_constraints):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            # mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            mask = (next_edges[:, 0] == cur_node)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support. Check for each body if there is a timestamp
        (larger than the timestamps in the rule body) for which the rule head holds.

        Parameters:
            unique_bodies (list): bodies from self.sample_body
            head_rel (int): head relation

        Returns:
            rule_support (int): rule support
        """

        rule_support = 0
        try:
            head_rel_edges = self.edges[head_rel]
        except Exception as e:
            print(head_rel)
        for body in unique_bodies:
            mask = (
                    (head_rel_edges[:, 0] == body[0])
                    * (head_rel_edges[:, 2] == body[-1])
                    * (head_rel_edges[:, 3] > body[-2])
            )

            if True in mask:
                rule_support += 1

        return rule_support

    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """

        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    def save_rules_verbalized(
            self, dt, rule_lengths, num_walks, transition_distr, seed, rel2idx, relation_regex
    ):
        """
        Save all rules in a human-readable format.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        output_original_dir = self.output_dir + 'original/'
        # 创建目标文件夹
        if not os.path.exists(output_original_dir):
            os.makedirs(output_original_dir)

        rules_str = ""
        rules_var = {}
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                single_rule = verbalize_rule(rule, self.id2relation) + "\n"
                part = re.split(r'\s+', single_rule.strip())
                rule_with_confidence = f"{part[-1]}"
                rules_var[rule_with_confidence] = rule
                rules_str += single_rule

        save_json_data(rules_var, output_original_dir + "rules_var.json")

        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.txt".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            fout.write(rules_str)

        original_rule_txt = self.output_dir + filename

        remove_filename = "remove_{0}_r{1}_n{2}_{3}_s{4}_rules.txt".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        remove_filename = remove_filename.replace(" ", "")

        rule_id_content = []

        with open(self.output_dir + filename, 'r') as input_file, open(self.output_dir + remove_filename, 'w',
                                                                       encoding="utf-8") as output_file:
            for line in input_file:
                # 分割每一行并移除前三列
                columns = line.split()
                new_line = ' '.join(columns[3:])
                new_line_for_rule_id = ' '.join(columns[3:]) + '&' + columns[0] + '\n'
                rule_id_content.append(new_line_for_rule_id)
                # 将修改后的行写入新文件
                output_file.write(new_line + '\n')

        output_file_path = self.output_dir + 'closed_rel_paths.jsonl'
        with open(self.output_dir + remove_filename, 'r') as file:
            lines = file.readlines()
            converted_rules = parse_rules_for_path(lines, list(rel2idx.keys()), relation_regex)
        with open(output_file_path, 'w') as file:
            for head, paths in converted_rules.items():
                json.dump({"head": head, "paths": paths}, file)
                file.write('\n')

        print(f'Rules have been converted and saved to {output_file_path}')

        # Read the rules from a file
        input_file_path = self.output_dir + remove_filename
        with open(input_file_path, 'r') as file:
            rules_content = file.readlines()
            rules_name_dict = parse_rules_for_name(rules_content, list(rel2idx.keys()), relation_regex)

        # Write the JSON to a file
        output_file_path = self.output_dir + 'rules_name.json'
        with open(output_file_path, 'w') as file:
            json.dump(rules_name_dict, file, indent=4)

        print(f'Rules have been converted and saved to {output_file_path}')

        rules_id_dict = parse_rules_for_id(rule_id_content, rel2idx, relation_regex)

        # Write the JSON to a file
        output_file_path = self.output_dir + 'rules_id.json'
        with open(output_file_path, 'w') as file:
            json.dump(rules_id_dict, file, indent=4)

        print(f'Rules have been converted and saved to {output_file_path}')

        save_rule_name_with_confidence(original_rule_txt, relation_regex,
                                       self.output_dir + 'relation_name_with_confidence.json', list(rel2idx.keys()))


def parse_rules_for_path(lines, relations, relation_regex):
    converted_rules = {}
    for line in lines:
        rule = line.strip()
        if not rule:
            continue
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')

        head = ""
        body_list = []
        for idx, regrex_item in enumerate(regrex_list):
            match = re.search(relation_regex, regrex_item)
            if match:
                rel_name = match.group(1).strip()
                if rel_name not in relations:
                    raise ValueError(f"Not exist relation:{rel_name}")
                if idx == 0:
                    head = rel_name
                    paths = converted_rules.setdefault(head, [])
                else:
                    body_list.append(rel_name)

        path = '|'.join(body_list)
        paths.append(path)

    return converted_rules


def parse_rules_for_name(lines, relations, relation_regex):
    rules_dict = {}
    for rule in lines:
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in relations:
                raise ValueError(f"Not exist relation:{head}")
        else:
            continue

        if head not in rules_dict:
            rules_dict[head] = []
        rules_dict[head].append(rule)

    return rules_dict


def save_rule_name_with_confidence(file_path, relation_regex, out_file_path, relations):
    rules_dict = {}
    with open(file_path, 'r') as fin:
        rules = fin.readlines()
        for rule in rules:
            # Split the string by spaces to get the columns
            columns = rule.split()

            # Extract the first and fourth columns
            first_column = columns[0]
            fourth_column = ''.join(columns[3:])
            output = f"{fourth_column}&{first_column}"

            # print(fourth_column)

            regrex_list = fourth_column.split('<-')
            match = re.search(relation_regex, regrex_list[0])
            if match:
                head = match[1].strip()
                if head not in relations:
                    raise ValueError(f"Not exist relation:{head}")
            else:
                continue

            if head not in rules_dict:
                rules_dict[head] = []
            rules_dict[head].append(output)
    save_json_data(rules_dict, out_file_path)


def parse_rules_for_id(rules, rel2idx, relation_regex):
    rules_dict = {}
    for rule in rules:
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in rel2idx:
                raise ValueError(f"Relation '{head}' not found in rel2idx")
        else:
            continue

        rule_id = rule2id(rule.rsplit('&', 1)[0], rel2idx, relation_regex)
        rule_id = rule_id + '&' + rule.rsplit('&', 1)[1].strip()
        rules_dict.setdefault(head, []).append(rule_id)
    return rules_dict


def rule2id(rule, relation2id, relation_regex):
    temp_rule = copy.deepcopy(rule)
    temp_rule = re.sub(r'\s*<-\s*', '&', temp_rule)
    temp_rule = temp_rule.split('&')
    rule2id_str = ""

    try:
        for idx, _ in enumerate(temp_rule):
            match = re.search(relation_regex, temp_rule[idx])
            rel_name = match[1].strip()
            subject = match[2].strip()
            object = match[3].strip()
            timestamp = match[4].strip()
            rel_id = relation2id[rel_name]
            full_id = f"{rel_id}({subject},{object},{timestamp})"
            if idx == 0:
                full_id = f"{full_id}<-"
            else:
                full_id = f"{full_id}&"

            rule2id_str += f"{full_id}"
    except KeyError as keyerror:
        # 捕获异常并打印调用栈信息
        traceback.print_exc()
        raise ValueError(f"KeyError: {keyerror}")

    except Exception as e:
        raise ValueError(f"An error occurred: {rule}")

    return rule2id_str[:-1]


def verbalize_rule(rule, id2relation):
    """
    Verbalize the rule to be in a human-readable format.

    Parameters:
        rule (dict): rule from Rule_Learner.create_rule
        id2relation (dict): mapping of index to relation

    Returns:
        rule_str (str): human-readable rule
    """

    if rule["var_constraints"]:
        var_constraints = rule["var_constraints"]
        constraints = [x for sublist in var_constraints for x in sublist]
        for i in range(len(rule["body_rels"]) + 1):
            if i not in constraints:
                var_constraints.append([i])
        var_constraints = sorted(var_constraints)
    else:
        var_constraints = [[x] for x in range(len(rule["body_rels"]) + 1)]

    rule_str = "{0:8.6f}  {1:4}  {2:4}  {3}(X0,X{4},T{5})<-"
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3})&".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    return rule_str[:-1]


def rules_statistics(rules_dict):
    """
    Show statistics of the rules.

    Parameters:
        rules_dict (dict): rules

    Returns:
        None
    """

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"]) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))
