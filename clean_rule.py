from chat_rule_generator import clear_folder
from data import *
from llms import get_registed_model
from utils import *
import re


def extract_rules(content_list, rule_start_with_regex, replace_regex):
    """ Extract the rules in the content without any explanation and the leading number if it has."""
    rule_pattern = re.compile(rule_start_with_regex)  # The rule always has (X,Y, T) <-
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


# def get_valid_rules(input_filepath, output_filepath, valid_response_filepath):
#     with open(input_filepath, "r") as f:
#         sum_rule_list = [line.strip() for line in f]
#         f.close()
#     valid_prompt = ("Logical rules define the relationship between two entities: X and Y.\n"
#                     "Now please analyse this relation rule path step by step to check whether it is correct. \n"
#                     "If the rules is correct please write (Correct) at the end of your analysis, otherwise please write (Incorrect).\n\n")
#
#     with open(output_filepath, "w") as f1, open(valid_response_filepath, 'w') as f2:
#         for sum_rule in sum_rule_list:
#             message = valid_prompt + sum_rule
#             response = query(message, model_name="gpt-4")
#             print(response)
#             f2.write("Input Rule: " + sum_rule + "\n")
#             f2.write("GPT-4 Response: \n" + response + '\n')
#             f2.write("\n=======================================\n")
#             if "incorrect" not in response.lower():
#                 f1.write(sum_rule + '\n')


def check_sample_times(content_list):
    """
    Determine the sample time, return True if only sample once
    """
    sample_times = 0
    for line in content_list:
        match = re.search(r'Sample \d+ time:', line)
        if match:
            sample_times += 1
    return sample_times == 1


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
    if llm_model is None and not args.force_summarize:  # just return the whole rule_list
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
        temp_rule = re.sub(r'\s*<-\s*', '&', input_rule)
        regrex_list = temp_rule.split('&')
        last_subject = None
        final_object = None
        time_squeque = []
        final_time = None
        is_save = True
        try:
            for idx, regrex in enumerate(regrex_list[:-1]):
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

                    if idx == len(regrex_list[:-1]) - 1:
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
                cleaned_rules.append(input_rule)
                fout_suc.write(input_rule + '\n')
                num_suc = num_suc + 1

        except Exception as e:
            print(f"Processing {input_rule} failed.\n Error: {str(e)}")
            fout_error.write(f"Processing {input_rule} failed.\n Error: {str(e)}\n")
            num_error = num_error + 1
    return cleaned_rules, num_error, num_suc


def clean(args, llm_model):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Dataset(data_root=data_path, inv=True)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    input_folder = os.path.join(args.rule_path, args.dataset, args.p)
    output_folder = os.path.join(args.output_path, args.dataset, args.p, args.model_name)
    output_statistic_folder_dir = os.path.join(output_folder, 'statistics')
    if not os.path.exists(output_statistic_folder_dir):
        os.makedirs(output_statistic_folder_dir)
    else:
        clear_folder(output_statistic_folder_dir)

    output_error_file_path = os.path.join(output_statistic_folder_dir, 'error.txt')
    output_suc_file_path = os.path.join(output_statistic_folder_dir, 'suc.txt')
    with open(output_error_file_path, 'w') as fout_error, open(output_suc_file_path, 'w') as fout_suc:
        num_error, num_suc = clean_processing(all_rels, args, fout_error, input_folder, llm_model, output_folder, fout_suc)
        fout_error.write(f"The number of cleaned rules is {num_error}\n")
        fout_suc.write(f"The number of retain rules is {num_suc}\n")


def clean_processing(all_rels, args, fout_error, input_folder, llm_model, output_folder, fout_suc):
    constant_config = load_json_data('./Config/constant.json')
    rule_start_with_regex = constant_config["rule_start_with_regex"]
    replace_regex = constant_config["replace_regex"]
    relation_regex = constant_config["relation_regex"][args.dataset]
    num_error = 0
    num_suc = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt") and "query" not in filename:
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

            with open(clean_filepath, "w") as f:
                f.write('\n'.join(cleaned_rules))
    return num_error, num_suc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='data directory')
    parser.add_argument("--rule_path", default="gen_rules", type=str, help="path to rule file")
    parser.add_argument("--output_path", default="clean_rules", type=str, help="path to output file")
    parser.add_argument('--dataset', default='family')
    parser.add_argument('--model_name', default='none', help='model name',
                        choices=['none', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    parser.add_argument('-p', default='gpt-3.5-turbo-top-0-f-50-l-10', help='rule prefix')
    parser.add_argument('-k', type=int, default=0, help='Number of summarized rules')
    parser.add_argument('--clean_only', action='store_true', help='Load summarized rules then clean rules only')
    parser.add_argument('--valid_clean', action='store_true', help='gpt-4 validation for rules')
    parser.add_argument('--force_summarize', action='store_true', help='force summarize rules')

    args, _ = parser.parse_known_args()
    # Parse the command-line arguments
    args = parser.parse_args()

    # Initialize the language model (LLM) based on the provided model name
    if args.model_name == 'none':
        llm_model = None
    else:
        # Get the registered model and add model-specific arguments
        LLM = get_registed_model(args.model_name)
        LLM.add_args(parser)
        # Re-parse the arguments to include model-specific arguments
        args = parser.parse_args()
        # Instantiate the model and prepare it for inference
        llm_model = LLM(args)
        print("Prepare pipeline for inference...")
        llm_model.prepare_for_inference()

    clean(args, llm_model)
