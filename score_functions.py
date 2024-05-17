import numpy as np


def score1(rule, c=0, confidence_type='TLogic', weight=0.0):
    """
    Calculate candidate score depending on the rule's confidence.

    Parameters:
        rule (dict): rule from rules_dict
        c (int): constant for smoothing

    Returns:
        score (float): candidate score
    """

    if confidence_type == 'TLogic':
        # score = rule["rule_supp"] / (rule["body_supp"] + c)
        score = rule['conf']
    elif confidence_type == 'LLM':
        score = rule['llm_confidence']
    elif confidence_type == 'Or':
        score = max(rule['conf'], rule['llm_confidence'])
    else:
        score = weight * rule['conf'] + (1 - weight) * rule['llm_confidence']

    return score


def score2(cands_walks, test_query_ts, lmbda):
    """
    Calculate candidate score depending on the time difference.

    Parameters:
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution

    Returns:
        score (float): candidate score
    """

    max_cands_ts = max(cands_walks["timestamp_0"])
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)
    )  # Score depending on time difference

    return score

def score4(cands_walks, test_query_ts, lmbda, rule):
    """
    Calculate candidate score depending on the time difference.

    Parameters:
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution

    Returns:
        score (float): candidate score
    """

    max_cands_ts = max(cands_walks[f'timestamp_{len(rule["body_rels"]) - 1}'])
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)
    )  # Score depending on time difference

    return score


def score_12(rule, cands_walks, test_query_ts, corr, lmbda, a, confidence_type, weight, min_conf, coor_weight):
    """
    Combined score function.

    Parameters:
        rule (dict): rule from rules_dict
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution
        a (float): value between 0 and 1

    Returns:
        score (float): candidate score
    """

    score = a * score1(rule, 0, confidence_type, weight) + (1 - a) * score2(cands_walks, test_query_ts,
                                                                            lmbda) + coor_weight * corr

    return score

def score_13(rule, cands_walks, test_query_ts, corr, lmbda, a, confidence_type, weight, min_conf, coor_weight):
    """
    Combined score function.

    Parameters:
        rule (dict): rule from rules_dict
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution
        a (float): value between 0 and 1

    Returns:
        score (float): candidate score
    """

    # score = a * score1(rule, 0, confidence_type, weight) + (1 - a) * score2(cands_walks, test_query_ts,
    #                                                                         lmbda) + coor_weight * corr

    score = a * (score1(rule, 0, confidence_type, weight) * corr) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)

    return score

def score_14(rule, cands_walks, test_query_ts, corr, lmbda, a, confidence_type, weight, min_conf, coor_weight):
    """
    Combined score function.

    Parameters:
        rule (dict): rule from rules_dict
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution
        a (float): value between 0 and 1

    Returns:
        score (float): candidate score
    """

    score = a * score1(rule, 0, confidence_type, weight) + (1 - a) * score4(cands_walks, test_query_ts,
                                                                            lmbda, rule) + coor_weight * corr

    return score
