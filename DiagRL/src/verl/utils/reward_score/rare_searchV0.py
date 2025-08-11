import re
import sys
import string
from typing import Union, List
from collections import Counter
import math
import random
import Levenshtein

import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # check match/refer pairs
    current_pos = 0
    while True:
        match_pos = text.find('<match>', current_pos)
        if match_pos == -1:
            break
            
        refer_pos = text.find('<refer>', match_pos)
        match_end_pos = text.find('</match>', match_pos)
        refer_end_pos = text.find('</refer>', refer_pos)
        
        if -1 in (refer_pos, match_end_pos, refer_end_pos):
            return False, "match/refer tags are incomplete"
            
        if not (match_pos < match_end_pos < refer_pos < refer_end_pos):
            return False, "match/refer tags are nested in the wrong order"
            
        current_pos = refer_end_pos
    
    # check search/result pairs
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"
            
        current_pos = result_end_pos
    
    # check overall order: match/refer should come before search/result
    last_refer_pos = text.rfind('</refer>')
    first_search_pos = text.find('<search>')
    
    if first_search_pos != -1 and last_refer_pos != -1:
        if first_search_pos < last_refer_pos:
            return False, "search/result must come after match/refer"
    
    # check if \textbf{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\textbf{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\textbf{} format"
    
    return True, "format is correct"

def extract_diseases_from_refer(text: str) -> tuple:
    """从refer标签中提取所有疾病名称
    
    处理形如：
    <refer>Alexander's disease, Canavan disease, Krabbe's disease</refer>
    的文本，并正确处理疾病名称中包含逗号和撇号的情况
    """
    try:
        # 首先提取refer标签内容
        pattern = r"<refer>(.*?)</refer>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, "No refer tag found"
        
        refer_content = match.group(1).strip()
        if not refer_content:
            return None, "Empty refer content"
            
        # 使用更智能的分割方法处理疾病名称
        diseases = []
        current_disease = ""
        in_quote = False
        
        for char in refer_content + ',':  # 添加末尾逗号以处理最后一个疾病
            if char == "'" and (not current_disease or current_disease[-1] != '\\'):
                in_quote = not in_quote
                current_disease += char
            elif char == ',' and not in_quote:
                # 当遇到逗号且不在引号内时，添加当前疾病到列表
                if current_disease:
                    diseases.append(current_disease.strip())
                current_disease = ""
            else:
                current_disease += char
        
        # 清理疾病名称列表
        diseases = [disease.strip() for disease in diseases if disease.strip()]
        
        if not diseases:
            return None, "No diseases found"
            
        return diseases, "successful extraction"
        
    except Exception as e:
        return None, f"Extract from reference failed: {str(e)}"
    
def extract_diseases_from_search(text: str) -> tuple:
    """从search标签中提取所有疾病名称
    
    处理形如：
    <search>Alexander's disease, Canavan disease</search>
    的文本，并将其转换为疾病名称列表
    
    Returns:
        tuple: (List[str] | None, str) - (疾病列表或None, 处理信息)
    """
    try:
        # 提取search标签内容
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, "No search tag found"
        
        search_content = match.group(1).strip()
        if not search_content:
            return None, "Empty search content"
        
        # 简单地按逗号分割，然后清理每个部分
        diseases = [disease.strip() for disease in search_content.split(',')]
        # 移除空字符串
        diseases = [d for d in diseases if d]
        
        if not diseases:
            return None, "No diseases found"
            
        return diseases, "successful extraction"
        
    except Exception as e:
        return None, f"Extract from search failed: {str(e)}"

def extract_diseases_from_textbf(text: str) -> tuple:
    """从答案中提取所有\textbf{}内的疾病名称"""
    try:
        # 首先提取answer标签内容
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if not answer_match:
            return None, "No answer tag found"
        
        answer_content = answer_match.group(1)
        
        # 提取所有\textbf{}中的内容
        textbf_pattern = r"\\textbf{([^}]*)}"
        textbf_matches = re.findall(textbf_pattern, answer_content)
        
        if not textbf_matches:
            return None, "No textbf found"
        
        return textbf_matches, "successful extraction"
    except Exception as e:
        return None, f"Extract from answer box failed: {str(e)}"

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_disease_similarity_score(predict_list, ground_truth):
    """
    计算ground_truth与predict_list疾病的相似度分数
    
    参数:
    predict_list -- 参考疾病列表
    ground_truth -- 真实疾病列表
    
    返回:
    加权后的总分数
    """
    if not ground_truth:
        return 0.0
    
    # 规范化所有疾病名称
    normalized_pred = [normalize_answer(disease) for disease in predict_list]
    normalized_gt = [normalize_answer(disease) for disease in ground_truth]
    
    # 移除空字符串
    normalized_pred = [d for d in normalized_pred if d]
    normalized_gt = [d for d in normalized_gt if d]
    
    if not normalized_gt or not normalized_pred:
        return 0.0
    
    # 为每个ground truth疾病计算最佳匹配分数
    best_scores = []
    for gt_disease in normalized_gt:
        if not gt_disease:
            continue
            
        max_score = 0.0
        for pred_disease in normalized_pred:
            if not pred_disease:
                continue
                
            # 计算Levenshtein距离
            distance = Levenshtein.distance(gt_disease, pred_disease)
            
            # 计算相似度分数
            max_len = max(len(gt_disease), len(pred_disease))
            if max_len == 0:
                similarity = 0.0
            else:
                # 转换为相似度 (1 - 归一化距离)
                similarity = 1.0 - (distance / max_len)
            
            max_score = max(max_score, similarity)
        
        best_scores.append(max_score)
    
    # 如果没有有效的分数，则返回0
    if not best_scores:
        return 0.0
    
    # 计算立方根加权平均分数
    weighted_scores = [score ** (3) for score in best_scores]
    final_score = (sum(weighted_scores) / len(weighted_scores))**(1/3)
    
    return final_score

def get_tokens(text: str) -> list:
    """
    将文本分割成更细粒度的token
    
    Args:
        text: 输入文本
        
    Returns:
        list: token列表
    """
    # 使用NLTK的word_tokenize进行分词
    tokens = word_tokenize(text)
    # 移除空token和纯标点符号的token
    tokens = [token.lower() for token in tokens if token.strip() and not all(c in string.punctuation for c in token)]
    return tokens

def calculate_token_similarity_score(predict_list, ground_truth):
    """
    计算ground_truth与predict_list疾病的相似度分数
    
    参数:
    predict_list -- 参考疾病列表
    ground_truth -- 真实疾病列表
    
    返回:
    加权后的总分数
    """
    if not ground_truth:
        return 0.0
    
    # 规范化所有疾病名称
    normalized_pred = [normalize_answer(disease) for disease in predict_list]
    normalized_gt = [normalize_answer(disease) for disease in ground_truth]
    
    # 移除空字符串
    normalized_pred = [d for d in normalized_pred if d]
    normalized_gt = [d for d in normalized_gt if d]
    
    if not normalized_gt or not normalized_pred:
        return 0.0
    
    # 将predict_list中的所有疾病名称分割成token集合
    pred_tokens = set()
    for pred in normalized_pred:
        pred_tokens.update(get_tokens(pred))
    
    # 统计ground truth中的token匹配情况
    total_tokens = 0
    matched_tokens = 0
    
    for gt in normalized_gt:
        gt_tokens = get_tokens(gt)
        total_tokens += len(gt_tokens)
        for token in gt_tokens:
            if token in pred_tokens:
                matched_tokens += 1
    
    # 如果没有token，返回0
    if total_tokens == 0:
        return 0.0
    
    # 计算匹配率，并进行立方根加权
    score = (matched_tokens / total_tokens) ** (1/3)
    
    return score

def compute_score(tokenizer, solution_str, ground_truth) -> tuple:
    # return random.random()
    # handling both the base model and the instruction-tuned model
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")
    
    response = solution_str_split[1]
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0.0, f'bad format: {reason}', None, None, None

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0.0, f'over length', None, None, None

    reference_list, reason = extract_diseases_from_refer(response)
    if reference_list is None:
        return 0.0, f'bad format: {reason}', None, None, None
    
    search_list, reason = extract_diseases_from_search(response)
    if search_list is None:
        return 0.0, f'bad format: {reason}', None, None, None
    
    answer_list, reason = extract_diseases_from_textbf(response)
    if answer_list is None:
        return 0.0, f'bad format: {reason}', None, None, None
    
    reference_list = [normalize_answer(disease) for disease in reference_list]
    search_list = [normalize_answer(disease) for disease in search_list]
    answer_list = [normalize_answer(disease) for disease in answer_list]
    ground_truth = [normalize_answer(disease) for disease in ground_truth]

    refer_score = calculate_token_similarity_score(reference_list, ground_truth)
    search_score = calculate_token_similarity_score(search_list, ground_truth)
    if len(search_list) > 10:
        search_score = 0.0
    answer_score = calculate_token_similarity_score(answer_list, ground_truth)

    final_score = refer_score * 0.3 + search_score * 0.3 + answer_score * 0.4
    
    return final_score, f'matched diseases, score: {final_score:.3f}', refer_score, search_score, answer_score