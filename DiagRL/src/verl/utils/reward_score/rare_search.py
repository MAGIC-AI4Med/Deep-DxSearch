import re
import sys
import string
from typing import Union, List, Tuple
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
    """验证格式，包括新的迭代结构要求"""
    # 检查基本的answer标签
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # 提取所有match/refer/think的组合
    match_refer_think_pattern = r'<match>(.*?)</match>\s*<refer>(.*?)</refer>\s*<think>(.*?)</think>'
    matches = re.findall(match_refer_think_pattern, text, re.DOTALL)
    
    if len(matches) == 0:
        return False, "No valid match/refer/think combinations found"
    
    if len(matches) > 3:
        return False, f"Too many match/refer/think combinations: {len(matches)} (max 3)"
    
    # 检查search/result对
    search_count = text.count('<search>')
    result_count = text.count('<result>')
    
    if search_count != result_count:
        return False, f"search/result tags not paired: {search_count} search vs {result_count} result"
    
    # 检查answer中是否包含textbf格式
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\textbf{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\textbf{} format"
    
    return True, "format is correct"

def enhanced_normalize_answer(s):
    """增强的normalize函数，增加去除末尾复数s的功能"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_trailing_s(text):
        """去除单词末尾的复数s"""
        words = text.split()
        result_words = []
        for word in words:
            if len(word) > 1 and word.endswith('s'):
                # 简单的复数处理：如果以s结尾且长度>1，去掉s
                result_words.append(word[:-1])
            else:
                result_words.append(word)
        return " ".join(result_words)

    return remove_trailing_s(white_space_fix(remove_articles(remove_punc(lower(s)))))

def extract_all_match_refer_think(text: str) -> tuple:
    """提取所有的match/refer/think组合"""
    try:
        pattern = r'<match>(.*?)</match>\s*<refer>(.*?)</refer>\s*<think>(.*?)</think>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            return None, "No match/refer/think combinations found"
        
        results = []
        for i, (match_content, refer_content, think_content) in enumerate(matches):
            # 处理match内容（phenotypes）
            phenotypes = [p.strip() for p in match_content.strip().split(',') if p.strip()]
            
            # 处理refer内容（diseases）
            diseases = []
            current_disease = ""
            in_quote = False
            
            for char in refer_content.strip() + ',':
                if char == "'" and (not current_disease or current_disease[-1] != '\\'):
                    in_quote = not in_quote
                    current_disease += char
                elif char == ',' and not in_quote:
                    if current_disease:
                        diseases.append(current_disease.strip())
                    current_disease = ""
                else:
                    current_disease += char
            
            diseases = [disease.strip() for disease in diseases if disease.strip()]
            
            results.append({
                'iteration': i + 1,
                'phenotypes': phenotypes,
                'diseases': diseases,
                'think': think_content.strip()
            })
        
        return results, "successful extraction"
        
    except Exception as e:
        return None, f"Extract match/refer/think failed: {str(e)}"

def extract_diseases_from_search(text: str) -> tuple:
    """从search标签中提取所有疾病名称"""
    try:
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, "No search tag found"
        
        search_content = match.group(1).strip()
        if not search_content:
            return None, "Empty search content"
        
        diseases = [disease.strip() for disease in search_content.split(',')]
        diseases = [d for d in diseases if d]
        
        if not diseases:
            return None, "No diseases found"
            
        return diseases, "successful extraction"
        
    except Exception as e:
        return None, f"Extract from search failed: {str(e)}"

def extract_diseases_from_textbf(text: str) -> tuple:
    """从答案中提取所有\textbf{}内的疾病名称"""
    try:
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if not answer_match:
            return None, "No answer tag found"
        
        answer_content = answer_match.group(1)
        textbf_pattern = r"\\textbf{([^}]*)}"
        textbf_matches = re.findall(textbf_pattern, answer_content)
        
        if not textbf_matches:
            return None, "No textbf found"
        
        return textbf_matches, "successful extraction"
    except Exception as e:
        return None, f"Extract from answer box failed: {str(e)}"

def check_disease_match(refer_diseases: List[str], ground_truth: List[str]) -> bool:
    """检查refer中的疾病是否与ground truth匹配"""
    normalized_refer = [enhanced_normalize_answer(disease) for disease in refer_diseases]
    normalized_gt = [enhanced_normalize_answer(disease) for disease in ground_truth]
    
    for refer_disease in normalized_refer:
        for gt_disease in normalized_gt:
            if refer_disease == gt_disease:
                return True
    return False

def count_phenotype_differences(phenotypes1: List[str], phenotypes2: List[str]) -> int:
    """计算两组phenotypes之间的差异数量"""
    set1 = set([enhanced_normalize_answer(p) for p in phenotypes1])
    set2 = set([enhanced_normalize_answer(p) for p in phenotypes2])
    
    # 计算对称差集的大小（增删替换的总数）
    return len(set1.symmetric_difference(set2))

def calculate_refer_score(match_refer_results: List[dict], ground_truth: List[str]) -> tuple[float, str]:
    """计算refer阶段的分数"""
    if len(match_refer_results) > 3:
        return 0.0, "More than 3 match iterations, score set to 0"
    
    # 检查是否有任何一次refer与ground truth匹配
    has_match = False
    match_details = []
    
    for result in match_refer_results:
        is_match = check_disease_match(result['diseases'], ground_truth)
        match_details.append(f"Iteration {result['iteration']}: {'Match' if is_match else 'No match'}")
        if is_match:
            has_match = True
    
    if has_match:
        # 基础分数1分
        score = 1.0
        deduction_reasons = []
        
        # 检查最后一次refer是否有匹配
        last_result = match_refer_results[-1]
        last_has_match = check_disease_match(last_result['diseases'], ground_truth)
        if not last_has_match and len(match_refer_results) > 1:
            score -= 0.5
            deduction_reasons.append("Last refer has no match (-0.5)")
        
        # 检查多次搜索时phenotype变化是否足够
        if len(match_refer_results) > 1:
            insufficient_changes = []
            for i in range(len(match_refer_results) - 1):
                diff_count = count_phenotype_differences(
                    match_refer_results[i]['phenotypes'],
                    match_refer_results[i + 1]['phenotypes']
                )
                if diff_count < 2:
                    insufficient_changes.append(f"Iteration {i+1} to {i+2}: only {diff_count} changes")
            
            if insufficient_changes:
                score -= 0.5
                deduction_reasons.append(f"Insufficient phenotype changes (-0.5): {'; '.join(insufficient_changes)}")
        
        reason = f"Base score 1.0 (has matches). {match_details}. " + "; ".join(deduction_reasons) if deduction_reasons else f"Base score 1.0 (has matches). {match_details}"
        return max(0.0, score), reason
    
    else:
        # 没有匹配的情况
        score = 0.0
        bonus_reasons = []
        
        # 检查是否可以加分
        if len(match_refer_results) >= 2:
            sufficient_changes = True
            change_details = []
            for i in range(len(match_refer_results) - 1):
                diff_count = count_phenotype_differences(
                    match_refer_results[i]['phenotypes'],
                    match_refer_results[i + 1]['phenotypes']
                )
                change_details.append(f"Iteration {i+1} to {i+2}: {diff_count} changes")
                if diff_count < 2:
                    sufficient_changes = False
            
            if sufficient_changes:
                if len(match_refer_results) == 2:
                    score += 0.2
                    bonus_reasons.append("2 iterations with sufficient changes (+0.1)")
                elif len(match_refer_results) == 3:
                    score += 0.4
                    bonus_reasons.append("3 iterations with sufficient changes (+0.2)")
                
                change_summary = "; ".join(change_details)
                bonus_reasons.append(f"Change details: {change_summary}")
        
        reason = f"No matches found. {match_details}. " + "; ".join(bonus_reasons) if bonus_reasons else f"No matches found. {match_details}"
        return score, reason

def get_tokens(text: str) -> list:
    """将文本分割成更细粒度的token"""
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.strip() and not all(c in string.punctuation for c in token)]
    return tokens

def calculate_token_similarity_score(predict_list, ground_truth):
    """计算token相似度分数"""
    if not ground_truth:
        return 0.0
    
    normalized_pred = [enhanced_normalize_answer(disease) for disease in predict_list]
    normalized_gt = [enhanced_normalize_answer(disease) for disease in ground_truth]
    
    normalized_pred = [d for d in normalized_pred if d]
    normalized_gt = [d for d in normalized_gt if d]
    
    if not normalized_gt or not normalized_pred:
        return 0.0
    
    pred_tokens = set()
    for pred in normalized_pred:
        pred_tokens.update(get_tokens(pred))
    
    total_tokens = 0
    matched_tokens = 0
    
    for gt in normalized_gt:
        gt_tokens = get_tokens(gt)
        total_tokens += len(gt_tokens)
        for token in gt_tokens:
            if token in pred_tokens:
                matched_tokens += 1
    
    if total_tokens == 0:
        return 0.0
    
    score = (matched_tokens / total_tokens) ** (1/3)
    return score

def compute_score(tokenizer, solution_str, ground_truth) -> tuple:
    """主评估函数"""
    # 处理模型输出格式
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")
    
    response = solution_str_split[1]
    
    # 验证格式
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0.0, f'bad format: {reason}', None, None, None

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0.0, f'over length', None, None, None

    # 提取match/refer/think结果
    match_refer_results, reason = extract_all_match_refer_think(response)
    if match_refer_results is None:
        return 0.0, f'bad format: {reason}', None, None, None
    
    # 计算refer分数
    refer_score, refer_reason = calculate_refer_score(match_refer_results, ground_truth)
    
    # 提取search结果
    search_list, reason = extract_diseases_from_search(response)
    if search_list is None:
        search_score = 0.0
        search_reason = f'no search results: {reason}'
    else:
        search_list = [enhanced_normalize_answer(disease) for disease in search_list]
        ground_truth = [enhanced_normalize_answer(disease) for disease in ground_truth]
        
        if len(search_list) > 10:
            search_score = 0.0
            search_reason = f'too many search diseases: {len(search_list)} > 10'
        else:
            search_score = calculate_token_similarity_score(search_list, ground_truth)
            search_reason = f'search score: {search_score:.3f} based on token similarity'
    
    # 提取answer结果
    answer_list, reason = extract_diseases_from_textbf(response)
    if answer_list is None:
        answer_score = 0.0
        answer_reason = f'no answer results: {reason}'
    else:
        answer_list = [enhanced_normalize_answer(disease) for disease in answer_list]
        ground_truth = [enhanced_normalize_answer(disease) for disease in ground_truth]
        answer_score = calculate_token_similarity_score(answer_list, ground_truth)
        answer_reason = f'answer score: {answer_score:.3f} based on token similarity'
    
    # 计算最终分数 (refer: 0.8, search: 0.05, answer: 0.15)
    final_score = refer_score * 0.8 + search_score * 0.05 + answer_score * 0.15
    
    detailed_reason = f'Final score: {final_score:.3f} | Refer: {refer_score:.3f} ({refer_reason}) | Search: {search_score:.3f} ({search_reason}) | Answer: {answer_score:.3f} ({answer_reason})'
    
    return final_score, detailed_reason, refer_score, search_score, answer_score