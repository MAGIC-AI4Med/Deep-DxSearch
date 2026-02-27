import re
import sys
import string
import json
from typing import Union, List, Tuple, Optional
from collections import Counter
import math
import random
import Levenshtein

import nltk
from nltk.tokenize import word_tokenize

# 只在punkt不存在时才下载
nltk_data_dir = '/mnt/vision_user/zhengqiaoyu/DiagRL/DiagRLGeneralFree/src/match/nltk_data'
nltk.data.path.insert(0, nltk_data_dir)
# 只在punkt不存在时才下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    raise ValueError("no tokenizers/punkt")
    nltk.download('punkt')

def validate_format(text: str) -> tuple[bool, str]:
    """验证格式，包括新的迭代结构要求"""
    # 检查基本的answer标签
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # 提取所有match/refer/think的组合（match现在是可选的）
    match_refer_think_pattern = r'(?:<match>(.*?)</match>\s*)?<refer>(.*?)</refer>\s*<think>(.*?)</think>'
    matches = re.findall(match_refer_think_pattern, text, re.DOTALL)
    
    # 检查match标签的数量（如果存在）
    match_count = text.count('<match>')
    if match_count > 3:
        return False, f"Too many match tags: {match_count} (max 3)"
    
    # 检查lookup/guide对
    lookup_count = text.count('<lookup>')
    guide_count = text.count('<guide>')

    if lookup_count != guide_count:
        return False, f"lookup/guide tags not paired: {lookup_count} lookup vs {guide_count} guide"
    
    # 检查answer中是否包含textbf格式，并限制其出现次数
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]

    # 检查\\textbf{}格式
    textbf_matches = re.findall(r'\\textbf{.*?}', answer_content)
    if not textbf_matches:
        return False, "answer is missing \\textbf{} format"
    if len(textbf_matches) > 5:
        return False, f"\\textbf{{}} appears too many times: {len(textbf_matches)} (max 5)"
    
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

def extract_diseases_from_cases(cases_json_str: str, keep_duplicates: bool = False) -> List[str]:
    """
    从match_cases返回的JSON字符串中提取所有疾病
    
    Args:
        cases_json_str: match_cases方法返回的JSON字符串
        keep_duplicates: 是否保留重复的疾病名称，默认False去重
    
    Returns:
        List[str]: 提取的疾病列表
    """
    try:
        # 解析JSON字符串
        cases_data = json.loads(cases_json_str)
        
        # 提取所有疾病
        all_diseases = []
        for case in cases_data:
            if 'diseases' in case and isinstance(case['diseases'], list):
                all_diseases.extend(case['diseases'])
        
        # 根据参数决定是否去重
        if keep_duplicates:
            return all_diseases
        else:
            # 去重但保持顺序
            unique_diseases = list(dict.fromkeys(all_diseases))
            return unique_diseases
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"提取疾病时发生错误: {e}")
        return []

def extract_all_match_refer_think(text: str) -> tuple:
    """提取所有的match/refer/think组合（match可选，但如果出现match必须有完整模式）"""
    try:
        # 先检查是否有孤立的match标签（出现了match但没有完整模式）
        match_tags = re.findall(r'<match>', text)
        if match_tags:
            # 如果有match标签，检查是否都有对应的完整模式
            complete_pattern = r'<match>(.*?)</match>\s*<refer>(.*?)</refer>\s*<think>(.*?)</think>'
            complete_matches = re.findall(complete_pattern, text, re.DOTALL)
            
            if len(match_tags) != len(complete_matches):
                return None, f"Found {len(match_tags)} match tags but only {len(complete_matches)} complete match/refer/think patterns"
        else:
            return 0.0, "No Match Applied"
        
        # 提取所有模式（包括没有match的refer/think）
        pattern = r'(?:<match>(.*?)</match>\s*)?<refer>(.*?)</refer>\s*<think>(.*?)</think>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            return None, "No refer/think combinations found"
        
        results = []
        for i, (match_content, refer_content, think_content) in enumerate(matches):
            # 处理match内容（phenotypes）- 现在可能为空
            phenotypes = []
            if match_content:
                phenotypes = [p.strip() for p in match_content.strip().split(',') if p.strip()]
            
            # 处理refer内容（现在是cases的JSON字符串）
            diseases = extract_diseases_from_cases(refer_content.strip())
            
            results.append({
                'iteration': i + 1,
                'phenotypes': phenotypes,
                'diseases': diseases,
                'think': think_content.strip(),
                'has_match': bool(match_content),
                'refer_raw': refer_content.strip()
            })
        
        return results, "successful extraction"
        
    except Exception as e:
        return None, f"Extract match/refer/think failed: {str(e)}"

def extract_diseases_from_lookup(text: str) -> tuple:
    """从lookup标签中提取所有疾病名称"""
    try:
        pattern = r"<lookup>(.*?)</lookup>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, "No lookup tag found"
        
        lookup_content = match.group(1).strip()
        if not lookup_content:
            return None, "Empty lookup content"
        
        diseases = [disease.strip() for disease in lookup_content.split(',')]
        diseases = [d for d in diseases if d]
        
        if not diseases:
            return None, "No diseases found"
            
        return diseases, "successful extraction"
        
    except Exception as e:
        return None, f"Extract from lookup failed: {str(e)}"

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

def calculate_refer_score(match_refer_results: List[dict], ground_truth: List[str]) -> tuple[Optional[float], str]:
    """计算refer调整分数（用于调整answer_score，范围[-0.3, 0.5]或None）"""
    # 统计使用match的次数
    match_count = sum(1 for result in match_refer_results if result['has_match'])
    
    # 新增约束：如果match次数小于2次，直接返回None（0分）
    if match_count < 2:
        return None, f"Insufficient match iterations: {match_count} (minimum 2 required), constraint violation"
    
    # 检查match标签使用次数是否超限
    if match_count > 3:
        return None, f"More than 3 match iterations: {match_count}, constraint violation"
    
    # 检查多次match时phenotype变化是否足够（约束检查）
    if match_count > 1:
        insufficient_changes = []
        for i in range(len(match_refer_results) - 1):
            # 只检查都有phenotypes的相邻迭代
            if match_refer_results[i]['has_match'] and match_refer_results[i+1]['has_match']:
                diff_count = count_phenotype_differences(
                    match_refer_results[i]['phenotypes'],
                    match_refer_results[i + 1]['phenotypes']
                )
                if diff_count < 2:
                    insufficient_changes.append(f"Iteration {i+1} to {i+2}: only {diff_count} changes")
        
        # 如果变化不充分，直接违背约束
        if insufficient_changes:
            return None, f"Constraint violation: Insufficient phenotype changes: {'; '.join(insufficient_changes)}"
    
    # 初始调整分数为0
    adjust_score = 0.0
    score_reasons = []
    
    # 检查是否有任何一次refer与ground truth匹配
    has_match = False
    match_details = []
    
    for result in match_refer_results:
        is_match = check_disease_match(result['diseases'], ground_truth)
        match_details.append(f"Iteration {result['iteration']}: {'Hit' if is_match else 'Miss'}")
        if is_match:
            has_match = True
    
    # 如果有命中，加0.5分
    if has_match:
        adjust_score += 0.5
        score_reasons.append("Disease hit (+0.5)")
    else:
        score_reasons.append("No disease hit")
    
    # 使用match的扣分：每次扣0.1分，最多扣0.3分
    if match_count > 0:
        match_deduction = min(match_count * 0.1, 0.3)
        adjust_score -= match_deduction
        score_reasons.append(f"Used match {match_count} times (-{match_deduction:.1f})")
    
    # 组合所有原因
    all_reasons = [f"{match_details}"] + score_reasons
    reason = ". ".join(all_reasons)
    
    return adjust_score, reason

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

def calculate_adjusted_answer_score(original_score: float, refer_adjust: Optional[float]) -> float:
    """计算调整后的answer分数：保底0.2分，最高0.8分，然后根据refer_adjust调整"""
    base_adjusted = 0.2 + 0.6 * original_score
    
    if refer_adjust is None:
        # 如果refer约束违背，answer分数为0
        return 0.0
    
    # 使用refer_adjust调整answer分数
    final_score = base_adjusted + refer_adjust
    
    # 确保分数在合理范围内（但可以为负）
    return final_score

def compute_score(tokenizer, solution_str, ground_truth) -> tuple:
    """主评估函数"""
    # 处理模型输出格式
    # import ipdb
    # ipdb.set_trace()
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    elif "<start_of_turn>model\n" in solution_str:
        solution_str_split = solution_str.split("<start_of_turn>model\n")
    else:
        solution_str_split = solution_str.split("Assistant:")
    
    try:
        response = solution_str_split[1]
    except:
        print(f"Failed to process the solution_str to response")
        return 0.0, f'bad format: invalid response', None, None, None
    
    # 验证格式
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0.0, f'bad format: {reason}', None, None, None

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    elif response.endswith("<end_of_turn>"):
        response = response[:-len("<end_of_turn>")]

    response_stripped = response.rstrip()
    if not response_stripped.endswith('</answer>'):
        return 0.0, f'over length or not ending with </answer>', None, None, None

    # 提取match/refer/think结果
    match_refer_results, reason = extract_all_match_refer_think(response)
    if match_refer_results is None:
        return 0.0, f'bad format: {reason}', None, None, None
    
    # 计算refer调整分数
    if match_refer_results == 0.0:
        refer_adjust, refer_reason = match_refer_results, reason
    else:
        refer_adjust, refer_reason = calculate_refer_score(match_refer_results, ground_truth)
    # if refer_adjust is None:
    #     return 0.0, f'{refer_reason}', None, None, None
    
    # 提取lookup结果
    lookup_list, reason = extract_diseases_from_lookup(response)
    if lookup_list is None:
        lookup_score = 0.0
        lookup_reason = f'no lookup results: {reason}'
    else:
        lookup_list = [enhanced_normalize_answer(disease) for disease in lookup_list]
        ground_truth_normalized = [enhanced_normalize_answer(disease) for disease in ground_truth]
        
        if len(lookup_list) > 10:
            lookup_score = 0.0
            lookup_reason = f'too many lookup diseases: {len(lookup_list)} > 10'
        else:
            lookup_score = calculate_token_similarity_score(lookup_list, ground_truth_normalized)
            lookup_reason = f'lookup score: {lookup_score:.3f} based on token similarity'
    
    # 提取answer结果
    answer_list, reason = extract_diseases_from_textbf(response)
    if answer_list is None:
        answer_score = 0.0
        answer_reason = f'no answer results: {reason}'
    else:
        answer_list = [enhanced_normalize_answer(disease) for disease in answer_list]
        ground_truth_normalized = [enhanced_normalize_answer(disease) for disease in ground_truth]
        original_answer_score = calculate_token_similarity_score(answer_list, ground_truth_normalized)
        answer_score = calculate_adjusted_answer_score(original_answer_score, refer_adjust)
        
        if refer_adjust is None:
            answer_reason = f'answer score: {answer_score:.3f} (constraint violation, score set to 0)'
        else:
            base_adjusted = 0.2 + 0.6 * original_answer_score
            answer_reason = f'answer score: {answer_score:.3f} (base: {base_adjusted:.3f}, refer_adjust: {refer_adjust:.3f})'
    
    # 计算最终分数（现在只有lookup和answer，权重重新分配）
    # 由于refer现在是调整因子而不是独立分数，重新分配权重：lookup: 0.05/(0.05+0.15) = 0.25, answer: 0.15/(0.05+0.15) = 0.75
    final_score = lookup_score * 0.05 + answer_score * 0.95
    
    # 确保最终分数不低于0
    final_score = max(0.0, final_score)
    final_score = min(1.0, final_score)
    
    detailed_reason = f'Final score: {final_score:.3f} | Refer adjust: {refer_adjust} ({refer_reason}) | Lookup: {lookup_score:.3f} ({lookup_reason}) | Answer: {answer_score:.3f} ({answer_reason})'    
    return final_score, detailed_reason, refer_adjust, lookup_score, answer_score