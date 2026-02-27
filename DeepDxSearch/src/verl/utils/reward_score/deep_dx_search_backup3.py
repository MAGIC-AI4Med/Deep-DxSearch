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

def extract_tag_pairs(text: str, tag_name: str) -> Optional[List[str]]:
    """
    提取指定标签的所有配对内容
    
    配对规则：
    - 找到所有结束标签位置，划分区间
    - 第1个结束标签：在[0, pos1)中找开始标签
    - 第2个结束标签：在[pos1, pos2)中找开始标签
    - 第n个结束标签：在[pos(n-1), posn)中找开始标签
    - 每个区间内找最后一个（最近的）开始标签
    
    Args:
        text: 输入文本
        tag_name: 标签名（不含<>），如 'think', 'lookup' 等
    
    Returns:
        None: 如果没有结束标签，或者任何一个结束标签在其区间内找不到开始标签
        List[str]: 配对内容列表（按结束标签出现顺序）
    """
    open_tag = f'<{tag_name}>'
    close_tag = f'</{tag_name}>'
    
    # 找出所有结束标签的位置
    close_positions = []
    pos = 0
    while pos < len(text):
        close_pos = text.find(close_tag, pos)
        if close_pos != -1:
            close_positions.append(close_pos)
            pos = close_pos + 1
        else:
            break
    
    # 如果没有结束标签，返回 None
    if not close_positions:
        return None
    
    # 对每个结束标签，在其对应的区间内找开始标签
    contents = []
    
    for i, close_pos in enumerate(close_positions):
        # 确定搜索区间的起始位置
        if i == 0:
            interval_start = 0
        else:
            # 从上一个结束标签之后开始搜索
            interval_start = close_positions[i - 1] + len(close_tag)
        
        interval_end = close_pos
        
        # 在区间内找最后一个（最近的）开始标签
        interval_text = text[interval_start:interval_end]
        last_open_pos_in_interval = interval_text.rfind(open_tag)
        
        if last_open_pos_in_interval == -1:
            # 在区间内没有找到开始标签，格式错误
            return None
        
        # 计算开始标签在原文本中的实际位置
        actual_open_pos = interval_start + last_open_pos_in_interval
        
        # 提取配对内容
        content_start = actual_open_pos + len(open_tag)
        content_end = close_pos
        content = text[content_start:content_end]
        contents.append(content)
    
    return contents


def extract_tag_pairs_with_positions(text: str, tag_name: str) -> Optional[List[Tuple[int, str]]]:
    """
    提取指定标签的所有配对内容及其结束标签位置
    
    Args:
        text: 输入文本
        tag_name: 标签名（不含<>），如 'lookup', 'match' 等
    
    Returns:
        None: 如果没有结束标签，或者任何一个结束标签在其区间内找不到开始标签
        List[Tuple[int, str]]: (结束标签位置, 配对内容) 的列表
    """
    open_tag = f'<{tag_name}>'
    close_tag = f'</{tag_name}>'
    
    # 找出所有结束标签的位置
    close_positions = []
    pos = 0
    while pos < len(text):
        close_pos = text.find(close_tag, pos)
        if close_pos != -1:
            close_positions.append(close_pos)
            pos = close_pos + 1
        else:
            break
    
    # 如果没有结束标签，返回 None
    if not close_positions:
        return None
    
    # 对每个结束标签，在其对应的区间内找开始标签
    results = []
    
    for i, close_pos in enumerate(close_positions):
        # 确定搜索区间的起始位置
        if i == 0:
            interval_start = 0
        else:
            # 从上一个结束标签之后开始搜索
            interval_start = close_positions[i - 1] + len(close_tag)
        
        interval_end = close_pos
        
        # 在区间内找最后一个（最近的）开始标签
        interval_text = text[interval_start:interval_end]
        last_open_pos_in_interval = interval_text.rfind(open_tag)
        
        if last_open_pos_in_interval == -1:
            # 在区间内没有找到开始标签，格式错误
            return None
        
        # 计算开始标签在原文本中的实际位置
        actual_open_pos = interval_start + last_open_pos_in_interval
        
        # 提取配对内容
        content_start = actual_open_pos + len(open_tag)
        content_end = close_pos
        content = text[content_start:content_end]
        
        # 保存结束标签位置和内容
        results.append((close_pos, content))
    
    return results


def extract_trajectory(text: str) -> Optional[str]:
    """
    按照顺序提取 lookup 和 match 标签对的出现轨迹
    
    Args:
        text: 输入文本
    
    Returns:
        Optional[str]: 轨迹字符串，如 "lookup,match,lookup,match"，出错则返回 None
    """
    try:
        # 提取所有 lookup 和 match pair 及其结束标签位置
        lookup_pairs = extract_tag_pairs_with_positions(text, 'lookup')
        match_pairs = extract_tag_pairs_with_positions(text, 'match')
        
        # 收集所有标签及其位置
        all_tags = []
        
        if lookup_pairs is not None:
            for close_pos, content in lookup_pairs:
                all_tags.append((close_pos, 'lookup'))
        
        if match_pairs is not None:
            for close_pos, content in match_pairs:
                all_tags.append((close_pos, 'match'))
        
        # 如果没有任何标签，返回 None
        if not all_tags:
            return None
        
        # 按结束标签位置排序
        all_tags.sort(key=lambda x: x[0])
        
        # 提取标签名称序列
        trajectory = ','.join([tag_name for _, tag_name in all_tags])
        
        return trajectory
        
    except Exception as e:
        return None


def extract_all_tag_pairs(text: str) -> dict:
    """
    提取所有指定类型的标签配对内容
    
    Args:
        text: 输入文本
    
    Returns:
        dict: {
            'think': None 或 List[str],
            'lookup': None 或 List[str],
            'match': None 或 List[str],
            'reflect': None 或 List[str],
            'guide': None 或 List[str],
            'refer': None 或 List[str],
            'diagnose': None 或 List[str]
        }
    """
    tag_names = ['think', 'lookup', 'match', 'reflect', 'guide', 'refer', 'diagnose']
    result = {}
    
    for tag_name in tag_names:
        result[tag_name] = extract_tag_pairs(text, tag_name)
    
    return result


def validate_format(text: str) -> tuple[bool, str]:
    """
    验证格式规则：
    1. <diagnose></diagnose> pair 至少出现一次，且最后一个</diagnose>必须是结束前的最后字符
    2. <lookup></lookup> pair 最多出现3次
    3. <match></match> pair 最多出现3次
    4. 最后一个<diagnose> pair中必须包含 \\textbf{}，且最多5次
    5. 除了<think></think>之间的内容，其他tag pair内容中不允许包含任何结束标签
    """
    
    # 提取所有tag pairs
    all_pairs = extract_all_tag_pairs(text)
    
    # 规则1: 检查diagnose pair
    diagnose_pairs = all_pairs.get('diagnose')
    if diagnose_pairs is None or len(diagnose_pairs) == 0:
        return False, "<diagnose></diagnose> pair must appear at least once"
    
    # 检查最后一个</diagnose>是否是结束前的最后字符
    last_diagnose_close_pos = text.rfind('</diagnose>')
    if last_diagnose_close_pos == -1:
        return False, "</diagnose> not found before end"
    
    # 去除尾部空白后检查
    text_stripped = text.rstrip()
    if not text_stripped.endswith('</diagnose>'):
        return False, "</diagnose> must be the last content (ignoring trailing whitespace)"
    
    # 规则2: 检查lookup pair数量
    lookup_pairs = all_pairs.get('lookup')
    if lookup_pairs is not None and len(lookup_pairs) > 3:
        return False, f"<lookup></lookup> pair appears {len(lookup_pairs)} times (max 3)"
    
    # 规则3: 检查match pair数量
    match_pairs = all_pairs.get('match')
    if match_pairs is not None and len(match_pairs) > 3:
        return False, f"<match></match> pair appears {len(match_pairs)} times (max 3)"
    
    # 规则4: 检查最后一个diagnose pair中的\\textbf{}
    last_diagnose_content = diagnose_pairs[-1]
    textbf_matches = re.findall(r'\\textbf\{.*?\}', last_diagnose_content)
    
    if not textbf_matches:
        return False, "Last <diagnose> pair must contain at least one \\textbf{}"
    
    if len(textbf_matches) > 5:
        return False, f"\\textbf{{}} appears {len(textbf_matches)} times in last <diagnose> pair (max 5)"
    
    # 规则5: 检查非think pair中不能包含结束标签
    # 定义所有可能的结束标签
    all_close_tags = ['</think>', '</lookup>', '</match>', '</reflect>', '</guide>', '</refer>', '</diagnose>']
    
    for tag_name, pairs in all_pairs.items():
        if pairs is None:
            continue
        
        # think标签内容允许包含任何结束标签，跳过检查
        if tag_name == 'think':
            continue
        
        # 检查其他标签的pair内容
        for idx, content in enumerate(pairs):
            for close_tag in all_close_tags:
                if close_tag in content:
                    return False, f"<{tag_name}></{tag_name}> pair #{idx+1} contains forbidden close tag: {close_tag}"
    
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
        # 使用新的extract_all_tag_pairs提取所有标签
        all_pairs = extract_all_tag_pairs(text)
        
        match_pairs = all_pairs.get('match')
        refer_pairs = all_pairs.get('refer')
        think_pairs = all_pairs.get('think')
        
        # 检查refer和think是否存在且数量相同
        if refer_pairs is None or think_pairs is None:
            if refer_pairs is None and think_pairs is None:
                return 0.0, "No Match Applied"
        
        # 检查match的情况
        has_match = match_pairs is not None
        
        if has_match:
            # 如果有match，数量必须与refer/think相同
            if len(match_pairs) != len(refer_pairs):
                return None, f"Mismatched counts: {len(match_pairs)} match tags but {len(refer_pairs)} refer tags"
        
        # 构建结果
        results = []
        num_iterations = len(refer_pairs)
        
        for i in range(num_iterations):
            # 处理match内容（phenotypes）- 可能为空
            phenotypes = []
            if has_match and match_pairs[i]:
                phenotypes = [p.strip() for p in match_pairs[i].strip().split(',') if p.strip()]
            
            # 处理refer内容（cases的JSON字符串）
            diseases = extract_diseases_from_cases(refer_pairs[i].strip())
            
            results.append({
                'iteration': i + 1,
                'phenotypes': phenotypes,
                'diseases': diseases,
                'has_match': has_match,
                'refer_raw': refer_pairs[i].strip()
            })
        
        return results, "successful extraction"
        
    except Exception as e:
        return None, f"Extract match/refer/think failed: {str(e)}"


def extract_all_diseases_from_lookup(text: str) -> tuple:
    """从所有lookup标签中提取疾病名称列表（每个lookup一个列表）"""
    try:
        # 使用extract_tag_pairs提取所有lookup内容
        lookup_pairs = extract_tag_pairs(text, 'lookup')
        
        if lookup_pairs is None:
            return None, "No lookup tag found"
        
        # 提取每个lookup中的疾病
        all_lookup_diseases = []
        for lookup_content in lookup_pairs:
            lookup_content = lookup_content.strip()
            if not lookup_content:
                all_lookup_diseases.append([])
                continue
            
            diseases = [disease.strip() for disease in lookup_content.split(',')]
            diseases = [d for d in diseases if d]
            all_lookup_diseases.append(diseases)
        
        return all_lookup_diseases, "successful extraction"
        
    except Exception as e:
        return None, f"Extract from lookup failed: {str(e)}"


def extract_diseases_from_textbf(text: str) -> tuple:
    """从答案中提取所有\textbf{}内的疾病名称"""
    try:
        # 使用extract_tag_pairs提取diagnose内容
        diagnose_pairs = extract_tag_pairs(text, 'diagnose')
        
        if diagnose_pairs is None:
            return None, "No diagnose tag found"
        
        # 取最后一个diagnose的内容
        diagnose_content = diagnose_pairs[-1]
        
        textbf_pattern = r"\\textbf{([^}]*)}"
        textbf_matches = re.findall(textbf_pattern, diagnose_content)
        
        if not textbf_matches:
            return None, "No textbf found"
        
        return textbf_matches, "successful extraction"
    except Exception as e:
        return None, f"Extract from diagnose box failed: {str(e)}"

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
    """计算refer调整分数（用于调整diagnose_score，范围[-0.3, 0.5]或None）"""
    # 统计使用match的次数
    match_count = sum(1 for result in match_refer_results if result['has_match'])
    
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

def calculate_adjusted_diagnose_score(original_score: float, refer_adjust: Optional[float]) -> float:
    """计算调整后的diagnose分数：保底0.2分，最高0.8分，然后根据refer_adjust调整"""
    base_adjusted = 0.2 + 0.6 * original_score
    
    if refer_adjust is None:
        # 如果refer约束违背，diagnose分数为0
        return 0.0
    
    # 使用refer_adjust调整diagnose分数
    final_score = base_adjusted + refer_adjust
    
    # 确保分数在合理范围内（但可以为负）
    return final_score

def compute_score(tokenizer, solution_str, ground_truth, trajectory_stats=None) -> tuple:
    """主评估函数
    
    Args:
        trajectory_stats: 字典，记录不同trajectory的出现次数，格式 {trajectory: count}
    
    Returns:
        tuple: (final_score, detailed_reason, refer_adjust, lookup_score, diagnose_score, trajectory, diversity_coefficient)
    """
    # 处理模型输出格式
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
        return 0.0, f'bad format: invalid response', None, None, None, None, 0.0

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    elif response.endswith("<end_of_turn>"):
        response = response[:-len("<end_of_turn>")]

    response_stripped = response.rstrip()
    if not response_stripped.endswith('</diagnose>'):
        return 0.0, f'over length or not ending with </diagnose>', None, None, None, None, 0.0
    
    # 提取trajectory
    trajectory = extract_trajectory(response)
    
    # 验证格式
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0.0, f'bad format: {reason}', None, None, None, trajectory, 0.0
    

    # 提取match/refer/think结果
    match_refer_results, reason = extract_all_match_refer_think(response)
    if match_refer_results is None:
        return 0.0, f'bad format: {reason}', None, None, None, trajectory, 0.0
    
    # 计算refer调整分数
    if match_refer_results == 0.0:
        refer_adjust, refer_reason = match_refer_results, reason
    else:
        refer_adjust, refer_reason = calculate_refer_score(match_refer_results, ground_truth)
    
    # 提取所有lookup结果并计算平均分数
    all_lookup_lists, reason = extract_all_diseases_from_lookup(response)
    if all_lookup_lists is None:
        lookup_score = 0.0
        lookup_reason = f'no lookup results: {reason}'
    else:
        # 检查每个lookup的疾病数量
        too_many_diseases = []
        valid_lookups = []
        
        for idx, lookup_list in enumerate(all_lookup_lists):
            if len(lookup_list) > 10:
                too_many_diseases.append(f"lookup #{idx+1}: {len(lookup_list)} diseases")
            else:
                valid_lookups.append(lookup_list)
        
        if too_many_diseases:
            lookup_score = 0.0
            lookup_reason = f'too many lookup diseases: {"; ".join(too_many_diseases)}'
        elif not valid_lookups:
            lookup_score = 0.0
            lookup_reason = 'no valid lookup results'
        else:
            # 计算所有有效lookup的平均相似度分数
            ground_truth_normalized = [enhanced_normalize_answer(disease) for disease in ground_truth]
            
            lookup_scores = []
            for lookup_list in valid_lookups:
                normalized_lookup = [enhanced_normalize_answer(disease) for disease in lookup_list]
                score = calculate_token_similarity_score(normalized_lookup, ground_truth_normalized)
                lookup_scores.append(score)
            
            lookup_score = sum(lookup_scores) / len(lookup_scores)
            lookup_reason = f'lookup score: {lookup_score:.3f} (average of {len(lookup_scores)} lookups)'
    
    # 提取diagnose结果
    diagnose_list, reason = extract_diseases_from_textbf(response)
    if diagnose_list is None:
        diagnose_score = 0.0
        diagnose_reason = f'no diagnose results: {reason}'
    else:
        diagnose_list = [enhanced_normalize_answer(disease) for disease in diagnose_list]
        ground_truth_normalized = [enhanced_normalize_answer(disease) for disease in ground_truth]
        original_diagnose_score = calculate_token_similarity_score(diagnose_list, ground_truth_normalized)
        diagnose_score = calculate_adjusted_diagnose_score(original_diagnose_score, refer_adjust)
        
        if refer_adjust is None:
            diagnose_reason = f'diagnose score: {diagnose_score:.3f} (constraint violation, score set to 0)'
        else:
            base_adjusted = 0.2 + 0.6 * original_diagnose_score
            diagnose_reason = f'diagnose score: {diagnose_score:.3f} (base: {base_adjusted:.3f}, refer_adjust: {refer_adjust:.3f})'
    
    # 计算最终分数
    final_score = lookup_score * 0.05 + diagnose_score * 0.95
    
    # 确保最终分数不低于0
    final_score = max(0.0, final_score)
    final_score = min(1.0, final_score)
    
    # ============= 新增：计算 diversity coefficient =============
    diversity_coefficient = 1.0  # 默认系数为1
    
    if trajectory_stats is not None:
        total_count = sum(trajectory_stats.values())
        
        # 只有当总数大于320时才应用惩罚
        if total_count > 320:
            if trajectory in trajectory_stats:
                # 计算该trajectory占总数的比例
                ratio = trajectory_stats[trajectory] / total_count
                # 系数 = sqrt(1 - ratio)
                diversity_coefficient = (1.0 - ratio) ** 0.5
            # 如果trajectory不存在，保持系数为1.0
        # 如果总数<=320，保持系数为1.0
    
    # 应用系数到最终分数
    final_score_with_diversity = final_score * diversity_coefficient
    final_score_with_diversity = max(0.0, min(1.0, final_score_with_diversity))
    # ============= 新增结束 =============
    
    detailed_reason = f'Final score: {final_score_with_diversity:.3f} (base: {final_score:.3f}, diversity_coef: {diversity_coefficient:.3f}) | Refer adjust: {refer_adjust} ({refer_reason}) | Lookup: {lookup_score:.3f} ({lookup_reason}) | Diagnose: {diagnose_score:.3f} ({diagnose_reason})'
    
    return final_score_with_diversity, detailed_reason, refer_adjust, lookup_score, diagnose_score, trajectory, diversity_coefficient