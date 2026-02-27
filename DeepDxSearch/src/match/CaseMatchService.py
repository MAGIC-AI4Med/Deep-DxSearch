import json
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

# 只在punkt不存在时才下载
nltk_data_dir = '.../your/path/to/DeepDxSearch/src/match/nltk_data'
nltk.data.path.insert(0, nltk_data_dir)
# 只在punkt不存在时才下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    raise ValueError("no tokenizers/punkt")
    nltk.download('punkt')

def has_intersection(predicted_diseases, ground_truth_diseases):
    """Check if there is any intersection between predicted diseases and ground truth."""
    # Convert to lowercase for case-insensitive matching
    pred_lower = [d.lower() for d in predicted_diseases]
    truth_lower = [d.lower() for d in ground_truth_diseases]
    
    # Check for direct matches
    for pred in pred_lower:
        if pred in truth_lower:
            return True
    
    # Check for substring matches (if a prediction is part of a ground truth or vice versa)
    for pred in pred_lower:
        for truth in truth_lower:
            if pred in truth or truth in pred:
                return True
    
    return False

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

class CaseMatchService:
    def __init__(self, source_path: str):
        """初始化搜索评估器"""
        # 加载案例数据
        with open(source_path, 'r', encoding='utf-8') as f:
            self.cases = json.load(f)
            
        # 准备BM25索引
        self.doc_texts = []
        self.doc_diseases = []
        self.doc_phenotypes = []
        self.doc_ids = []
        
        for doc_id, case in self.cases.items():
            text = case['Phenotype_Text'].lower()
            self.doc_texts.append(word_tokenize(text))
            self.doc_diseases.append(case['Disease_List'])
            self.doc_phenotypes.append(case['Phenotype_Text'])
            self.doc_ids.append(doc_id)
            
        self.bm25 = BM25Okapi(self.doc_texts)

    def match_diseases(self, query: str, top_n: int = 5) -> List[str]:
        """根据查询返回前top_n个案例的所有疾病列表"""
        query_tokens = word_tokenize(query.lower())
        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[-top_n:][::-1]
        
        all_diseases = []
        for idx in top_indices:
            all_diseases.extend(self.doc_diseases[idx])
        
        # 去重但保持顺序
        unique_diseases = list(dict.fromkeys(all_diseases))
        return unique_diseases
    
    def match_cases(self, query: str, top_n: int = 5) -> str:
        """搜索相似案例并返回详细信息"""
        query_tokens = word_tokenize(query.lower())
        doc_scores = self.bm25.get_scores(query_tokens)
        
        if len(doc_scores) == 0:
            return json.dumps([], ensure_ascii=False, indent=2)
        
        # 获取top-n索引
        top_indices = np.argsort(doc_scores)[-top_n:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            case_info = {
                # "case_id": self.doc_ids[idx],
                "phenotype": self.doc_phenotypes[idx],
                "diseases": self.doc_diseases[idx],
                # "similarity_score": float(doc_scores[idx])  # 添加相似度分数
            }
            results.append(case_info)
        
        return json.dumps(results, ensure_ascii=False, indent=2)

def evaluate_disease_matching_via_cases(val_data_path: str, search_db_path: str, top_n: int = 20):
    """
    通过match_cases方法评估疾病匹配效果
    
    Args:
        val_data_path: 验证数据路径 (disease_val.json)
        search_db_path: 搜索数据库路径 (大字典文件)
        top_n: 返回前n个相似案例
    """
    # 初始化搜索服务
    matcher = CaseMatchService(search_db_path)
    
    # 加载验证数据
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    total_cases = len(val_data)
    hit_count = 0
    results = []
    
    # 按Source分类统计
    source_stats = {}  # {'Source1': {'total': X, 'hit': Y}, ...}
    
    print(f"开始评估（通过match_cases方法），共有 {total_cases} 个测试案例...")
    print(f"使用top_{top_n}相似案例进行疾病匹配")
    
    for case_id, case_info in tqdm(val_data.items(), desc="处理案例"):
        # 获取查询的表型文本
        query_phenotype = case_info['Phenotype_Text']
        
        # 获取真实疾病列表
        ground_truth_diseases = case_info['Disease_List']
        
        # 获取数据来源
        source = case_info.get('Source', 'Unknown')
        
        # 初始化该Source的统计
        if source not in source_stats:
            source_stats[source] = {'total': 0, 'hit': 0}
        
        source_stats[source]['total'] += 1
        
        # 通过match_cases搜索相似案例
        cases_result = matcher.match_cases(query_phenotype, top_n)
        
        # 从匹配的案例中提取疾病
        predicted_diseases = extract_diseases_from_cases(cases_result)
        
        # 判断是否有交集
        has_hit = has_intersection(predicted_diseases, ground_truth_diseases)
        
        if has_hit:
            hit_count += 1
            source_stats[source]['hit'] += 1
        
        # 记录结果
        result = {
            'case_id': case_id,
            'ground_truth_diseases': ground_truth_diseases,
            'predicted_diseases': predicted_diseases,
            'has_hit': has_hit,
            'query_phenotype': query_phenotype,
            'source': source,
            'matched_cases_count': len(json.loads(cases_result))
        }
        results.append(result)
        
        # 可选：打印一些详细信息
        if len(results) <= 5:  # 只打印前5个案例的详细信息
            print(f"\n案例 {case_id}:")
            print(f"来源: {source}")
            print(f"真实疾病: {ground_truth_diseases}")
            print(f"匹配到的案例数: {result['matched_cases_count']}")
            print(f"从案例提取的疾病: {predicted_diseases}")
            print(f"是否命中: {has_hit}")
            
            # 显示匹配的案例详情
            matched_cases = json.loads(cases_result)
            # print(f"匹配案例详情:")
            # for i, case in enumerate(matched_cases[:3], 1):  # 只显示前3个
            #     print(f"  案例{i} (相似度: {case['similarity_score']:.4f}): {case['diseases']}")
    
    # 计算总体准确率
    accuracy = hit_count / total_cases if total_cases > 0 else 0
    
    # 计算各Source的准确率
    source_accuracies = {}
    for source, stats in source_stats.items():
        source_total = stats['total']
        source_hit = stats['hit']
        source_acc = source_hit / source_total if source_total > 0 else 0
        source_accuracies[source] = {
            'total': source_total,
            'hit': source_hit,
            'accuracy': source_acc
        }
    
    # 打印表格形式的结果
    print(f"\n=== 评估结果 (通过match_cases方法, top_{top_n}) ===")
    print("+----------------------+------------+-----------+-----------+")
    print("| Source               | Total Cases| Hit Cases | Accuracy  |")
    print("+----------------------+------------+-----------+-----------+")
    
    # 打印每个Source的结果
    for source, stats in source_accuracies.items():
        source_name = source[:18].ljust(18) + ".." if len(source) > 20 else source.ljust(20)
        print(f"| {source_name} | {str(stats['total']).center(10)} | {str(stats['hit']).center(9)} | {stats['accuracy']*100:7.2f}% |")
        print("+----------------------+------------+-----------+-----------+")
    
    # 打印总体结果
    print(f"| {'Overall'.ljust(20)} | {str(total_cases).center(10)} | {str(hit_count).center(9)} | {accuracy*100:7.2f}% |")
    print("+----------------------+------------+-----------+-----------+")
    
    return results, accuracy, source_accuracies

def compare_methods(val_data_path: str, search_db_path: str, top_n: int = 20):
    """
    比较match_diseases和match_cases+extract_diseases两种方法的结果
    """
    print(f"\n=== 比较两种方法的一致性 (top_{top_n}) ===")
    
    # 初始化搜索服务
    matcher = CaseMatchService(search_db_path)
    
    # 加载验证数据
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 随机选择几个案例进行比较
    import random
    sample_cases = dict(random.sample(list(val_data.items()), min(10, len(val_data))))
    
    inconsistent_count = 0
    
    for case_id, case_info in sample_cases.items():
        query_phenotype = case_info['Phenotype_Text']
        
        # 方法1: 直接调用match_diseases
        diseases_direct = matcher.match_diseases(query_phenotype, top_n)
        
        # 方法2: 通过match_cases然后提取疾病
        cases_result = matcher.match_cases(query_phenotype, top_n)
        diseases_via_cases = extract_diseases_from_cases(cases_result)
        
        # 检查一致性
        consistency = set(diseases_direct) == set(diseases_via_cases)
        
        if not consistency:
            inconsistent_count += 1
            print(f"\n❌ 案例 {case_id} 结果不一致:")
            print(f"   直接方法: {diseases_direct}")
            print(f"   案例方法: {diseases_via_cases}")
            
            # 分析差异
            only_in_direct = set(diseases_direct) - set(diseases_via_cases)
            only_in_cases = set(diseases_via_cases) - set(diseases_direct)
            if only_in_direct:
                print(f"   仅在直接方法中: {list(only_in_direct)}")
            if only_in_cases:
                print(f"   仅在案例方法中: {list(only_in_cases)}")
        else:
            print(f"✅ 案例 {case_id} 结果一致")
    
    consistency_rate = (len(sample_cases) - inconsistent_count) / len(sample_cases) * 100
    print(f"\n方法一致性: {consistency_rate:.1f}% ({len(sample_cases) - inconsistent_count}/{len(sample_cases)})")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    val_data_path = ".../your/path/to/DeepDxSearch/data/val_data_rare.json"
    search_db_path = ".../your/path/to/DeepDxSearch/src/match/match_source_rare.json"
    
    # 通过match_cases方法运行评估
    print("=== 使用match_cases方法进行评估 ===")
    results_via_cases, accuracy_via_cases, source_accuracies_via_cases = evaluate_disease_matching_via_cases(
        val_data_path=val_data_path,
        search_db_path=search_db_path,
        top_n=1
    )
    print(f"\n通过match_cases方法的总体准确率: {accuracy_via_cases:.4f}")
    
    # 比较两种方法的一致性
    compare_methods(val_data_path, search_db_path, top_n=20)