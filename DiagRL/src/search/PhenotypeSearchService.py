import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class PhenotypeSearchService:
    def __init__(self, map_path: str):
        """初始化疾病-指南映射器"""
        with open(map_path, 'r', encoding='utf-8') as f:
            self.guideline_data = json.load(f)
            
        # 准备BM25索引
        self.guideline_diseases = list(self.guideline_data.keys())
        self.disease_tokens = [word_tokenize(disease.lower()) for disease in self.guideline_diseases]
        self.bm25 = BM25Okapi(self.disease_tokens)
        
        # 构建疾病到症状的映射
        self.disease_to_symptoms = {}
        for disease, info in self.guideline_data.items():
            symptoms = info.get('symptom_list', [])
            # 限制最多15个症状
            if len(symptoms) > 15:
                symptoms = symptoms[:15]
            self.disease_to_symptoms[disease] = symptoms

    def find_best_match(self, disease: str) -> tuple[str, float]:
        """查找最匹配的疾病及其分数"""
        query_tokens = word_tokenize(disease.lower())
        scores = self.bm25.get_scores(query_tokens)
        best_idx = scores.argmax()
        best_score = scores[best_idx]
        
        # 检查是否有完全匹配（大小写不敏感）
        disease_lower = disease.lower()
        for guideline_disease in self.guideline_diseases:
            if disease_lower == guideline_disease.lower():
                return guideline_disease, float('inf')  # 完全匹配给最高分
        
        if best_score > 14:  # 分数高于14认为是好的匹配
            return self.guideline_diseases[best_idx], best_score
        return None, best_score

    def get_phenotypes_for_diseases(self, diseases_str: str, max_n: int = None) -> Dict:
        """
        获取疾病列表对应的指南映射
        
        Args:
            diseases_str: 逗号分隔的疾病名称字符串，如 "disease1, disease2, disease3"
            max_n: 最多返回的疾病数量，None表示返回所有
            
        Returns:
            Dict: 疾病-指南映射结果，格式为 {disease: {"guideline": [symptoms]}}
        """
        # 将输入字符串分割成疾病列表
        disease_list = [d.strip() for d in diseases_str.split(',') if d.strip()]
        
        # 如果设置了max_n，限制处理的疾病数量
        if max_n is not None:
            disease_list = disease_list[:max_n]
        
        result = {}
        for disease in disease_list:
            best_match, score = self.find_best_match(disease)
            if best_match:
                # 使用原始输入的疾病名称作为键
                result[disease] = {"guideline": self.disease_to_symptoms[best_match]}
            else:
                result[disease] = {"guideline": "no reference"}
        
        return result

if __name__ == "__main__":
    test_guideline_mapper()