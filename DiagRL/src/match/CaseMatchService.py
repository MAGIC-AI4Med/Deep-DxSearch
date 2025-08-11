import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Union
import numpy as np
from tqdm import tqdm

class CaseMatchService:
    def __init__(self, source_path: str):
        """初始化基于BioLORD的搜索评估器"""
        
        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载案例数据
        if source_path.endswith('.pt'):
            self.cases = torch.load(source_path, map_location='cpu')
        else:
            with open(source_path, 'r', encoding='utf-8') as f:
                self.cases = json.load(f)
        
        # 加载BioLORD模型
        model_path = 'FremyCompany/BioLORD-2023-C'
        print("Loading BioLORD model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 准备案例数据 - 存储症状级别的嵌入
        self.doc_diseases = []
        self.doc_phenotypes = []
        self.doc_ids = []
        self.doc_symptom_embeddings = []  # 存储每个案例的症状嵌入列表
        
        print("Processing cases and computing embeddings...")
        self._prepare_case_embeddings()
        
        # 嵌入缓存
        self.embedding_cache = {}
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 在GPU上计算"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled
    
    def _prepare_case_embeddings(self):
        """预处理所有案例，计算症状级别的嵌入特征"""
        
        for doc_id, case in tqdm(self.cases.items(), desc="Computing case embeddings"):
            # 提取基本信息
            self.doc_diseases.append(case['Disease_List'])
            self.doc_phenotypes.append(case['Phenotype_Text'])
            self.doc_ids.append(doc_id)
            
            # 计算案例的症状嵌入
            if 'BioLORD' in case:
                # 如果已经有BioLORD嵌入，直接使用
                case_embedding = case['BioLORD']  # shape: [num_symptoms, embedding_dim]
                if isinstance(case_embedding, torch.Tensor) and case_embedding.numel() > 0:
                    # 直接使用症状级别的嵌入
                    self.doc_symptom_embeddings.append(case_embedding.to(self.device))
                else:
                    # 如果嵌入为空，使用空张量
                    self.doc_symptom_embeddings.append(torch.empty(0, 768).to(self.device))
            else:
                # 如果没有预计算的嵌入，实时计算
                symptom_embeddings = self._compute_case_symptom_embeddings(case['Phenotype_Text'])
                self.doc_symptom_embeddings.append(symptom_embeddings)
        
        print(f"Prepared embeddings for {len(self.doc_symptom_embeddings)} cases")
    
    def _compute_case_symptom_embeddings(self, phenotype_text: str):
        """计算单个案例的症状级别嵌入特征"""
        # 解析症状
        symptoms = [symptom.strip() for symptom in phenotype_text.split(',')]
        symptoms = [symptom for symptom in symptoms if symptom]
        
        if not symptoms:
            return torch.empty(0, 768).to(self.device)
        
        # 计算症状嵌入
        symptom_embeddings = []
        
        for symptom in symptoms:
            if symptom in self.embedding_cache:
                embedding = self.embedding_cache[symptom].to(self.device)
            else:
                embedding = self._get_symptom_embedding(symptom)
                self.embedding_cache[symptom] = embedding.cpu()  # 缓存CPU版本
                embedding = embedding.to(self.device)
            
            symptom_embeddings.append(embedding)
        
        # 返回症状嵌入矩阵
        if symptom_embeddings:
            return torch.stack(symptom_embeddings)  # [num_symptoms, 768]
        else:
            return torch.empty(0, 768).to(self.device)
    
    def _get_symptom_embedding(self, symptom: str):
        """获取单个症状的嵌入"""
        with torch.no_grad():
            encoded_input = self.tokenizer(
                symptom,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            # 移动到GPU
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # 计算嵌入
            model_output = self.model(**encoded_input)
            embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1).squeeze()
            
            # 返回CPU版本供缓存
            return embedding.cpu()
    
    def _compute_query_symptom_embeddings(self, query: str):
        """计算查询的症状级别嵌入特征"""
        # 解析查询中的症状
        symptoms = [symptom.strip() for symptom in query.split(',')]
        symptoms = [symptom for symptom in symptoms if symptom]
        
        if not symptoms:
            return torch.empty(0, 768).to(self.device)
        
        # 获取所有症状的嵌入
        symptom_embeddings = []
        
        for symptom in symptoms:
            if symptom in self.embedding_cache:
                embedding = self.embedding_cache[symptom].to(self.device)
            else:
                embedding = self._get_symptom_embedding(symptom).to(self.device)
                self.embedding_cache[symptom] = embedding.cpu()  # 缓存CPU版本
            
            symptom_embeddings.append(embedding)
        
        # 返回症状嵌入矩阵
        if symptom_embeddings:
            return torch.stack(symptom_embeddings)  # [num_query_symptoms, 768]
        else:
            return torch.empty(0, 768).to(self.device)
    
    def compute_similarity_score(self, query_features, reference_features):
        """
        计算相似度分数 - 全部在GPU上
        query_features: n x 768 (查询案例的表型特征)
        reference_features: m x 768 (参考案例的表型特征)
        返回: 单个相似度分数
        """
        if query_features.size(0) == 0 or reference_features.size(0) == 0:
            return 0.0
        
        # 确保都在GPU上
        query_features = query_features.to(self.device)
        reference_features = reference_features.to(self.device)
        
        # 计算相似度矩阵 n x m - 在GPU上
        similarity_matrix = torch.mm(query_features, reference_features.T)
        
        # 每行取最大值，然后平均 - 在GPU上
        max_similarities = torch.max(similarity_matrix, dim=1)[0]  # n
        avg_similarity = torch.mean(max_similarities).item()
        
        return avg_similarity
    
    def _compute_similarities(self, query_symptom_embeddings):
        """计算查询与所有案例的相似度"""
        if len(self.doc_symptom_embeddings) == 0:
            return np.array([])
        
        similarities = []
        
        for doc_symptom_embeddings in self.doc_symptom_embeddings:
            similarity = self.compute_similarity_score(query_symptom_embeddings, doc_symptom_embeddings)
            similarities.append(similarity)
        
        return np.array(similarities)

    def match_cases(self, query: str, top_n: int = 5) -> Union[List[Dict], str]:
        """搜索相似案例并返回详细信息"""
        
        # 计算查询的症状嵌入
        query_symptom_embeddings = self._compute_query_symptom_embeddings(query)
        
        # 计算相似度
        similarities = self._compute_similarities(query_symptom_embeddings)
        
        if len(similarities) == 0:
            return []
        
        # 获取top-n索引
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            results.append({
                "phenotype": self.doc_phenotypes[idx],
                "disease": self.doc_diseases[idx],
            })
        
        # return ", ".join(results)
        return json.dumps(results, ensure_ascii=False, indent=2)

    def match_diseases(self, query: str, top_n: int = 5) -> str:
        """只返回疾病名称列表"""
        
        # 计算查询的症状嵌入
        query_symptom_embeddings = self._compute_query_symptom_embeddings(query)
        
        # 计算相似度
        similarities = self._compute_similarities(query_symptom_embeddings)
        
        if len(similarities) == 0:
            return ""
        
        # 获取top-n索引
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # 收集疾病名称
        diseases = []
        for idx in top_indices:
            diseases.extend(self.doc_diseases[idx])
        
        return ", ".join(list(dict.fromkeys(diseases)))  # 去重并用逗号连接


def has_intersection(predicted_diseases, ground_truth_diseases):
    """Check if there is any intersection between predicted diseases and ground truth."""
    # Convert to lowercase for case-insensitive matching
    predicted_diseases = [disease.strip() for disease in predicted_diseases.split(',')]
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

    

if __name__ == "__main__":
    # test_evaluation()
    test_specific_phenotypes()