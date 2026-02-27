import json
import os
import nltk
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from rapidfuzz import process, fuzz

# --- 1. NLTK 资源检查 (保持不变) ---
def ensure_nltk_resources(nltk_data_path: str = None):
    if nltk_data_path:
        nltk.data.path.insert(0, nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载 nltk punkt tokenizer...")
        nltk.download('punkt')

# 请确保这里的路径是你环境中的真实路径
CUSTOM_NLTK_PATH = '.../your/path/to/DeepDxSearch/src/match/nltk_data'
ensure_nltk_resources(CUSTOM_NLTK_PATH)

class PhenotypeLookupService:
    def __init__(self, map_path: str):
        """
        :param map_path: 合并后的 merged_disease_symptoms.json 路径
        """
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"找不到文件: {map_path}")

        print(f"正在加载 Guideline 数据: {map_path} ...")
        with open(map_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # --- 数据预处理 ---
        self.db: Dict[str, List[str]] = {}           
        self.lower_index: Dict[str, str] = {}        
        self.disease_names: List[str] = []           
        self.disease_tokens: List[List[str]] = []    

        for item in raw_data:
            for name, symptoms in item.items():
                clean_name = name.strip()
                lower_name = clean_name.lower()

                if clean_name not in self.db:
                    self.db[clean_name] = symptoms
                    self.disease_names.append(clean_name)
                    self.disease_tokens.append(word_tokenize(lower_name))
                    
                    if lower_name not in self.lower_index:
                        self.lower_index[lower_name] = clean_name

        # --- 构建 BM25 索引 ---
        self.bm25 = BM25Okapi(self.disease_tokens)
        print(f"初始化完成。加载了 {len(self.disease_names)} 种疾病。")

    def _format_symptoms(self, symptoms: List[str]) -> str:
        """简单的列表转字符串工具"""
        if not symptoms:
            return "no specific symptom information available"
        return ", ".join(symptoms)

    def get_phenotypes_for_diseases(self, query_string: str) -> Dict:
        """
        核心搜索函数
        :param query_string: 分号或逗号分隔的疾病查询字符串
        """
        if ';' in query_string:
            queries = [q.strip() for q in query_string.split(';') if q.strip()]
        else:
            queries = [q.strip() for q in query_string.split(',') if q.strip()]

        result_dict = {}

        for query in queries:
            query_lower = query.lower()
            query_tokens = word_tokenize(query_lower)
            
            # --- A. 寻找 Exact Match (优先) ---
            exact_match_str = None
            found_exact_name = None
            
            if query_lower in self.lower_index:
                found_exact_name = self.lower_index[query_lower]
                symptoms = self.db.get(found_exact_name, [])
                # Exact match 保持原有格式: "some typical symptoms of {name} are {symptoms}"
                symptoms_str = self._format_symptoms(symptoms)
                exact_match_str = f"some typical symptoms of {found_exact_name} are {symptoms_str}"
            
            # --- B. 寻找 Related References ---
            related_refs = []
            
            # 1. BM25 粗筛 Top 20
            candidates = self.bm25.get_top_n(query_tokens, self.disease_names, n=20)
            
            # 2. RapidFuzz 精排
            scored_candidates = process.extract(
                query, 
                candidates, 
                scorer=fuzz.token_sort_ratio, 
                limit=10
            )
            
            # 3. 过滤和格式化
            count = 0
            for name, score, _ in scored_candidates:
                # 过滤掉 Exact Match
                if found_exact_name and name.lower() == found_exact_name.lower():
                    continue
                
                # --- 修改点 2: 阈值改为 > 20 ---
                if score > 20: 
                    symptoms = self.db.get(name, [])
                    symptoms_str = self._format_symptoms(symptoms)
                    
                    # --- 修改点 1: Related Reference 格式变为 "{name}: some typical symptoms are..." ---
                    related_str = f"{name}: some typical symptoms are {symptoms_str}"
                    related_refs.append(related_str)
                    count += 1
                
                if count >= 3: # 最多取 3 个
                    break
            
            # --- C. 组装结果 ---
            result_dict[query] = {
                "exact_match": exact_match_str if exact_match_str else f"Exact match for '{query}' not found.",
                "related_reference": related_refs
            }

        return result_dict

# --- 测试代码 ---
def test_new_service():
    new_map_path = '.../your/path/to/disease_guidelines.json'
    
    try:
        service = PhenotypeLookupService(map_path=new_map_path)
        
        # 测试 query (包含完全匹配和模糊匹配)
        query_str = "Alexander disease; Seborrheic Dermatitis of Scalp; Alexander; scalp dermatitis"
        
        print("\n=== 开始搜索 ===")
        results = service.get_phenotypes_for_diseases(query_str)
        
        # 打印结果
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    test_new_service()