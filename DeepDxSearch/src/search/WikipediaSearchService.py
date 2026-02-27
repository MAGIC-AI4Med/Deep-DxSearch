import json
import requests
import time
from typing import List, Dict, Optional

class WikiSearchService:
    def __init__(self, wiki_port: int = 8001):
        """初始化搜索服务"""
        self.search_url = f"http://0.0.0.0:{wiki_port}/retrieve"
        
    def _format_search_results(self, retrieval_results: List[Dict]) -> str:
        """将搜索结果格式化为可读字符串"""
        formatted_text = ''
        for idx, doc_item in enumerate(retrieval_results):
            content = doc_item['document']['contents']
            # 拆分标题和正文
            parts = content.split("\n")
            title = parts[0]
            text = "\n".join(parts[1:])
            formatted_text += f"Doc {idx+1}(Title: {title}) {text}\n"
        return formatted_text

    def get_documents(self, query_str: str, topk: int = 3) -> Dict:
        """
        将输入的 query_str 作为一个整体进行检索
        
        Args:
            query_str: 完整的查询字符串（不再按逗号分割）
            topk: 每个查询检索的文档数量
        """
        # 直接使用原始字符串，仅去除两端的空格
        query_list = [query_str.strip()]
        
        if not query_list[0]:
            return {"error": "Empty query string"}

        # 准备请求负载
        payload = {
            "queries": query_list,
            "topk": topk,
            "return_scores": True
        }
        
        # 发送请求
        try:
            response = requests.post(self.search_url, json=payload)
            response.raise_for_status()
            results = response.json()['result']
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
        
        # 处理结果
        result_dict = {}
        for query, result in zip(query_list, results):
            result_dict[query] = {
                "WIKI": self._format_search_results(result)
            }
        
        return result_dict

def test_mapper():
    """测试搜索服务"""
    mapper = WikiSearchService(wiki_port=8001)
    
    # 现在的输入：即使带逗号，也会被当做一个 query 整体处理
    test_query_str = ", how far is the moon"
    
    result = mapper.get_documents(test_query_str, topk=3)
    print("Test Results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_mapper()