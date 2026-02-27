import json
import requests
import time
from typing import List, Dict, Optional

class PubmedSearchService:
    def __init__(self, pubmed_port: int = 8000):
        """Initialize the search service with the retrieval API endpoint"""
        self.search_url = f"http://0.0.0.0:{pubmed_port}/retrieve"
        
    def _format_search_results(self, retrieval_results: List[Dict]) -> str:
        """Format the search results into a readable string"""
        formatted_text = ''
        for idx, doc_item in enumerate(retrieval_results):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_text += f"Doc {idx+1}(Title: {title}) {text}\n"
        return formatted_text

    def get_documents(self, querys_str: str, topk: int = 3) -> Dict:
        """
        将输入的 querys_str 视为一个完整的 query 进行检索
        """
        # 不再根据逗号分割，直接去除首尾空格后放入列表
        query_list = [querys_str.strip()]
        
        # 如果字符串为空，则直接返回空结果
        if not query_list[0]:
            return {}

        # 准备请求负载
        payload = {
            "queries": query_list,
            "topk": topk,
            "return_scores": True
        }
        
        # 发送请求到检索 API
        try:
            response = requests.post(self.search_url, json=payload)
            response.raise_for_status() # 检查 HTTP 状态码
            results = response.json()['result']
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
        
        # 处理结果 (此时 results 列表里只会有一个元素)
        result_dict = {}
        for query, result in zip(query_list, results):
            result_dict[query] = {
                "PMC": self._format_search_results(result)
            }
        
        return result_dict


def test_mapper():
    """Test the search service functionality"""
    mapper = PubmedSearchService(pubmed_port=8000)
    
    # Test case - string of query names
    # test_querys_str = "Legius syndrome, Acute transverse myelitis, Some rare query, Leukemia"
    test_querys_str = "relations between fever and cough"
    
    result = mapper.get_documents(test_querys_str, topk=3)
    print("Test Results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(type(result))
    # print(type(result["Leukemia"]))

if __name__ == "__main__":
    test_mapper()