import json
import requests
from typing import List, Dict, Optional

class TextbookSearchService:
    def __init__(self, textbook_port: int = 8002):
        """Initialize the search service with the textbook retrieval API endpoint"""
        self.search_url = f"http://0.0.0.0:{textbook_port}/retrieve"
        
    def _format_search_results(self, retrieval_results: List[Dict]) -> str:
        """Format the search results into a readable string"""
        formatted_text = ''
        for idx, doc_item in enumerate(retrieval_results):
            content = doc_item['document']['contents']
            # 分离标题与正文
            parts = content.split("\n")
            title = parts[0]
            text = "\n".join(parts[1:])
            formatted_text += f"Doc {idx+1}(Title: {title}) {text}\n"
        return formatted_text

    def get_documents(self, query_str: str, topk: int = 3) -> Dict:
        """
        将 query_str 作为一个完整的 query 进行检索
        
        Args:
            query_str: 完整的查询字符串（不进行分割）
            topk: 检索的文档数量
            
        Returns:
            Dict: 包含检索结果的字典
        """
        # 直接去除首尾空格，不再使用 split(',')
        query_list = [query_str.strip()]
        
        if not query_list[0]:
            return {"error": "Query string is empty"}

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
            return {"error": f"Textbook API request failed: {str(e)}"}
        
        # 封装结果
        result_dict = {}
        for query, result in zip(query_list, results):
            result_dict[query] = {
                "TEXTBOOK": self._format_search_results(result)
            }
        
        return result_dict

def test_mapper():
    """测试教科书检索服务"""
    mapper = TextbookSearchService(textbook_port=8002)
    
    # 即使包含逗号，现在也会被当作一个整体 query 处理
    test_query_str = "Human anatomy, specific functions of the heart"
    
    # 移除 max_n 参数，因为现在只有一个整体 query
    result = mapper.get_documents(test_query_str, topk=2)
    
    print("Textbook Search Results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_mapper()