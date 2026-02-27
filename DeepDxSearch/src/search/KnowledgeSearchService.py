import re
import json
import sglang as sgl
from typing import List, Dict, Tuple, Optional

# 请确保这三个文件在同级目录下，或者在 Python Path 中
try:
    # 尝试相对导入（作为模块使用时）
    from .WikipediaSearchService import WikiSearchService
    from .TextbookSearchService import TextbookSearchService
    from .PubmedSearchService import PubmedSearchService
except ImportError:
    # 回退到绝对导入（直接运行时）
    try:
        from WikipediaSearchService import WikiSearchService
        from TextbookSearchService import TextbookSearchService
        from PubmedSearchService import PubmedSearchService
    except ImportError as e:
        # 如果还是失败，添加当前目录到路径
        _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        if _CURRENT_DIR not in sys.path:
            sys.path.insert(0, _CURRENT_DIR)
        
        from WikipediaSearchService import WikiSearchService
        from TextbookSearchService import TextbookSearchService
        from PubmedSearchService import PubmedSearchService

# ==========================================
# 1. SGLang Summarization Prompt
# ==========================================

@sgl.function
def summarize_query(s, query_text: str, search_content: str):
    # 按照需求：尝试基于 search_content 总结回答。
    # 如果不相关或为空，则基于自身知识回答。
    # 限制在 64 个单词以内。
    s += (
        f"Query: {query_text}\n"
        f"Context: {search_content}\n\n"
        f"Instruction: Answer the query based on the context. "
        f"If the context is irrelevant or empty, answer using your own knowledge. "
        f"Be precise, accurate, and use fewer than 64 words.\n"
        f"Answer:"
    )
    # max_tokens=128 足够覆盖 64 个单词
    s += sgl.gen("response", max_tokens=128, stop=["\n", "Query:"])

# ==========================================
# 2. Search Manager Class
# ==========================================

class KnowledgeSearchService:
    def __init__(self, 
                 wiki_port: int = 8001, 
                 pmc_port: int = 8000, 
                 book_port: int = 8002,
                 sgl_port: int = 31680):
        """
        初始化管理器，连接各个搜索服务和 SGLang 后端
        """
        # 1. 初始化搜索客户端
        self.wiki_service = WikiSearchService(wiki_port=wiki_port)
        self.pmc_service = PubmedSearchService(pubmed_port=pmc_port)
        self.book_service = TextbookSearchService(textbook_port=book_port)
        
        # 2. 初始化 SGLang 后端
        try:
            backend = sgl.RuntimeEndpoint(f"http://localhost:{sgl_port}")
            sgl.set_default_backend(backend)
        except Exception as e:
            print(f"[Warning] SGLang backend init failed: {e}")

        # 3. 预编译正则：匹配 |TAG| 形式
        # 使用捕获组 () 以便 split 后保留标签
        self.tag_pattern = re.compile(r'(\|WIKI\||\|PMC\||\|BOOK\|)')

    def _parse_single_query_string(self, raw_str: str) -> List[Tuple[str, str]]:
        """
        解析单个字符串，例如: "|WIKI| q1. |PMC| q2"
        返回: [('WIKI', 'q1'), ('PMC', 'q2')]
        """
        # 分割字符串
        parts = self.tag_pattern.split(raw_str)
        # 结果类似于 ['', '|WIKI|', ' q1 content. ', '|PMC|', ' q2 content']
        
        parsed_results = []
        current_source = None
        
        for part in parts:
            if not part.strip():
                continue
                
            stripped_part = part.strip()
            
            # 判断是否是标签
            if stripped_part in ['|WIKI|', '|PMC|', '|BOOK|']:
                current_source = stripped_part.strip('|') # 去掉 | 得到 WIKI/PMC/BOOK
            else:
                # 是内容部分
                if current_source:
                    # 去掉结尾的句号和两端空白
                    clean_query = stripped_part.strip(' .')
                    if clean_query:
                        parsed_results.append((current_source, clean_query))
                else:
                    # 如果开头没有标签的内容，忽略或记录
                    pass
                    
        return parsed_results

    def _fetch_search_result(self, source: str, query: str) -> str:
        """
        执行具体的搜索调用 (串行阻塞)
        """
        try:
            result_dict = {}
            target_key = ""
            
            # 默认返回 3 个 doc (由 Service 内部默认值或这里指定)
            topk = 3
            
            if source == 'WIKI':
                result_dict = self.wiki_service.get_documents(query, topk=topk)
                target_key = 'WIKI'
            elif source == 'PMC':
                result_dict = self.pmc_service.get_documents(query, topk=topk)
                target_key = 'PMC'
            elif source == 'BOOK':
                result_dict = self.book_service.get_documents(query, topk=topk)
                target_key = 'TEXTBOOK'
            
            # 解析返回的字典结构 {query: {SOURCE: content}}
            if result_dict and query in result_dict:
                inner_dict = result_dict[query]
                if target_key in inner_dict:
                    return inner_dict[target_key]
            
            return ""
            
        except Exception as e:
            print(f"[Error] Search {source} for '{query}' failed: {e}")
            return ""

    def process_batch(self, 
                      query_strings_list: List[str], 
                      summarize: bool = False) -> List[Dict[str, str]]:
        """
        主处理函数
        Args:
            query_strings_list: 包含多个 sample 字符串的列表
            summarize: 是否启用 LLM 总结
            
        Returns:
            List[Dict]: 列表长度与输入一致，每个元素是形如 {"Q1: query": "result", ...} 的字典
        """
        
        # 最终结果容器，预先占位
        final_results = [{} for _ in range(len(query_strings_list))]
        
        # SGLang 批处理队列
        sgl_inputs = []
        # 记录映射关系: index -> (sample_idx, result_key)
        sgl_map_info = [] 

        # --- 第一阶段：串行处理所有搜索请求 ---
        for s_idx, sample_str in enumerate(query_strings_list):
            
            # 1. 解析当前字符串
            queries = self._parse_single_query_string(sample_str)
            
            # 2. 遍历解析出的每一个 (Source, Query) 对
            for q_idx, (source, q_text) in enumerate(queries):
                
                # 构造返回结果的 Key，如 "Q1: historical figure..."
                result_key = f"Q{q_idx+1}: {q_text}"
                
                # 3. 执行搜索 (Blocking)
                search_content = self._fetch_search_result(source, q_text)
                
                # --- 分支处理 ---
                if not summarize:
                    # 模式 A: 截断 (Truncate)
                    if not search_content:
                        final_val = ""
                    else:
                        # 取前 512 字符
                        final_val = search_content[:512]
                    
                    final_results[s_idx][result_key] = final_val
                    
                else:
                    # 模式 B: 总结 (Summarize) - 先收集，后处理
                    # 即使内容为空，也传给 LLM 让其用自身知识回答
                    content_for_llm = search_content if search_content else "No external documents found."
                    
                    sgl_inputs.append({
                        "query_text": q_text,
                        "search_content": content_for_llm
                    })
                    # 记录这个请求属于哪个 Sample 的哪个 Key
                    sgl_map_info.append((s_idx, result_key))

        # --- 第二阶段：批处理总结 (如果需要) ---
        if summarize and sgl_inputs:
            # 调用 SGLang (内部并发)
            # print(f"Sending {len(sgl_inputs)} queries to SGLang...")
            
            states = summarize_query.run_batch(
                sgl_inputs,
                progress_bar=False, # 可以根据需要开启
                num_threads=16      # 用于 HTTP 请求的并发度
            )
            
            # 将结果填回 final_results
            for i, state in enumerate(states):
                s_idx, r_key = sgl_map_info[i]
                response_text = state['response'].strip()
                final_results[s_idx][r_key] = response_text

        return final_results

# ==========================================
# 3. 测试入口
# ==========================================

if __name__ == "__main__":
    # 配置
    manager = KnowledgeSearchService(
        wiki_port=8001, 
        pmc_port=8000, 
        book_port=8002,
        sgl_port=31680
    )

    # 构造 batch 输入列表
    test_batch = [
        # Sample 0: 混合源，带句号
        "|WIKI| historical figure with disease1. |PMC| relations between fever and cough.",
        # Sample 1: 单个源，无句号
        "|BOOK| definition of heart failure",
        # Sample 2: 多个相同源
        "|WIKI| Python programming language. |WIKI| Rust programming language"
    ]

    print("=== Mode 1: Truncate (No Summarize) ===")
    results_raw = manager.process_batch(test_batch, summarize=False)
    print(json.dumps(results_raw, indent=2, ensure_ascii=False))

    print("\n=== Mode 2: Summarize (With SGLang) ===")
    # 确保本地 31680 端口开启了 sglang server
    try:
        results_sum = manager.process_batch(test_batch, summarize=True)
        print(json.dumps(results_sum, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Summarize test failed (check SGLang server): {e}")