import requests
import time
# 你的批量query列表
batch_query = [
    "Autoimmune hepatitis",
    "Genetic cystic renal disease",
    "Cystic hygroma"
]

SEARCH_URL = "http://0.0.0.0:8001/retrieve"
TOPK = 3

def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference

# 构造请求
payload = {
    "queries": batch_query,
    "topk": TOPK,
    "return_scores": True
}
start_time = time.time()
response = requests.post(SEARCH_URL, json=payload)
results = response.json()['result']
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# 输出每个query的检索结果
for i, (query, result) in enumerate(zip(batch_query, results)):
    print(f"\n================ Query {i+1} ================")
    print(f"Query: {query}")
    print("Search Results:")
    print(_passages2string(result))
