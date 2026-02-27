import requests
port = 35609
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": [
            "List 3 countries and their capitals.",
            "AI is a field of computer science focused on",
            "Please give me the detail of A star"
        ],
        # "sampling_params": {"max_new_tokens": 32, "temperature": 0}
    }
)
if response.status_code == 200:
    result = response.json()
    # 提取每个响应项中的'text'字段
    summaries = [item['text'] for item in result]
    print(summaries) 
else:
    print("Error")