import requests
import threading
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor
from sglang.utils import wait_for_server, print_highlight, terminate_process

def generate_complex_query():
    """生成较为复杂的查询内容"""
    topics = [
        "详细解释一下量子计算的基本原理，并给出几个实际应用场景。",
        "比较中国古代与西方文艺复兴时期的艺术特点与哲学思想异同。",
        "分析全球气候变化趋势，并讨论各国应对策略的有效性。",
        "请解释神经网络在深度学习中的应用，并讨论其局限性。",
        "评价数字货币对全球金融体系的潜在影响及其监管挑战。",
        "讨论人工智能发展对就业市场的影响，以及社会应如何应对。",
        "分析当代城市化进程中的环境与社会问题，并提出可持续发展建议。",
        "探讨太空探索的未来方向，并评估人类移民火星的可行性。",
        "比较不同教育体系的优缺点，并提出结合各国优势的教育改革方案。",
        "分析生物技术在医疗领域的突破性进展，并讨论相关伦理问题。"
    ]
    return random.choice(topics)

def make_request(thread_id, port=32890):
    """发送请求并记录响应时间"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    data = {
        "model": "/mnt/vision_user/huggingface/gpt-oss-120b",
        "messages": [{"role": "user", "content": generate_complex_query()}],
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            content_preview = content[:50] + "..." if len(content) > 50 else content
            
            result = {
                "thread_id": thread_id,
                "status": "success",
                "response_time": end_time - start_time,
                "content_length": len(content),
                "content_preview": content_preview
            }
        else:
            result = {
                "thread_id": thread_id,
                "status": "error",
                "response_time": end_time - start_time,
                "error": f"Status code: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        end_time = time.time()
        result = {
            "thread_id": thread_id,
            "status": "exception",
            "response_time": end_time - start_time,
            "error": str(e)
        }
    
    return result

def run_concurrent_test(num_threads=50, max_workers=20, port=32890):
    """运行并发测试"""
    print_highlight(f"开始并发测试，线程数：{num_threads}，最大并发数：{max_workers}")
    
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, i, port) for i in range(num_threads)]
        for future in futures:
            result = future.result()
            all_results.append(result)
            
            # 实时输出单个结果
            if result["status"] == "success":
                print(f"线程 {result['thread_id']} 成功，响应时间：{result['response_time']:.2f}秒，内容长度：{result['content_length']}")
            else:
                print(f"线程 {result['thread_id']} 失败，响应时间：{result['response_time']:.2f}秒，错误：{result.get('error', 'Unknown')}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 汇总结果
    successful_requests = sum(1 for r in all_results if r["status"] == "success")
    failed_requests = len(all_results) - successful_requests
    
    response_times = [r["response_time"] for r in all_results if r["status"] == "success"]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    print_highlight("\n测试结果汇总")
    print(f"总线程数: {num_threads}")
    print(f"成功请求数: {successful_requests}")
    print(f"失败请求数: {failed_requests}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均响应时间: {avg_response_time:.2f}秒")
    print(f"吞吐量: {successful_requests / total_time:.2f}请求/秒")
    
    # 保存详细结果到文件
    with open("/mnt/vision_user/zhengqiaoyu/DiagRL/sglangSever/concurrent_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_threads": num_threads,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_time": total_time,
                "avg_response_time": avg_response_time,
                "throughput": successful_requests / total_time if total_time > 0 else 0
            },
            "detailed_results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print("详细结果已保存到 concurrent_test_results.json")

if __name__ == "__main__":

    # 运行并发测试，可以调整参数
    # num_threads: 总共发送的请求数量
    # max_workers: 同时并发的最大线程数
    run_concurrent_test(num_threads=100, max_workers=30, port=31680)