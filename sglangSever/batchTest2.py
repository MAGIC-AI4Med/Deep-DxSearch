import sglang as sgl

# ... (你之前的启动代码保持不变) ...
port = 31680
# 1. 设置默认后端为你刚启动的 Server
# 注意：这里使用你代码中获取到的 port
backend = sgl.RuntimeEndpoint(f"http://localhost:{port}")
sgl.set_default_backend(backend)

# 2. 定义处理单个 Prompt 的 SGLang 函数
# 这里的参数名 'prompt_text' 可以自定义，稍后在 run_batch 中对应即可
@sgl.function
def batch_generate(s, prompt_text):
    s += prompt_text
    # 你可以在这里配置 max_tokens, stop words 等参数
    s += sgl.gen("response", max_tokens=128)

# 3. 准备你的 Prompt List
prompts = [
    "What is the capital of France?",
    "List 3 prime numbers.",
    "Write a haiku about coding.",
    "Explain quantum computing in one sentence."
]

# 4. 构造参数列表
# run_batch 接收一个字典列表，字典的 key 必须对应 @sgl.function 中的参数名
input_args = [{"prompt_text": p} for p in prompts]

print(f"正在并发发送 {len(input_args)} 个请求...")

# 5. 执行并发请求 (无显式多线程)
# run_batch 会自动管理并发，并将结果按顺序返回
states = batch_generate.run_batch(
    input_args, 
    progress_bar=True,  # 显示进度条
    num_threads=4      # *注意*：这是 SGLang 内部用于网络 IO 的并发度，不是你的业务逻辑线程
)

# 6. 处理结果
print("\n--- 结果 ---")
for i, state in enumerate(states):
    print(f"Prompt: {prompts[i]}")
    print(f"Output: {state['response']}\n")

# 7. 清理资源 (如果你需要关闭 server)
# server_process.terminate()