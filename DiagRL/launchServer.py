from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process


server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 36053 --tp 4 --mem-fraction-static 0.08"
)

print(port)
# port = 8002

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")