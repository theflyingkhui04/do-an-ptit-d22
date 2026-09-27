import time
import requests

SERVER_URL = "http://localhost:8001/v1/chat/completions"

payload = {
    "model": "qwen2.5-7b-vanilla",
    "messages": [
        {"role": "system", "content": "Bạn là trợ lý AI thông minh và súc tích."},
        {"role": "user", "content": "Hãy giải thích ngắn gọn trong 1 câu: Giải mã suy đoán (Speculative Decoding) là gì?"}
    ],
    "temperature": 0.0,
    "max_tokens": 128
}

print(" Đang gửi request kiểm tra tới vLLM Làn 1 (cổng 8001)...")
start_time = time.time()

try:
    response = requests.post(SERVER_URL, json=payload, timeout=60)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]
        print(f"\nThành công! (Thời gian phản hồi: {latency:.2f}s | Sinh: {tokens} tokens)")
        print(f"Output:\n{content}")
    else:
        print(f"Lỗi HTTP {response.status_code}: {response.text}")
except Exception as e:
    print(f"Không thể kết nối tới server: {e}")
