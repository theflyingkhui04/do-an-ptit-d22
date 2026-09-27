import time
import json
import concurrent.futures
from urllib import request, error

# ==============================================================================
# CẤU HÌNH THỬ THÁCH CỰC HẠN (EXTREME STRESS TEST)
# ==============================================================================
URL = "https://tstrtz2l-8001.usw2.devtunnels.ms/v1/chat/completions"
MODEL = "qwen2.5-7b-vanilla"

# Bắn đồng thời 32 luồng (chạm trần max-num-seqs của server vLLM trên Colab T4)
CONCURRENCY = 32
MAX_TOKENS = 384

PROMPT = (
    "Hãy viết một bài luận phân tích sâu sắc về cơ chế PagedAttention và Speculative Decoding "
    "trong việc tối ưu hóa thông lượng phục vụ mô hình ngôn ngữ lớn (LLM). Phân tích chi tiết "
    "về hiện tượng nghẽn băng thông bộ nhớ GPU (Memory Bandwidth Bottleneck) và cách khắc phục."
)

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer none"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Bạn là chuyên gia kiến trúc máy tính và tối ưu hóa hệ thống AI."},
        {"role": "user", "content": PROMPT}
    ],
    "temperature": 0.7,
    "max_tokens": MAX_TOKENS
}

data_bytes = json.dumps(payload).encode("utf-8")

def worker(worker_id):
    t_start = time.time()
    req = request.Request(URL, data=data_bytes, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=180) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            elapsed = time.time() - t_start
            
            if status == 200:
                res_json = json.loads(body)
                tokens = res_json.get("usage", {}).get("completion_tokens", 0)
                tok_per_s = tokens / elapsed if elapsed > 0 else 0
                print(f" [Worker {worker_id:02d}] THÀNH CÔNG: {elapsed:.2f}s | Sinh: {tokens} tok | Tốc độ: {tok_per_s:.1f} tok/s")
                return {"status": "success", "time": elapsed, "tokens": tokens}
            else:
                print(f"❌ [Worker {worker_id:02d}] LỖI HTTP {status}")
                return {"status": "http_error", "code": status}
    except error.HTTPError as e:
        elapsed = time.time() - t_start
        print(f"❌ [Worker {worker_id:02d}] LỖI HTTP {e.code} sau {elapsed:.2f}s: {e.reason}")
        return {"status": "http_error", "code": e.code, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"💥 [Worker {worker_id:02d}] THẤT BẠI sau {elapsed:.2f}s: {e}")
        return {"status": "error", "error": str(e), "time": elapsed}

def run_stress_test():
    print("=" * 70)
    print("      🚀 BẮT ĐẦU THỬ THÁCH CỰC HẠN (EXTREME STRESS TEST) 🚀")
    print(f"Target Endpoint : {URL}")
    print(f"Target Model    : {MODEL}")
    print(f"Mức Concurrency : {CONCURRENCY} requests đồng thời cùng lúc")
    print(f"Max Tokens/req  : {MAX_TOKENS} tokens")
    print(f"Kỳ vọng sinh    : ~{CONCURRENCY * MAX_TOKENS} tokens cùng lúc trên GPU T4 (15GB VRAM)")
    print("=" * 70)
    print("Đang khai hỏa toàn bộ 32 luồng song song qua Internet...\n")

    overall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(worker, i + 1) for i in range(CONCURRENCY)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    overall_time = time.time() - overall_start

    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") != "success"]
    total_tokens = sum(r.get("tokens", 0) for r in successes)
    total_throughput = total_tokens / overall_time if overall_time > 0 else 0

    times = [r["time"] for r in successes]
    avg_latency = sum(times) / len(times) if times else 0
    min_latency = min(times) if times else 0
    max_latency = max(times) if times else 0

    print("\n" + "=" * 70)
    print("               📊 BẢNG TỔNG KẾT KẾT QUẢ ĐO TẢI")
    print("=" * 70)
    print(f"Tổng thời gian toàn bộ bài test: {overall_time:.2f} giây")
    print(f"Thành công                     : {len(successes)} / {CONCURRENCY} requests")
    print(f"Thất bại / Lỗi / Sập           : {len(failures)} / {CONCURRENCY} requests")
    print(f"Tổng số token GPU đã sinh      : {total_tokens} tokens")
    print(f"🔥 THÔNG LƯỢNG TOÀN HỆ THỐNG   : {total_throughput:.2f} tokens / giây")
    print(f"Độ trễ trung bình mỗi request  : {avg_latency:.2f} giây")
    print(f"Độ trễ nhanh nhất / chậm nhất  : {min_latency:.2f}s / {max_latency:.2f}s")
    print("=" * 70)

    if len(successes) == CONCURRENCY:
        print("🏆 KẾT LUẬN: vLLM TRÊN TESLA T4 ĐÃ VƯỢT QUA BÀI TEST HOÀN HẢO! KHÔNG HỀ SẬP!")
    elif len(successes) > 0:
        print(f"⚠️ KẾT LUẬN: Chịu tải được {len(successes)}/{CONCURRENCY}, đã chạm ngưỡng nghẽn VRAM hoặc hàng đợi.")
    else:
        print("💥 KẾT LUẬN: MÁY CHỦ BỊ SẬP HOẶC KHÔNG THỂ PHẢN HỒI!")

if __name__ == "__main__":
    run_stress_test()
