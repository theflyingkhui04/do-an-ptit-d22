<div align="center">

<img src="logo/flexsds_banner.gif" alt="PTIT — FlexSDS" width="100%" />

# FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs

**Hệ thống phục vụ giải mã suy đoán EAGLE thích ứng theo phần cứng cho mô hình ngôn ngữ lớn**

Đồ án tốt nghiệp D22 · Học viện Công nghệ Bưu chính Viễn thông (PTIT)

</div>

## 1. Mục tiêu đề tài

FlexSDS nghiên cứu và xây dựng serving engine kết hợp **EAGLE speculative decoding** với bộ điều khiển **Hardware-Adaptive**. Bộ điều khiển sử dụng chi phí nháp/xác thực đo trên GPU, tỷ lệ chấp nhận token và tình trạng bộ nhớ để lựa chọn ngân sách speculative token `K` phù hợp khi phục vụ LLM.

Bài toán trọng tâm là giảm độ trễ decode và hạn chế negative speedup khi workload, batch hoặc phần cứng thay đổi. Các cấu hình vanilla và speculative decoding bằng draft model độc lập được dùng làm đối chứng. Ứng dụng hóa đơn tiếng Việt cung cấp workload để đánh giá engine.

**Hardware-Adaptive EAGLE là mục tiêu chính của sản phẩm.** CLI đã nối EAGLE vào đường scalar adaptive K của nhánh M4: controller → scheduler → proposer → verification feedback/lineage. Tích hợp đã qua regression CPU/source contract và functional GPU smoke test trên Modal A10G ở batch 1. So sánh lossless và hiệu năng end-to-end vẫn cần nghiệm thu riêng.

## 2. Trạng thái triển khai

| Thành phần | Trạng thái và giới hạn |
|---|---|
| EAGLE serving qua CLI | Endpoint Modal adaptive đã trả health/models/completion và ba request 128 tokens ở max_num_seqs=1. Local GPU và full capacity cần nghiệm thu riêng. |
| Hardware-Adaptive EAGLE | Đã nối scalar dynamic K, async scheduler và paired lineage executor theo nhánh M4. Yêu cầu profile measured v2 và source vLLM 0.30.0 đúng hash; đã quan sát K=0/2/3 và feedback có lineage trên GPU ở batch 1. |
| Calibration GPU | Đã đo thật K=0..4, batch 1, context=1024 trên Modal A10G: 275 raw records. Full grid tới batch 32 chưa được kiểm tra trong lượt này. |
| Hardware profile | Strict gate từ chối synthetic profile cũ; profile được truyền tới container và cost consumer của scalar scheduler. |
| Scalar scheduler | Dùng chung cơ chế dynamic K cho sequential EAGLE và nhánh nghiên cứu `draft_model`; lịch sử các lane đối chứng giữ nguyên. |
| Readiness | CLI kiểm tra `/health`, `/v1/models` và completion trước khi báo `Inference verified`. Tạo URL chưa chứng minh model sẵn sàng. |
| Tree-adaptive | Hoãn theo quyết định phạm vi M4; scalar K không chứng minh cây nháp đã được thực thi. |

Phần cần nghiệm thu tiếp theo là calibration, chuyển K và feedback thực trên GPU, kiểm tra lossless và đo hiệu năng theo workload. `K=0` ngừng sinh token nháp nhưng EAGLE vẫn forward để đồng bộ KV cache; không coi đó là tắt hẳn drafter hoặc bảo đảm nhanh bằng vanilla.

**Lossless** đề cập đến tính đúng đắn/phân phối đầu ra của speculative decoding. Nó không đồng nghĩa với bảo đảm speedup ≥1, tránh OOM tuyệt đối hoặc đạt một mức tăng tốc cố định trên mọi workload. Kết luận hiệu năng phải gắn với cấu hình và bằng chứng đo cụ thể.

Bằng chứng GPU và giới hạn phép thử: [verification summary](logs/problems/evidence/2026-11-15-modal-a10g/verification_summary.json).

Chi tiết: [problem log CLI/Modal](logs/problems/2026-11-14-flexsds-cli-modal-calibration-and-serving.md), [quyết định phạm vi scalar/tree](logs/problems/2026-11-12-m4-02-04-tree-adaptive-vllm-flattening-and-lineage-v02.md).

## 3. Cài đặt và chạy thử

```bash
git clone https://github.com/theflyingkhui04/do-an-ptit-d22.git
cd do-an-ptit-d22
pip install -e .
flexsds
```

Package CLI không tự cài GPU runtime. Local serving/calibration cần CUDA và vLLM 0.30.0 trong môi trường chạy. Modal cần Modal CLI và credentials đã cấu hình; image GPU được khai báo trong source.

### Chạy Hardware-Adaptive EAGLE trên Modal A10G

```bash
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
```

Đo profile v2 bằng lệnh bên dưới trước khi chạy; wizard có thể đề nghị calibration khi profile còn thiếu. Thêm `--no-adaptive` nếu muốn kiểm tra EAGLE static `K=4` mà không cần profile. Chờ `Inference verified` để xác nhận completion trả về. Development dùng `modal serve`: giữ terminal mở; Ctrl-C dừng app. Thêm `--deploy` để dùng persistent deployment.

### Đo hardware profile trên GPU Modal

```bash
flexsds profile --benchmark --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
```

Calibration cấp GPU thật, load model, chạy warmup và đo nhiều lần; có thể mất nhiều phút và sử dụng tài nguyên Modal. Kết quả lưu ở `~/.flexsds/profiles/`. Profile cũ tạo bằng công thức không được chấp nhận là empirical profile.

Profile v2 đo target forward + draft proposal ở eager mode, bao gồm drafter KV-cache sync tại `K=0`, chưa bao gồm toàn bộ overhead serving. Profile synthetic cũ và profile v1 cần được đo lại; CLI pin model revisions theo profile.

Hướng dẫn và giới hạn: [runbook CLI/Modal](docs/runbooks/flexsds-cli-modal-calibration.md).

## 4. Thiết kế đánh giá

Repo duy trì các lane đối chứng và công cụ thực nghiệm phục vụ quá trình nghiên cứu:

| Lane | Vai trò |
|---|---|
| `vanilla` | Target model giải mã tuần tự, làm baseline. |
| `static` | Draft model độc lập với ngân sách nháp cố định. |
| `adaptive` | Nhánh scalar adaptive để đánh giá quyết định ngân sách K. |
| `eagle` | EAGLE speculator để đo đường nháp chuyên biệt. |

Tên lane lịch sử được giữ nguyên. CLI adaptive EAGLE sử dụng đường tích hợp M4 `eagle_adaptive`; không đổi ý nghĩa lane `adaptive` dùng draft model độc lập. Kết quả phải ghi rõ target/draft revision, GPU, quantization, K, batch/concurrency, context, workload và phiên bản runtime.

Chỉ số đánh giá gồm TTFT, TPOT, throughput, acceptance rate, VRAM và speedup so với vanilla. Phân biệt timing tại worker với timing tại client; không diễn giải khoảng cách giữa các SSE chunk thành TPOT của engine.

## 5. Ứng dụng thực nghiệm: hóa đơn tiếng Việt

Pipeline hóa đơn cung cấp ba nhóm workload để đánh giá engine: trích xuất thông tin thành JSON (MC-OCR), sinh truy vấn SQL và hỏi đáp/phân tích nghiệp vụ. Đây là ứng dụng minh họa cho serving engine; acceptance rate và speedup được đo theo từng chiến dịch, không mặc định theo loại tác vụ.

## 6. Cấu trúc và tài liệu

```text
src/flexsds/       CLI, preflight, hardware profiles, calibration và serving runners
src/server/        Scalar scheduler, budget planner và lineage
src/proposer/      Controller và cost consumer
src/pipeline/      Ứng dụng hóa đơn
experiments/       Benchmark, profiler và bằng chứng thực nghiệm
docs/              Thiết kế, bản thảo báo cáo và runbooks
logs/problems/     Vấn đề kỹ thuật, quyết định phạm vi và trạng thái xác minh
tests/             Regression tests
```

- [Bản đồ tài liệu](docs/README.md)
- [Giới thiệu và câu hỏi nghiên cứu](docs/01-gioi-thieu.md)
- [Đặc tả mục tiêu sản phẩm](docs/09-dac-ta-san-pham-flexsds.md)
- [Công thức và chỉ số](logs/maths_expression_used.md)

Các tài liệu chiến dịch và nhật ký cũ được giữ theo bối cảnh tại thời điểm ghi nhận; trạng thái sản phẩm hiện hành nằm ở README và runbook.

## 7. Đội ngũ thực hiện

Nhóm áp dụng mô hình **Ghép cặp Luân phiên 70/30 (Pair-Rotation System)**: Mỗi thành viên chủ trì (Lead) 70% mảng chuyên môn cốt lõi, đồng thời luân phiên đồng hành (Partner) 30% ở các mảng còn lại:

| Chuyên môn cốt lõi (Lead 70%) | Thành viên thực hiện | Vai trò trong dự án |
| :--- | :--- | :--- |
| **AI Systems & Core-Engine** | **Cấn Đức Khôi** ([@theflyingkhui04](https://github.com/theflyingkhui04)) | Lead Architecture & Core vLLM Scheduler |
| **MLOps & Infrastructure** | **Bùi Đức Đại** | Lead Benchmark Harness & GPU Telemetry |
| **AI Application & Data** | **Dương Nguyễn Thế** | Lead Pipeline MC-OCR & Invoices Analytics |
| **Giảng viên hướng dẫn** | *(Cập nhật theo quyết định của khoa)* | Cố vấn khoa học & Giám sát chất lượng |

---

## 8. Bản quyền

Dự án được phát triển phục vụ đồ án tốt nghiệp đại học tại **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**, niên khóa 2022–2026. Mã nguồn được phát hành theo giấy phép mã nguồn mở Apache 2.0.
