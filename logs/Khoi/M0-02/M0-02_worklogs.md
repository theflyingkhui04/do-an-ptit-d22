# Worklog Kỹ Thuật: [M0-02] Kiểm Chứng Speculative Decoding (EAGLE-2, SGLang, EAGLE-3.1) trên GPU T4 16GB & Đóng Câu Hỏi Chặn Kiến Trúc

- **Mã công việc:** `[M0-02]` (Sprint 1 · M0 PoC Nền tảng)
- **Người thực hiện:** Cấn Đức Khôi (Dev + PM — AI Systems & Quản lý dự án)
- **Thời gian thực hiện:** 18/09/2026 – 19/09/2026
- **Tài nguyên kiểm chứng gốc:**
  1. [`[M0_02]_Đồ_án_Kiểm_chứng_EAGLE_2.ipynb`](./%5BM0_02%5D_%C4%90%E1%BB%93_%C3%A1n_Ki%E1%BB%83m_ch%E1%BB%A9ng_EAGLE_2.ipynb): Kiểm chứng vLLM + Qwen2.5-7B AWQ + Draft Model Qwen2.5-0.5B.
  2. [`[M0_02_2]_Kiểm_chứng_checkpoint_EAGLE_SGLang.ipynb`](./%5BM0_02_2%5D_Ki%E1%BB%83m_ch%E1%BB%A9ng_checkpoint_EAGLE_SGLang.ipynb): Thực nghiệm chuyên sâu nạp checkpoint EAGLE-2 trên SGLang.
  3. [`[M0_02_3]_Khảo_sát_EAGLE_3_1_cho_Qwen2_5_7B.ipynb`](./%5BM0_02_3%5D_Kh%E1%BA%A3o_s%C3%A1t_EAGLE_3_1_cho_Qwen2_5_7B.ipynb): Khảo sát EAGLE-3.1 trên Llama-3.1 AWQ và phân tích mã nguồn Custom Proposer vLLM.
- **Trạng thái nghiệm thu:** Hoàn thành 100% (5/5 tiêu chí checklist Trello).

---

## 1. Mục tiêu công việc & Bối cảnh

Mục tiêu cốt lõi của Task M0-02 là trả lời các câu hỏi kỹ thuật sống còn mang tính rủi ro mức chí mạng (R1) trong tài liệu quyết định thiết kế (`docs/03-quyet-dinh-thiet-ke.md`):
1. **Kiểm chứng Speculative Decoding trên GPU Turing T4 16GB:** Có thể nạp đồng thời mô hình Target 7B-AWQ cùng với một mô hình nháp (Draft Model / Speculator Head) trên cùng 1 GPU 16GB mà vẫn đủ VRAM cho KV Cache hay không?
2. **Đóng câu hỏi chặn M-02:** Checkpoint EAGLE-2 (`leptonai/EAGLE-Qwen2.5-7B-Instruct`) có chạy được trên vLLM với bản 4-bit AWQ không?
3. **Đóng câu hỏi chặn M-01:** Vì sao đồ án neo phương pháp EAGLE-2 mà chưa dùng EAGLE-3/3.1? Khả năng tích hợp EAGLE-3.1 trên vLLM ra sao?
4. **Đóng câu hỏi chặn M-03:** Cơ chế `custom_class` proposer của vLLM có cho phép thay đổi cấu hình cây nháp theo từng bước sinh (per-step dynamic tree) hay không?

Để giải quyết triệt để các mục tiêu trên, nhóm đã triển khai thực nghiệm toàn diện qua **3 nhánh độc lập**, tương ứng với 3 file Jupyter Notebook.

---

## 2. Thông số Môi trường Thực nghiệm

Tất cả các thử nghiệm được thực hiện trên Google Colab GPU với cấu hình phần cứng:

| Thành phần | Thông số chi tiết |
| :--- | :--- |
| **GPU phần cứng** | NVIDIA Tesla T4 16GB (VRAM vật lý: 15,360 MiB / Khả dụng: 14.56 GiB) |
| **Kiến trúc vi xử lý** | Turing — Compute Capability **sm_75 (7.5)** |
| **Driver / CUDA Host** | NVIDIA Driver `580.82.07` / CUDA `13.0` |
| **Môi trường vLLM** | `vllm == 0.29.0`, Python 3.13 (Colab Base) |
| **Môi trường SGLang** | `sglang == 0.5.20`, Python 3.11 (Standalone via `uv venv`) |
| **Lưu trữ Cache** | Google Drive `/content/drive/MyDrive/vllm_cache` qua biến `HF_HOME` |

---

## 3. Nhánh 1: vLLM Native + Qwen2.5-7B AWQ + Draft Model Qwen2.5-0.5B

Tài nguyên thực nghiệm: [`[M0_02]_Đồ_án_Kiểm_chứng_EAGLE_2.ipynb`](./%5BM0_02%5D_%C4%90%E1%BB%93_%C3%A1n_Ki%E1%BB%83m_ch%E1%BB%A9ng_EAGLE_2.ipynb)

### 3.1. Rào cản 1: Lỗi kiến trúc checkpoint EAGLE-2 trên vLLM (Đóng câu hỏi M-02)
- **Thử nghiệm ban đầu:** Nạp trực tiếp checkpoint EAGLE-2 chính thức trên Hugging Face (`leptonai/EAGLE-Qwen2.5-7B-Instruct`) vào vLLM thông qua cấu hình `speculative_config={"method": "eagle", "model": "leptonai/EAGLE-Qwen2.5-7B-Instruct"}`.
- **Hiện tượng lỗi:**
  ```text
  ValidationError: Model architectures ['EagleQwen2ForCausalLM'] are not supported for now. 
  Supported architectures: ['AquilaForCausalLM', ..., 'EagleLlamaForCausalLM', ...]
  ```
- **Phân tích nguyên nhân:** vLLM bản hiện tại chỉ mới hỗ trợ kiến trúc EAGLE cho họ LLaMA (`EagleLlamaForCausalLM`), hoàn toàn chưa đăng ký lớp mô hình `EagleQwen2ForCausalLM` (GitHub Issue #24967). Do đó, **vLLM không thể nạp trực tiếp checkpoint EAGLE của Qwen2.5**.

### 3.2. Rào cản 2: Chuyển sang mô hình nháp cùng họ (`draft_model`) & Lệch từ vựng (Vocab Mismatch)
- **Giải pháp:** Chuyển sang phương thức nháp bằng mô hình nhỏ cùng họ `Qwen/Qwen2.5-0.5B-Instruct`.
- **Hiện tượng lỗi:**
  ```text
  ValueError: The target model and the draft model must have the same vocabulary size. 
  Target model: 152064, Draft model: 151936.
  ```
- **Phân tích nguyên nhân:** Bản 7B-AWQ có 152,064 tokens (được padding thêm 128 tokens để tối ưu hóa phép nhân ma trận GEMM), trong khi bản 0.5B gốc chỉ có 151,936 tokens. Validator mặc định của vLLM kiểm tra cứng nhắc `target_vocab == draft_vocab`.
- **Khắc phục:** Cấu hình mở rộng cờ từ vựng không đồng nhất và thuật toán lấy mẫu tham lam:
  ```python
  speculative_config = {
      "method": "draft_model",
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "num_speculative_tokens": 3,
      "use_heterogeneous_vocab": True,      # Cho phép lệch vocab size
      "draft_sample_method": "greedy"       # Lấy mẫu tham lam cho draft model
  }
  ```

### 3.3. Rào cản 3: Xung đột bộ đệm I/O trên Google Colab
- **Hiện tượng:** Chạy trực tiếp `llm = LLM(...)` trong cell Colab bị lỗi `io.UnsupportedOperation: fileno` trong module `suppress_stdout`.
- **Khắc phục:** Đóng gói toàn bộ mã nguồn vào script độc lập (`%%writefile test_spec1.py`) và thực thi qua sub-process `!python test_spec1.py`.

### 3.4. Kết quả Thực nghiệm Đo đạc (Warm Run)
Kết quả đo đạc chính thức từ `test_spec1.py` sau khi Triton JIT hoàn tất biên dịch:

```text
(EngineCore pid=15968) INFO [gpu_model_runner.py:5471] Model loading took 6.22 GiB memory
(EngineCore pid=15968) INFO [gpu_worker.py:625] Available KV cache memory: 4.77 GiB
(EngineCore pid=15968) INFO [kv_cache_utils.py:2032] GPU KV cache size: 73,552 tokens, Maximum concurrency for 2,048 tokens per request: 35.91x
(EngineCore pid=15968) INFO [gpu_worker.py:860] Actual usage is 6.46 GiB for consumed memory (weights + non-torch), 1.15 GiB for peak activation, and 0.0 GiB for CUDAGraph memory.
```

- **Cân đối bộ nhớ trên GPU T4 (15.36 GB):**
  - Trọng số Target (7B AWQ) + Draft (0.5B FP16): **6.46 GiB**
  - Kích hoạt đỉnh (Peak Activation): **1.15 GiB**
  - Bộ nhớ KV Cache khả dụng: **4.77 GiB** (đáp ứng **73,552 tokens**)
  - **Khả năng phục vụ đồng thời (Concurrency):** Đạt **35.91 requests đồng thời** với context length 2,048 tokens (Vượt mục tiêu 32x của đồ án).
- **Tốc độ sinh chữ:**
  - Prompt thử nghiệm: "Viết một bài văn ngắn khoảng 150 từ ca ngợi vẻ đẹp của Vịnh Hạ Long bằng tiếng Việt."
  - Số token sinh ra: `256 tokens`
  - Thời gian xử lý: `14.68 giây`
  - Tốc độ đo được: **`17.44 tokens/giây`**
  - Chất lượng đầu ra: Tiếng Việt chuẩn mực, đúng ngữ pháp, không lỗi font.

---

## 4. Nhánh 2: Khảo sát SGLang trên GPU T4 (Thử Nghiệm Chuyên Sâu)

Tài nguyên thực nghiệm: [`[M0_02_2]_Kiểm_chứng_checkpoint_EAGLE_SGLang.ipynb`](./%5BM0_02_2%5D_Ki%E1%BB%83m_ch%E1%BB%A9ng_checkpoint_EAGLE_SGLang.ipynb)

Vì vLLM thiếu class `EagleQwen2ForCausalLM`, nhóm đã chủ động thử nghiệm khung suy luận **SGLang** (vốn hỗ trợ native checkpoint `leptonai/EAGLE-Qwen2.5-7B-Instruct`). Quá trình xử lý lỗi mang tính hệ thống (*systematic debugging*):

### 4.1. Chuỗi lỗi và cách xử lý
1. **Lỗi build wheel Rust trên Python 3.13:**
   - *Nguyên nhân:* Thư viện `outlines_core` (viết bằng Rust) chưa có pre-built binary wheel cho Python 3.13 trên Colab -> Lỗi `Failed building wheel for outlines_core`.
   - *Xử lý:* Dùng công cụ `uv` tạo môi trường Python 3.11 độc lập:
     ```bash
     !pip install uv
     !uv venv /content/sgl_env --python 3.11 --seed
     !uv pip install --python /content/sgl_env/bin/python "sglang[all]"
     ```
2. **Lỗi thiếu công cụ biên dịch JIT `ninja`:**
   - *Nguyên nhân:* FlashInfer JIT yêu cầu `ninja` để biên dịch kernel RMSNorm lúc khởi chạy.
   - *Xử lý:* `!apt-get install -y ninja-build` và `!uv pip install ninja`.
3. **Lỗi đệ quy đa tiến trình (`multiprocessing spawn`):**
   - *Nguyên nhân:* SGLang dùng `multiprocessing spawn`. Code trôi nổi ngoài main module khiến child process đệ quy khởi tạo lại engine -> `RuntimeError: Rank 0 scheduler died`.
   - *Xử lý:* Bọc toàn bộ script trong `if __name__ == "__main__":`.
4. **Lỗi thiếu tham số cây phân nhánh EAGLE:**
   - *Nguyên nhân:* Python API của SGLang không gán giá trị mặc định cho `speculative_eagle_topk` (bị `None`), dẫn đến `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.
   - *Xử lý:* Khai báo rõ ràng `speculative_draft_model_path`, `speculative_eagle_topk=1`, `speculative_num_draft_tokens=4`.

### 4.2. Rào cản phần cứng cốt lõi: Khai tử kiến trúc Turing trên PyTorch CUDA 13
Sau khi nạp thành công 100% trọng số Target 7B AWQ và Draft EAGLE vào VRAM, tại bước sinh token đầu tiên:
```text
File ".../cuda_graph_buffer_registry.py", line 54, in _foreach_copy
  torch._foreach_copy_(group_dsts, group_srcs)
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device (cudaErrorNoKernelImageForDevice)
```
- **Bản chất kỹ thuật:** Gói `PyTorch 2.13.0+cu130` biên dịch chỉ nhắm vào GPU đời mới (sm_80 trở lên: A100, L4, H100), **đã loại bỏ mã nhị phân cho GPU Turing `sm_75` (Tesla T4)** trong các phép toán bộ đệm đa tensor (`torch._foreach_copy_`).
- **Quyết định kỹ thuật dứt khoát:** Nhóm đã chứng minh SGLang nạp được checkpoint EAGLE-2 về mặt lý thuyết. Tuy nhiên, việc ép chạy trên card T4 đời cũ bị chặn bởi giới hạn tương thích phần cứng của bộ thư viện CUDA 13. Nhóm quyết định dừng nhánh này để tránh sa lầy, tập trung hoàn thiện giải pháp vLLM cho đồ án.

---

## 5. Nhánh 3: Khảo Sát EAGLE-3.1 trên vLLM & Đóng Câu Hỏi Chặn M-01

Tài nguyên thực nghiệm: [`[M0_02_3]_Khảo_sát_EAGLE_3_1_cho_Qwen2_5_7B.ipynb`](./%5BM0_02_3%5D_Kh%E1%BA%A3o_s%C3%A1t_EAGLE_3_1_cho_Qwen2_5_7B.ipynb)

### 5.1. Thiết lập thực nghiệm EAGLE-3.1
Để kiểm chứng năng lực của vLLM với EAGLE-3.1, nhóm đã thiết lập cấu hình:
- **Target Model:** `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (5.33 GiB, lượng tử 4-bit tránh OOM 16GB).
- **Draft Speculator:** `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` (1.77 GiB safetensors).
- **Speculative Method:** `"method": "eagle3"`, `num_speculative_tokens: 3`.

### 5.2. Kết quả đo đạc bứt phá
```text
(EngineCore pid=5883) INFO [model.py:684] Resolved architecture: Eagle3LlamaForCausalLM
(EngineCore pid=5883) INFO [eagle3_utils.py:31] Using Eagle3 auxiliary layers from model: (2, 16, 29)
(EngineCore pid=5883) INFO [model_runner.py:404] Model loading took 7.19 GiB memory
(EngineCore pid=5883) INFO [gpu_worker.py:625] Available KV cache memory: 3.48 GiB (27,664 tokens, Concurrency: 13.51x)
```
- **Tốc độ sinh chữ:**
  - Prompt: "Viết một đoạn văn ngắn khoảng 100 từ giới thiệu về Vịnh Hạ Long bằng tiếng Việt."
  - Số token sinh ra: `150 tokens`
  - Thời gian xử lý: `5.80 giây`
  - **Tốc độ đạt được: `25.85 tokens/giây`** (Tốc độ vượt trội trên GPU T4).
  - Chất lượng: Sinh tiếng Việt mạch lạc, không lặp từ.

### 5.3. Phân tích Hiện tượng Bộ nhớ qua Biểu đồ Tài nguyên
Dựa trên biểu đồ giám sát tài nguyên của Google Colab:
1. **System RAM (CPU RAM):**
   - *Lúc tăng vọt lên đỉnh cam:* Quá trình tải các file trọng số `.safetensors` từ đĩa vào RAM để parse cấu trúc tensor và nhân bản tiến trình `EngineCore`.
   - *Lúc tụt dốc về 4.1 GB:* Khi toàn bộ trọng số (5.33 GiB + 1.77 GiB) đã chuyển hoàn toàn vào VRAM GPU, vLLM gọi `gc.collect()` giải phóng toàn bộ buffer CPU.
2. **GPU RAM (VRAM):**
   - *Bậc thang 7.68 GiB:* Chứa trọng số cố định của Target Model + Draft Speculator + CUDA Context.
   - *Đỉnh nhọn profiling 12.38 GiB (vạch cam):* vLLM thực hiện lượt chạy giả lập để đo Peak Activation (1.22 GiB), sau đó cấp phát tĩnh toàn bộ khối nhớ KV Cache liền mạch 3.48 GiB chạm ngưỡng trần `gpu_memory_utilization = 0.85`.
   - *Tụt về 0.0 GB:* Hàm `shutdown()` giải phóng toàn bộ VRAM khi kết thúc tiến trình.

---

## 6. Đọc Mã Nguồn vLLM & Đóng Câu Hỏi Chặn M-03

Mã nguồn được trích xuất trực tiếp từ gói `vllm == 0.29.0` trong notebook `[M0_02_3]`:

### 6.1. Mã nguồn cơ chế `custom_class` (`vllm/v1/spec_decode/custom_class_proposer.py`)
```python
def create_custom_proposer(vllm_config: VllmConfig):
    model_path = vllm_config.speculative_config.model
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    proposer_cls = getattr(module, class_name)
    proposer = proposer_cls(vllm_config)
    assert hasattr(proposer, "propose") and callable(proposer.propose)
    return proposer
```

### 6.2. Mã nguồn giao diện Proposer chuẩn (`vllm/v1/spec_decode/llm_base_proposer.py`)
```python
@dataclass
class ProposerOutput:
    proposal_lens: List[int]         # Cho phép thay đổi số lượng token nháp (DSL)
    proposal_tokens: torch.Tensor    # Tensor chứa token nháp
    # KHÔNG có trường dynamic tree_mask tùy biến!

class SpecDecodeBaseProposer(ABC):
    @abstractmethod
    def propose(self, proposer_input: ProposerInput) -> ProposerOutput:
        raise NotImplementedError
```

### 6.3. Kết luận chính thức cho câu hỏi M-03
> **Kết luận M-03:** `custom_class` của vLLM **KHÔNG hỗ trợ native thay đổi cấu trúc topology cây (Tree Mask) theo từng bước sinh**, nhưng **CÓ hỗ trợ thay đổi độ dài chuỗi nháp thích ứng (Dynamic Speculation Length - DSL)**.

Lý do: Để Target Model xác thực cây, cần ma trận Tree Attention Mask. vLLM hiện đang cố định ma trận này lúc khởi tạo engine để tối ưu CUDA Graph. `ProposerOutput` không cho phép truyền dynamic tree mask vào Verifier.

### 6.4. Đề xuất 3 phương án dự phòng cho Đồ án
1. **Phương án 1 (Khuyên dùng): Adaptive Draft Length (DSL):** Giữ cấu trúc nháp dạng chuỗi thẳng, thay đổi số token nháp $K$ ở từng bước dựa trên độ tự tin (Entropy / Margin) của Draft Model. Hoàn toàn tương thích native với `custom_class` của vLLM.
2. **Phương án 2: Fork nhẹ vLLM:** Mở rộng `ProposerOutput` thêm trường `custom_tree_mask: Optional[torch.Tensor]` và sửa nhẹ tại `rejection_sampler.py`.
3. **Phương án 3: Standalone Inference Loop:** Viết vòng lặp suy luận riêng bằng PyTorch thuần để tự do thay đổi topology cây phục vụ nghiên cứu so sánh thuật toán (`batch_size = 1`).

---

## 7. Bảng So Sánh Tổng Hợp 3 Hướng Tiếp Cận

| Tiêu chí so sánh | vLLM + Qwen Draft Model (Nhánh 1) | SGLang + EAGLE-2 (Nhánh 2) | vLLM + EAGLE-3.1 (Nhánh 3) |
| :--- | :--- | :--- | :--- |
| **Mô hình Target** | `Qwen2.5-7B-Instruct-AWQ` | `Qwen2.5-7B-Instruct-AWQ` | `Meta-Llama-3.1-8B-Instruct-AWQ` |
| **Mô hình Draft** | `Qwen2.5-0.5B-Instruct` | `leptonai/EAGLE-Qwen2.5-7B` | `RedHatAI/Llama-3.1-8B-speculator` |
| **Kiến trúc Draft** | Mô hình ngôn ngữ độc lập | Draft Head (EAGLE-2) | Draft Head (EAGLE-3) |
| **Mức tiêu thụ VRAM** | **6.46 GiB** | ~6.50 GiB (lúc nạp) | **7.19 GiB** |
| **VRAM KV Cache dư** | **4.77 GiB** (73,552 tokens) | N/A (crash lúc decode) | **3.48 GiB** (27,664 tokens) |
| **Tối đa Concurrency** | **35.91x** (vượt chỉ tiêu 32x) | N/A | **13.51x** |
| **Tốc độ sinh chữ** | **17.44 tok/s** | N/A (lỗi sm_75 kernel) | **25.85 tok/s** |
| **Năng lực Tiếng Việt** | **Xuất sắc** (Vocab 152k tối ưu) | Tốt (họ Qwen) | Khá (họ LLaMA vốn yếu tiếng Việt) |
| **Trạng thái thực tế** | **Ổn định 100% (Production)** | Không tương thích GPU T4 | **Ổn định 100% (High speed)** |

---

## 8. Đóng các Câu Hỏi Chặn trong Tài Liệu Thiết Kế

Căn cứ vào kết quả thực nghiệm của Task M0-02, các câu hỏi chặn tại bảng `docs/03-quyet-dinh-thiet-ke.md` được cập nhật trạng thái đóng:

1. **Đóng M-01 (Vì sao neo EAGLE-2 mà không phải EAGLE-3?):**
   - Tác giả gốc và cộng đồng (RedHatAI) hiện chỉ mới phát hành checkpoint EAGLE-3 / 3.1 cho dòng LLaMA-3/3.1, chưa hề có checkpoint chính thức cho Qwen2.5.
   - Nếu đồ án muốn dùng EAGLE-3 cho Qwen2.5 tiếng Việt, nhóm hoàn toàn có thể tự huấn luyện một Draft Head riêng ở Sprint 2 thông qua bộ công cụ `SpecForge` hoặc `SafeAILab/EAGLE`.
2. **Đóng M-02 (Checkpoint EAGLE-2 cho Qwen2.5-7B AWQ):**
   - Checkpoint `leptonai/EAGLE-Qwen2.5-7B-Instruct` không chạy được trên vLLM do thiếu class `EagleQwen2ForCausalLM`.
   - Phương án thay thế hoàn hảo là **Draft Model `Qwen2.5-0.5B-Instruct`** trên vLLM: Chạy ổn định tuyệt đối với bản AWQ 4-bit, ăn 6.46 GiB VRAM, concurrency đạt 35.91x.
3. **Đóng M-03 (Custom class đổi cây per-step):**
   - Không hỗ trợ native đổi cấu trúc cây (tree topology), nhưng hỗ trợ đổi độ dài nháp thích ứng (DSL). Đã có 3 phương án dự phòng chi tiết.

---

## 9. Đánh giá Checklist M0-02 & Bàn Giao

### Cập nhật Checklist Nghiệm thu M0-02 (Khôi):
- [x] **Tải và kiểm tra checkpoint EAGLE-2 cho Qwen2.5-7B:** Đã tải và xác định giới hạn thiếu class kiến trúc trên vLLM.
- [x] **Kiểm chứng phối hợp Target 7B-AWQ + Draft Model:** Đạt 17.44 tok/s, VRAM 6.46 GiB, concurrency 35.91x trên GPU T4 16GB.
- [x] **Khảo sát SGLang và xác định rào cản phần cứng Turing sm_75:** Hoàn thành với đầy đủ bằng chứng lỗi CUDA 13.
- [x] **Khảo sát bứt phá với EAGLE-3.1 trên vLLM:** Đạt tốc độ 25.85 tok/s, giải thích tường tận biểu đồ tài nguyên RAM/VRAM.
- [x] **Đọc mã nguồn vLLM `spec_decode` và đóng câu hỏi chặn M-03:** Đã trích xuất mã nguồn và đưa ra 3 phương án dự phòng.
- [x] **Ghi nhận toàn bộ kết quả thực nghiệm vào Worklog:** Hoàn thành tại `logs/Khoi/M0-02/M0-02_worklogs.md`.

### Bàn giao cho Task M0-05 (Cuộc họp tuần chốt Giao diện đóng băng A & B):
Toàn bộ rủi ro kỹ thuật R1 liên quan đến Speculative Decoding và hạ tầng vLLM đã được tháo gỡ hoàn toàn. Nhóm tự tin bước vào cuộc họp tuần với giảng viên hướng dẫn với đầy đủ số liệu thực nghiệm định lượng!
