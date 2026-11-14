# Task 2 Brief: Decouple EAGLE-3.1 Speculation Flags from Adaptive Scheduler in `local_runner.py`

**Mục tiêu**: Loại bỏ cờ `--scheduler-cls AdaptiveAsyncScheduler` và cấu hình token batch không tương thích khi sử dụng phương pháp đầu cơ `draft_method == "eagle3"`, đồng thời giữ nguyên custom scheduler cho phương pháp `draft_model` truyền thống.

## Các tệp liên quan:
- Modify: `src/flexsds/server/local_runner.py`
- Modify: `tests/test_local_runner.py`

## Chi tiết các bước thực hiện:
1. Cập nhật `tests/test_local_runner.py`:
   - Viết bài test `test_local_runner_build_command_eagle3_omits_incompatible_scheduler` với `draft_method="eagle3"`:
     - Assert `num_speculative_tokens_per_batch_size` KHÔNG có trong `--speculative-config`.
     - Assert `--scheduler-cls` KHÔNG có trong `cmd`.
     - Assert `--distributed-executor-backend` KHÔNG có trong `cmd`.
   - Viết bài test `test_local_runner_build_command_draft_model_includes_adaptive_scheduler` với `draft_method="draft_model"`:
     - Assert `num_speculative_tokens_per_batch_size` CÓ trong `--speculative-config`.
     - Assert `--scheduler-cls` CÓ trong `cmd`.
     - Assert `src.server.adaptive_scheduler.AdaptiveAsyncScheduler` CÓ trong `cmd`.
2. Cập nhật `src/flexsds/server/local_runner.py`:
   - Trong `build_command`:
     ```python
     is_eagle = (draft_method == "eagle3")

     speculative_config: Dict[str, Any] = {
         "method": draft_method,
         "model": draft_model,
         "num_speculative_tokens": 4,
         "draft_sample_method": "greedy",
     }
     if not is_eagle:
         speculative_config["num_speculative_tokens_per_batch_size"] = [[1, 32, 4]]

     if use_hetero:
         speculative_config["use_heterogeneous_vocab"] = True

     # Chỉ gắn scheduler tùy biến cho phương pháp draft_model truyền thống
     if serving_cfg.get("enable_adaptive", True) and not is_eagle:
         cmd.extend([
             "--speculative-config", json.dumps(speculative_config),
             "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
             "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
             "--per-request-spec-decode-metrics", "summary",
         ])
     else:
         cmd.extend([
             "--speculative-config", json.dumps(speculative_config),
             "--per-request-spec-decode-metrics", "summary",
         ])
     ```
   - Trong `setup_environment`:
     ```python
     resolved = Path(__file__).resolve()
     project_root = str(resolved.parents[3]) if len(resolved.parents) > 3 else "/root/project"
     ```
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_local_runner.py -v`
   Đảm bảo 100% test pass.
4. Commit thay đổi:
   ```bash
   git add src/flexsds/server/local_runner.py tests/test_local_runner.py
   git commit -m "fix(local_runner): decouple eagle3 speculation from adaptive scheduler"
   ```
5. Báo cáo lại kết quả thực thi và diff.
