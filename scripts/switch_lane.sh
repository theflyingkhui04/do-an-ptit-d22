#!/usr/bin/env bash
set -e

# ==============================================================================
# SCRIPT ĐIỀU PHỐI CHUYỂN ĐỔI LÀN PHỤC VỤ (vLLM LANE SWITCHER)
# Đồ án tốt nghiệp D22 PTIT: Hardware-Aware Speculative Decoding
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

PID_FILE_LANE1="/tmp/vllm_lane1.pid"
PID_FILE_LANE2="/tmp/vllm_lane2.pid"

stop_all() {
    echo "==> Đang dừng tất cả tiến trình vLLM và giải phóng GPU VRAM..."
    
    # 1. Giải phóng theo cổng mạng
    fuser -k 8001/tcp 2>/dev/null || true
    fuser -k 8002/tcp 2>/dev/null || true

    # 2. Dừng theo tên tiến trình
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true

    # 3. Dọn dẹp file PID
    rm -f "$PID_FILE_LANE1" "$PID_FILE_LANE2"

    # Chờ GPU thu hồi bộ nhớ hoàn toàn
    sleep 3
    echo "    Đã dọn dẹp xong tiến trình cũ."
}

wait_for_health() {
    local port="$1"
    local timeout="$2"
    local log_file="$3"
    local elapsed=0
    local url="http://127.0.0.1:${port}/health"

    echo "==> Đang chờ server trên cổng ${port} sẵn sàng (${url})..."
    while [ $elapsed -lt $timeout ]; do
        local status
        status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

        if [ "$status" = "200" ]; then
            echo ""
            echo "    Server trên cổng ${port} đã sẵn sàng hoạt động (HTTP 200) sau ${elapsed}s!"
            return 0
        fi

        # Kiểm tra xem tiến trình vLLM có bị dừng đột ngột không
        if ! pgrep -f "vllm" > /dev/null; then
            echo ""
            echo "[LỖI] Tiến trình vLLM đã dừng đột ngột trước khi hoàn tất khởi động!"
            echo "--- 20 DÒNG LOG CUỐI CÙNG TỪ ${log_file} ---"
            tail -n 20 "$log_file" 2>/dev/null || true
            echo "-----------------------------------------------"
            return 1
        fi

        echo -n "."
        sleep 3
        elapsed=$((elapsed + 3))
    done

    echo ""
    echo "[LỖI] Quá thời gian chờ (${timeout}s). Server trên cổng ${port} chưa sẵn sàng!"
    echo "--- 20 DÒNG LOG CUỐI CÙNG TỪ ${log_file} ---"
    tail -n 20 "$log_file" 2>/dev/null || true
    echo "-----------------------------------------------"
    return 1
}

start_lane1() {
    stop_all
    local log_file="$LOG_DIR/vllm_lane1.log"
    echo "==> Đang khởi chạy Làn 1: Vanilla Baseline (Cổng 8001)..."
    echo "    Log được ghi vào: ${log_file}"

    chmod +x "$PROJECT_ROOT/scripts/launch_lane1_vanilla.sh"
    nohup "$PROJECT_ROOT/scripts/launch_lane1_vanilla.sh" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE_LANE1"
    echo "    Tiến trình chạy nền với PID: ${pid}"

    wait_for_health 8001 120 "$log_file"
}

start_lane2() {
    stop_all
    local log_file="$LOG_DIR/vllm_lane2.log"
    echo "==> Đang khởi chạy Làn 2: Static Draft Baseline (Cổng 8002)..."
    echo "    Log được ghi vào: ${log_file}"

    chmod +x "$PROJECT_ROOT/scripts/launch_lane2_static.sh"
    nohup "$PROJECT_ROOT/scripts/launch_lane2_static.sh" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE_LANE2"
    echo "    Tiến trình chạy nền với PID: ${pid}"

    wait_for_health 8002 180 "$log_file"
}

show_status() {
    echo "================ TRẠNG THÁI HỆ THỐNG PHỤC VỤ ================"
    local lane1_status lane2_status
    lane1_status=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health 2>/dev/null || echo "000")
    lane2_status=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8002/health 2>/dev/null || echo "000")

    if [ "$lane1_status" = "200" ]; then
        echo "Làn 1 (Vanilla - Cổng 8001): ĐANG CHẠY (ONLINE)"
    else
        echo "Làn 1 (Vanilla - Cổng 8001): TẮT (OFFLINE)"
    fi

    if [ "$lane2_status" = "200" ]; then
        echo "Làn 2 (Static Draft - Cổng 8002): ĐANG CHẠY (ONLINE)"
    else
        echo "Làn 2 (Static Draft - Cổng 8002): TẮT (OFFLINE)"
    fi

    echo "------------------------------------------------------------"
    if command -v nvidia-smi &> /dev/null; then
        echo "Thông tin GPU VRAM hiện tại:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    fi
    echo "============================================================"
}

case "$1" in
    lane1|1)
        start_lane1
        ;;
    lane2|2)
        start_lane2
        ;;
    stop)
        stop_all
        echo "Đã dừng toàn bộ dịch vụ vLLM."
        ;;
    status)
        show_status
        ;;
    *)
        echo "Cách sử dụng: $0 {lane1|lane2|stop|status}"
        echo "  - lane1 : Dừng server cũ, bật Làn 1 (Vanilla) trên cổng 8001 và chờ sẵn sàng."
        echo "  - lane2 : Dừng server cũ, bật Làn 2 (Static Draft) trên cổng 8002 và chờ sẵn sàng."
        echo "  - stop  : Dừng toàn bộ server vLLM và giải phóng GPU VRAM."
        echo "  - status: Kiểm tra cổng nào đang mở và dung lượng VRAM hiện tại."
        exit 1
        ;;
esac
