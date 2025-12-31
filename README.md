# Hướng Dẫn Sử Dụng Project

Dưới đây là hướng dẫn chi tiết cách cài đặt môi trường, huấn luyện (training) và chạy thử nghiệm (demo) cho các thuật toán RL (PPO, DQN, MCTS). Hãy sử dụng python 3.10.x để tương thích thư viện 

## 1. Cài đặt môi trường

Bước đầu tiên, hãy pull code về và thiết lập môi trường ảo:

```bash
# 1. Pull code về máy
git pull origin main

# 2. Tạo môi trường ảo (Virtualenv hoặc Conda)
# Ví dụ với Conda:
conda create -n rl_env python=3.9
conda activate rl_env

# 3. Cài đặt các thư viện cần thiết
pip install -r requirement.txt

# 4. Cài đặt gói hiện tại ở chế độ editable
pip install -e .
```

## 2. Yêu cầu phần cứng & Chuẩn bị

> **⚠️ CẢNH BÁO PHẦN CỨNG:**
> Để train hiệu quả, yêu cầu sử dụng GPU mạnh.
> * **TFLOPS:** Tầm 80 - 90 (Tương đương RTX 4090 hoặc cụm GPU mạnh).
> * **VRAM:** >= 10GB.

**Trước khi bắt đầu Train:**
* Đăng ký tài khoản tại [WandB (Weights & Biases)](https://wandb.ai/) để theo dõi số liệu trực tuyến trong quá trình train.

## 3. Huấn luyện (Training)

File chạy chính: `scripts/train_InRL.py`. Bạn cần chỉnh sửa file này để chọn config phù hợp.

### Training PPO
1. Mở file `scripts/train_InRL.py`.
2. Sửa dòng config thành:
   ```python
   config_name = "train_InRL.yaml"
   ```
3. Chạy lệnh:
   ```bash
   python scripts/train_InRL.py
   ```

### Training DQN
1. Mở file `scripts/train_InRL.py`.
2. Sửa dòng config thành:
   ```python
   config_name = "train_InRL_DQN.yaml"
   ```
3. Chạy lệnh:
   ```bash
   python scripts/train_InRL.py
   ```

> **Lưu ý:** **MCTS** không cần train vì đây là thuật toán tìm kiếm cây (Search Tree) để nâng cấp quá trình tư duy dựa trên Policy của PPO.

## 4. Chạy Demo (Test sản phẩm)

Dưới đây là cách chạy thử nghiệm các thuật toán sau khi đã có model.

### Chơi với DQN
* **Cấu hình phe (Quân Đen/Trắng):** Vào `cfg/demo.yaml`. Nếu muốn chơi quân đen, đổi sang sử dụng model `1.pt` (thay vì `0.pt`).
* **Chạy lệnh:**
    ```bash
    python scripts/demo.py
    ```

### Chơi với PPO
* **Chạy lệnh:**
    ```bash
    python scripts/demo1.py
    ```

### Chơi với MCTS
* **Chạy lệnh:**
    ```bash
    python scripts/demo_MCTS.py
    ```

## 5. Đấu thuật toán (Benchmark)

Muốn xem kết quả các thuật toán đấu với nhau, bạn có 2 cách:

* **Cách 1:** Bật các tiến trình demo (terminal) song song cho các thuật toán và để chúng tự chơi (Đen vs Trắng).
* **Cách 2 (Khuyên dùng):** Chạy script test tự động.
    ```bash
    python tests/test_MTCS.py
    ```
    *Kịch bản test:* PPO (500 epoch) + MCTS **VS** PPO (1000 epoch).
    *Kết quả dự kiến:* MCTS sẽ áp đảo PPO thuần.
    *(Note: DQN hiện tại khá yếu nên PPO chắc chắn thắng).*

## 6. Link Demo Video

Dưới đây là video demo thực tế của từng thuật toán:

* **Demo DQN:** [Xem tại đây](https://drive.google.com/file/d/1b694Z-pzHnAu7qWIx4_vTBl0hkrza8Y_/view?usp=sharing)
* **Demo PPO:** [Xem tại đây](https://drive.google.com/file/d/1b4Oftv-Yypyonh4lQlOJKYnx8sHZZ-WU/view?usp=sharing)
* **Demo MCTS:** [Xem tại đây](https://drive.google.com/file/d/1b1NmXKfcuQxDd4LtqNJKygfuueF8UtOR/view?usp=sharing)