Hướng dẫn sử dụng
pull về
tạo môi trường ảo hoặc conda
pip install -r requirement.txt
pip install -e .


Trước khi bắt đầu train thì tạo tài khoản wandb(web cho xem số liệu trực tuyến ngay lúc train)
Training PPO
vào scripts/train_InRL.py
kiểm tra xem config_name  = "train_InRL.yaml"
Training DQN
vào scripts/train_InRL.py
kiểm tra xem config_name  = "train_InRL_DQN.yaml"
Sau đó chạy python scripts/train_InRL.py
MCTS ko cần train vì nó là bộ nâng cấp quá trình tư duy của PPO
(cảnh cáo dùng gpu xịn 1 tý TFLOPS tầm 80 90 , VRAM>=10GB)


Muốn test thử sản phẩm thì làm các bước sau

Chơi với DQN
(nếu muốn chơi quân đen thì chỉnh cfg/demo.yaml sang sử dụng mỗi model 1.pt thay vì sử dụng mỗi 0.pt như trc)
rồi chạy python scripts/demo.py thôi hẹ hẹ hẹ

Chơi với PPO
chạy python scripts/demo1.py

Chơi với MCTS
chạy python scripts/demo_MCTS.py


muốn xem kết quả đấu nhau của các thuật toán
cách 1: bật cách tiến trình demo cho các thuật toán rồi chơi song song đen trắng
cách 2:  chạy python tests/test_MTCS.py để xem MCTS nghiền nát PPO như thế nào (còn về PPO vs DQN thì 100% PPO thắng nhé :) DQN kém lắm , ở đây t cho PPO 500 epoch + MCTS đấu với PPO 1000 epoch)

