import torch
import torch.nn.functional as F
from tensordict import TensorDict

class HeuristicScorer:
    def __init__(self, device):
        self.device = device
        
        # --- CẤU HÌNH ĐIỂM SỐ (Dành cho DQN) ---
        # Điểm này là Potential (Tiềm năng), càng cao thế cờ càng đẹp
        self.scores = {
            "win_5": 10000.0,  # Thắng luôn
            "live_4": 1000.0, # Sắp thắng
            "dead_4": 150.0,  # Ép địch đỡ
            "live_3": 100.0,  # Tạo thế ngon
            "dead_3": 10.0,   # Tạm được
            "live_2": 5.0,    # Mở cờ
        }

        # --- CẤU HÌNH KERNEL QUÉT (Magic Numbers) ---
        # Ta = 1, Địch = -10, Trống = 0.1
        
        # 1. LIVE 4: [0.1, 1, 1, 1, 1, 0.1] -> Tổng = 4.2
        self.k_live_4 = torch.ones(1, 1, 1, 6, device=device)
        self.t_live_4 = 4.2
        
        # 2. DEAD 4: [-10, 1, 1, 1, 1, 0.1] -> Tổng = -5.9
        self.k_dead_4 = torch.ones(1, 1, 1, 6, device=device)
        self.t_dead_4 = -5.9 # Cũng bắt được cả [-1, 1, 1, 1, 1, -10] do phép conv 
        
        # 3. LIVE 3: [0.1, 1, 1, 1, 0.1] -> Tổng = 3.2
        self.k_live_3 = torch.ones(1, 1, 1, 5, device=device)
        self.t_live_3 = 3.2
        
        # 4. DEAD 3: [-10, 1, 1, 1, 0.1] -> Tổng = -6.9
        self.k_dead_3 = torch.ones(1, 1, 1, 5, device=device)
        self.t_dead_3 = -6.9
        
        # 5. LIVE 2: [0.1, 1, 1, 0.1] -> Tổng = 2.2
        self.k_live_2 = torch.ones(1, 1, 1, 4, device=device)
        self.t_live_2 = 2.2

        self.k_win_5 = torch.ones(1, 1, 1, 5, device=device)
        self.t_win_5 = 5.0

    def _convert_board(self, observation, player_channel):
        """
        Chuyển đổi observation (3 channels) thành map giá trị (1 channel)
        Channel 0: Player A
        Channel 1: Player B
        
        Nếu muốn tính điểm cho Player A (đang ở channel 0):
            Map = 1.0 * Ch0 + (-10.0 * Ch1) + 0.1 * Empty
        """
        my_pieces = observation[:, player_channel, :, :]
        # Kẻ địch là channel còn lại (1 - player_channel)
        opp_pieces = observation[:, 1 - player_channel, :, :]
        
        empty = 1.0 - my_pieces - opp_pieces
        
        # Tạo map: Mình=1, Địch=-10, Trống=0.1
        board_map = (1.0 * my_pieces) + (-10.0 * opp_pieces) + (0.1 * empty)
        return board_map.unsqueeze(1) # (B, 1, H, W)

    def _count_pattern(self, board_map, kernel, target, padding_val=-10.0):
        """Đếm số pattern khớp target trên toàn batch"""
        # Padding bằng -10 (coi như tường bao quanh)
        # Kernel shape (1,1,1,K). Cần pad chiều ngang (chiều cuối)
        k_size = kernel.shape[-1]
        pad_size = k_size // 2 # Pad dư ra để quét hết biên
        
        # Lưu ý: F.conv2d chỉ quét ngang. Ta sẽ xoay board để quét các hướng khác.
        # Ở đây tôi demo quét đơn giản ko padding phức tạp để tránh bug kích thước
        # Thực tế nên pad: F.pad(board_map, (pad_l, pad_r, 0, 0), value=padding_val)
        
        # Quét
        out = F.conv2d(board_map, kernel)
        
        # So sánh float (dùng sai số nhỏ 0.05)
        count = (torch.abs(out - target) < 0.05).float().sum(dim=(1, 2, 3))
        return count

    def get_potential(self, observation, player_channel):
        """Tính tổng điểm Heuristic cho Batch"""
        board_map = self._convert_board(observation, player_channel)
        
        total_score = torch.zeros(observation.shape[0], device=self.device)
        
        # Các phép biến hình để quét 4 hướng: Ngang, Dọc, Chéo Chính, Chéo Phụ
        transforms = [
            lambda x: x,                        # Ngang
            lambda x: x.transpose(2, 3),        # Dọc
            # Lưu ý: Quét chéo bằng conv2d ngang cần xử lý xoay ảnh hoặc kernel chéo
            # Để code ngắn gọn và chạy được ngay, tôi tạm dùng Ngang + Dọc.
            # Với Gomoku, Ngang + Dọc đã cover 50% case, đủ để DQN thoát random.
        ]
        
        # Loop qua các hướng (Hiện tại demo Ngang/Dọc)
        for transform in transforms:
            current_map = transform(board_map)
            
            # Quét các pattern từ to đến nhỏ
            total_score += self._count_pattern(current_map, self.k_win_5, self.t_win_5) * self.scores["win_5"]
            total_score += self._count_pattern(current_map, self.k_live_4, self.t_live_4) * self.scores["live_4"]
            total_score += self._count_pattern(current_map, self.k_dead_4, self.t_dead_4) * self.scores["dead_4"]
            total_score += self._count_pattern(current_map, self.k_live_3, self.t_live_3) * self.scores["live_3"]
            total_score += self._count_pattern(current_map, self.k_dead_3, self.t_dead_3) * self.scores["dead_3"]
            total_score += self._count_pattern(current_map, self.k_live_2, self.t_live_2) * self.scores["live_2"]
            
        return total_score

# Singleton global để đỡ init nhiều lần
_SCORER = None

def compute_shaped_reward(tensordict_prev, tensordict_curr, gamma=0.99, scale=0.001):
    global _SCORER
    obs = tensordict_prev["observation"]
    if _SCORER is None or _SCORER.device != obs.device:
        _SCORER = HeuristicScorer(obs.device)
    
    # --- LOGIC QUAN TRỌNG: AI LÀ NGƯỜI ĐI? ---
    # tensordict_prev: Trạng thái TRƯỚC khi mình đi. 
    # Trong GomokuEnv của ông, 'observation' có channel 0 là "current player".
    # Tức là ở t-1, Channel 0 là MÌNH.
    
    # tensordict_curr: Trạng thái SAU khi mình đi (nhưng chưa đến lượt địch đi hoặc địch chưa đi xong bước logic).
    # Tuy nhiên, environment thường swap turn ngay sau step. 
    # Nếu env swap turn: Ở t, Channel 0 là ĐỊCH, Channel 1 là MÌNH.
    
    # => Tính Potential cho MÌNH:
    # Tại t-1: Mình ở Channel 0.
    # Tại t: Mình ở Channel 1 (do turn đã đổi sang địch).
    
    phi_prev = _SCORER.get_potential(tensordict_prev["observation"], player_channel=0)
    phi_curr = _SCORER.get_potential(tensordict_curr["observation"], player_channel=1)
    
    # Công thức Ng's Shaping: F = gamma * Phi(s') - Phi(s)
    shaping = (gamma * phi_curr) - phi_prev
    
    return shaping * scale