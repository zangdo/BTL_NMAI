import torch
import torch.nn.functional as F
import numpy as np
import math
import logging

# Giả định các file này tồn tại trong project của bạn
from gomoku_rl.env import GomokuEnv
from gomoku_rl.policy import get_policy
from tensordict import TensorDict
from gomoku_rl import CONFIG_PATH


from typing import Optional

class Node:
    """Một node trong cây MCTS."""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.state: Optional[torch.Tensor] = None # Chỉ dùng cho node gốc để tiện debug

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, policy_black, policy_white, c_puct=1.5, num_simulations=200, root:Node= Node(prior=0)):
        self.policy_black = policy_black
        self.policy_white = policy_white
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root = root
        self.board_size = 15 # Giả định kích thước bàn cờ

    def expand_node(self, root: Node, env: GomokuEnv, tensordict: TensorDict):
        current_policy = self.policy_black if (env.gomoku.turn == 0).item() else self.policy_white    
        with torch.no_grad():
            tensordict = current_policy(tensordict)
            policy_probs = tensordict.get("probs").squeeze(0) # Giả định policy trả về logits
        # Mở rộng node gốc
        for action, prob in enumerate(policy_probs):
            if tensordict.get("action_mask").squeeze(0)[action]: # Chỉ thêm các nước đi hợp lệ
                root.children[action] = Node(prior=prob.item())
        return tensordict
    def advance_root(self, action: int):
        """
        Cập nhật gốc của cây sau khi một nước đi được thực hiện.
        Node con tương ứng với hành động sẽ trở thành gốc mới.
        """
        if self.root.expanded():
            if action in self.root.children:
                self.root = self.root.children[action]
                #print(f"Advanced MCTS root to action {action}.")
                # Không còn cha, không cần tham chiếu ngược
            else:
                # Nếu nước đi không có trong cây (ví dụ: nước đi của người), 
                # chúng ta phải tạo một cây mới
                error_message = f"FATAL ERROR: Move {action} not found in MCTS tree children. Aborting."
                print(error_message)
                logging.error(error_message)
                # Dừng chương trình ngay lập tức
                raise ValueError(error_message)
    def run(self, env: GomokuEnv):
        """Chạy MCTS để quyết định nước đi tốt nhất từ trạng thái hiện tại của env."""
        root = self.root
        tensordict = TensorDict(
            {
                "observation": env.gomoku.get_encoded_board(),
                "action_mask": env.gomoku.get_action_mask(),
            },
            env.batch_size,
            device=env.device,
        )
        if not root.expanded():
            tensordict = self.expand_node(root, env, tensordict)
            value = tensordict.get("state_value").item()
            # Lan truyền ngược giá trị của gốc
            root.value_sum += value
            root.visit_count += 1
        # Chạy N lần mô phỏng
        for _ in range(self.num_simulations):
            # Tạo một bản sao của môi trường để không làm ảnh hưởng env gốc
            sim_env = env.clone()
            # Nếu có thêm trạng thái nào cần copy, thêm vào đây
            tensordick = tensordict.clone()
            # --- Giai đoạn 1: Selection ---
            node = root
            search_path = [root] # Lưu lại đường đi
            
            while node.expanded():
                action, node = self.select_child(node)
                tensordick.set("action", torch.tensor([action], device=env.device))
                tensordick = sim_env.step(tensordick)
                if node is not None:
                    search_path.append(node)
                else:
                    break
            
            # --- Giai đoạn 2 & 3: Expansion & Evaluation ---
            leaf_node = search_path[-1]

            # Kiểm tra xem game đã kết thúc ở node lá chưa
            if not sim_env.gomoku.done:
                tensordick = self.expand_node(leaf_node, sim_env, tensordick)
                value = tensordick.get("state_value").item()
            else:
                # Game đã kết thúc, giá trị là kết quả thực tế
                # (1.0 nếu người chơi ở node cha thắng, -1.0 nếu thua)
                winner_turn = (sim_env.gomoku.move_count.item() - 1) % 2
                parent_turn = sim_env.gomoku.move_count.item() % 2
                value = 1.0 if winner_turn == parent_turn else -1.0

            # --- Giai đoạn 4: Backpropagation ---
            for node_in_path in reversed(search_path):
                node_in_path.visit_count += 1
                # Giá trị phải được lật ngược ở mỗi bước
                node_in_path.value_sum += value
                value = -value # Lật dấu cho người chơi trước đó
    
        # --- Ra quyết định cuối cùng ---
        visit_counts = torch.zeros(self.board_size * self.board_size)
        for action, node in root.children.items():
            visit_counts[action] = node.visit_count

        # Chọn nước đi được thăm nhiều nhất (chế độ thi đấu)
        best_action = int(visit_counts.argmax().item())
        return best_action

    def select_child(self, node: Node):
        """
        Chọn node con có điểm PUCT cao nhất.
        Nếu có nhiều node con cùng đạt điểm cao nhất, chọn ngẫu nhiên một trong số chúng.
        """
        best_score = -99999.0
        best_actions_and_children = () # Sử dụng một danh sách để lưu các lựa chọn tốt nhất

        for action, child in node.children.items():
            score = self.puct_score(node, child)
            if score > best_score:
                # Tìm thấy một lựa chọn tốt hơn hẳn, bắt đầu lại danh sách
                best_score = score
                best_actions_and_children = (action, child)
        if not best_actions_and_children:
            # Trường hợp hiếm gặp: không có children, trả về None
            return None, None
        #print(f"Selected action {best_actions_and_children[0]} with PUCT score {best_score}")
        return best_actions_and_children

    def puct_score(self, parent: Node, child: Node) -> float:
        """Tính điểm PUCT (UCB1 + prior)."""
        pb_c = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        # Q-value được tính từ góc nhìn của người chơi hiện tại
        q_value = child.value()
        return -q_value + pb_c