import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
import logging
import random
import torch
from tensordict import TensorDict
import time

from gomoku_rl.policy import get_policy
from gomoku_rl.core import Gomoku
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)

# --- Bỏ các import của PyQt5 ---

# --- Giữ lại các hàm và lớp cần thiết ---

def make_model(cfg: DictConfig, algo_cfg: DictConfig):
    board_size = cfg.board_size
    device = cfg.device
    action_spec = DiscreteTensorSpec(
        board_size * board_size, shape=[1,], device=device,
    )
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=cfg.device, shape=[2, 3, board_size, board_size],
            ),
            "action_mask": BinaryDiscreteTensorSpec(
                n=board_size * board_size,
                device=device,
                shape=[2, board_size * board_size],
                dtype=torch.bool,
            ),
        },
        shape=[2,],
        device=device,
    )
    model = get_policy(
        name=algo_cfg.name,
        cfg=algo_cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )
    return model

def get_ai_move(policy, env: Gomoku):
    """Hàm lấy nước đi từ một policy cụ thể"""
    with torch.no_grad():
        tensordict = TensorDict(
            {
                "observation": env.get_encoded_board(),
                "action_mask": env.get_action_mask(),
            },
            batch_size=1,
            device=policy.device,
        )
        tensordict = policy(tensordict)
        return tensordict["action"].item()

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="demo_battle")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    # --- Tải các model vào 2 đội ---
    team_A_paths = cfg.get("team_A_checkpoints", [])
    team_B_paths = cfg.get("team_B_checkpoints", [])

    if not team_A_paths or not team_B_paths:
        logging.error("Please define 'team_A_checkpoints' and 'team_B_checkpoints' in your config file.")
        return

    logging.info("--- Loading AI Teams ---")
    
    team_A_policies = []
    for path in team_A_paths:
        logging.info(f"Loading Team A model from: {path}")
        model_instance = make_model(cfg, cfg.algo_ppo)
        model_instance.load_state_dict(torch.load(path, map_location=cfg.device))
        model_instance.eval()
        team_A_policies.append(model_instance)

    team_B_policies = []
    for path in team_B_paths:
        logging.info(f"Loading Team B model from: {path}")
        model_instance = make_model(cfg, cfg.algo_dqn)
        model_instance.load_state_dict(torch.load(path, map_location=cfg.device))
        model_instance.eval()
        team_B_policies.append(model_instance)
    
    logging.info("--- All models loaded. Starting the tournament! ---")

    # --- Thiết lập giải đấu ---
    num_games = 200
    board_size = cfg.get("board_size", 15)
    
    team_A_wins = 0
    team_B_wins = 0

    start_time = time.time()

    for i in range(1, num_games + 1):
        env = Gomoku(num_envs=1, board_size=board_size, device="cpu") # Dùng CPU cho env
        env.reset()

        # Ngẫu nhiên chọn đội đi trước (quân Đen)
        if random.random() < 0.5:
            black_player_team = team_A_policies
            white_player_team = team_B_policies
            black_team_name = "Team A"
            white_team_name = "Team B"
        else:
            black_player_team = team_B_policies
            white_player_team = team_A_policies
            black_team_name = "Team B"
            white_team_name = "Team A"

        # Mỗi đội ngẫu nhiên cử ra một thành viên
        black_policy = random.choice(black_player_team)
        white_policy = random.choice(white_player_team)
        
        logging.info(f"\n--- Game {i}/{num_games} ---")
        logging.info(f"Black: {black_team_name} | White: {white_team_name}")

        done = False
        while not done:
            if env.turn == 0: # Lượt của quân Đen
                action = get_ai_move(black_policy, env)
            else: # Lượt của quân Trắng
                action = get_ai_move(white_policy, env)
            
            done_tensor, _ = env.step(torch.tensor([action]))
            # Lấy giá trị boolean từ tensor
            done = done_tensor.item()

        # --- Ghi nhận kết quả ---
        move_count = env.move_count.item()
        winner_team_name = "Draw" # Mặc định là Hòa

        # Chỉ xác định người thắng nếu không phải là hòa
        if move_count < board_size * board_size:
            # move_count - 1 là chỉ số của nước đi cuối cùng (bắt đầu từ 0)
            if (move_count - 1) % 2 == 0: # Nước cuối là của quân Đen
                winner_team_name = black_team_name
            else: # Nước cuối là của quân Trắng
                winner_team_name = white_team_name
        
        if winner_team_name == "Team A":
            team_A_wins += 1
        elif winner_team_name == "Team B":
            team_B_wins += 1
        # Nếu hòa thì không ai được cộng điểm

        logging.info(f"Winner: {winner_team_name}")
        
        logging.info(f"Current Score: Team A [{team_A_wins}] - [{team_B_wins}] Team B")

    # --- In kết quả cuối cùng ---
    end_time = time.time()
    logging.info("\n\n--- TOURNAMENT FINISHED ---")
    logging.info(f"Total games played: {num_games}")
    logging.info(f"Final Score: Team A [{team_A_wins}] - [{team_B_wins}] Team B")
    
    win_rate_A = (team_A_wins / num_games) * 100
    win_rate_B = (team_B_wins / num_games) * 100
    
    logging.info(f"Team A (0.pt, 1.pt) Win Rate: {win_rate_A:.2f}%")
    logging.info(f"Team B (2.pt, 3.pt) Win Rate: {win_rate_B:.2f}%")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()