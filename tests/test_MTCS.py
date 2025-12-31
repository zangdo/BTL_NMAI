import torch
import torch.nn.functional as F
import numpy as np
import math
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Giả định các file này tồn tại trong project của bạn
from gomoku_rl.env import GomokuEnv
from gomoku_rl.policy import get_policy
from tensordict import TensorDict
import random
from gomoku_rl import CONFIG_PATH
from gomoku_rl.policy.MCTS import MCTS

from typing import Optional

# --- Hàm chính để chạy thử nghiệm ---
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_InRL")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Tải 2 policy đã huấn luyện ---
    print("Loading pretrained PPO models...")
    env = GomokuEnv(num_envs=1, board_size=15, device=device)
    policy_0 = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec= env.action_spec,
        observation_spec= env.observation_spec,
        device= device,
    )
    policy_0.load_state_dict(torch.load(cfg.black_checkpoint, map_location=device))
    policy_0.eval()
    
    policy_1 = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec= env.action_spec,
        observation_spec= env.observation_spec,
        device= device,
    )
    policy_1.load_state_dict(torch.load(cfg.white_checkpoint, map_location=device))
    policy_1.eval()


    print("Models loaded. Starting MCTS vs PPO match...")

    # --- Kịch bản: MCTS(0,1) vs PPO(0,1) ---
    num_games = 30
    mcts_wins = 0
    ppo_wins = 0
    for i in tqdm(range(num_games), desc="Playing games"):
        tensordick=env.reset()
        # MCTS đi trước (quân Đen) ở các ván chẵn
        mcts_player_is_black = (i % 2 == 0)
        
        mcts_planner = MCTS(policy_0, policy_1, num_simulations=200) # 50 mô phỏng cho mỗi nước

        done = False
        while not done:
            is_mcts_turn = ((env.gomoku.turn == 0).item() and mcts_player_is_black) or \
                           ((env.gomoku.turn == 1).item() and not mcts_player_is_black)

            if is_mcts_turn:
                # Lượt của MCTS
                action = mcts_planner.run(env)
                #print(mcts_planner.root.visit_count)
                tensordick.set("action", torch.tensor([action], device=device))
            else:
                current_policy = random.choice([policy_2, policy_3])

                with torch.no_grad():
                    tensordick = current_policy(tensordick)
                action = tensordick.get("action").item()

            # Thực hiện nước đi
            tensordick = env.step(tensordick)
            mcts_planner.advance_root(action)
            done = tensordick.get("done").item()
        
        # Ghi nhận kết quả
        winner_turn = (env.gomoku.move_count.item() - 1) % 2
        
        if (winner_turn == 0 and mcts_player_is_black) or \
           (winner_turn == 1 and not mcts_player_is_black):
            mcts_wins += 1
            print(f"Game {i+1}: MCTS wins!")
            print(env.gomoku.board.squeeze(0))
            print(env.gomoku.move_count.squeeze(0).item())
        else:
            ppo_wins += 1
            print(f"Game {i+1}: PPO wins!")
            print(env.gomoku.board.squeeze(0))
            print(env.gomoku.move_count.squeeze(0).item())

    print("\n--- Match Results ---")
    print(f"MCTS wins: {mcts_wins}/{num_games}")
    print(f"PPO wins: {ppo_wins}/{num_games}")

if __name__ == "__main__":
    main()