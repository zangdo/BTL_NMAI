from typing import Dict, Any
import torch
import torch.nn as nn
from tensordict import TensorDict
from omegaconf import DictConfig

from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.data.replay_buffers.samplers import RandomSampler

from .base import Policy
from .common import (
    make_sac_actor,
    make_sac_critic,
    get_optimizer,
    get_replay_buffer,
)

class SAC(Policy):
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: _device_t = "cuda",
    ) -> None:
        super().__init__(cfg, action_spec, observation_spec, device)
        self.action_spec = action_spec 
        self.cfg: DictConfig = cfg
        self.device: _device_t = device

        # 1. Init Networks
        self.actor = make_sac_actor(cfg, action_spec, device)
        
        # SAC dùng 2 Q-networks để giảm overestimation
        self.qnet_1 = make_sac_critic(cfg, action_spec, device)

        # Warm up networks
        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.qnet_1(fake_input)

        # 2. Replay Buffer (Off-policy)
        self.batch_size: int = cfg.batch_size
        self.buffer_size: int = cfg.buffer_size
        self.n_optim: int = cfg.n_optim # Số lần update per step
        
        self.replay_buffer = get_replay_buffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            sampler=RandomSampler(),
            device=cfg.buffer_device,
        )

        # 3. Loss Module (Discrete SAC)
        # alpha_init: nhiệt độ entropy ban đầu. 
        # target_entropy: mục tiêu entropy (thường là -0.98 * log(1/action_space) hoặc hằng số)
        self.loss_module = DiscreteSACLoss(
            actor_network=self.actor,
            qvalue_network=self.qnet_1,
            num_actions=action_spec.space.n,
            loss_function="smooth_l1",
            delay_qvalue=True, # Sử dụng target network cho Q
            target_entropy_weight=cfg.get("target_entropy_weight", 0.98), # Auto-tune alpha target
            alpha_init=cfg.get("alpha_init", 1.0),
        )
        self.loss_module.make_value_estimator(gamma=cfg.gamma)

        # 4. Optimizers
        # SAC thường dùng 3 optimizer riêng hoặc gom lại
        self.optimizer_actor = get_optimizer(
            cfg.optimizer_actor, 
            self.actor.parameters()  # <-- Thay đổi ở đây
        )
        
        # Optimizer cho Critic (Q-Network)
        # Vì bạn đang dùng Single Q-net (do bỏ qnet_2 ở bước trước), chỉ cần qnet_1
        self.optimizer_critic = get_optimizer(
            cfg.optimizer_critic, 
            self.qnet_1.parameters() # <-- Thay đổi ở đây
        )
        
        # Optimizer cho Alpha (Entropy temperature)
        # log_alpha là một tensor nằm trong loss_module, cái này giữ nguyên được
        # hoặc dùng list wrap lại cho chắc ăn
        self.optimizer_alpha = get_optimizer(
            cfg.optimizer_alpha, 
            [self.loss_module.log_alpha]
        )
        # 5. Target Updaters (Polyack averaging / Soft Update)
        self.target_update_interval = cfg.target_update_interval
        self.target_updater = SoftUpdate(self.loss_module, eps=cfg.polyak) # eps thường là 0.995 (1-tau)

        self._eval = False

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Trong SAC, actor output ra action dựa trên sampling (khi train) 
        # hoặc mode/argmax (khi eval)
        if self._eval:
            with torch.no_grad():
                # Lấy mode (hành động có xác suất cao nhất) hoặc mean
                return self.actor(tensordict.to(self.device))
        else:
            # Sampling theo phân phối Categorical
            return self.actor(tensordict.to(self.device))

    def learn(self, data: TensorDict) -> Dict[str, Any]:
        # Filter invalid transitions if needed
        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]
        
        # Chuẩn bị dữ liệu cho buffer
        data["next", "done"] = data["next", "done"].unsqueeze(-1)
        data = data.reshape(-1)
        self.replay_buffer.extend(data)

        print("Replay buffer size:", len(self.replay_buffer))
        if len(self.replay_buffer) < self.batch_size:
            return {}

        metrics = {
            "loss_actor": [],
            "loss_qvalue": [],
            "loss_alpha": [],
            "alpha": [],
            "entropy": []
        }

        # Training Loop
        for i in range(self.n_optim):
            batch = self.replay_buffer.sample().to(self.device)
            action_indices = batch["action"].long()
            
            # Nếu shape là (Batch, 1) thì squeeze để thành (Batch,)
            if action_indices.dim() > 1:
                action_indices = action_indices.squeeze(-1)
            
            # Chuyển sang One-Hot: (Batch, 225)
            # num_actions = 15*15 = 225. Lấy từ action_spec hoặc hardcode
            num_actions = self.action_spec.space.n
            action_one_hot = torch.nn.functional.one_hot(action_indices, num_classes=num_actions)
            
            # Ghi đè lại action trong batch bằng dạng one-hot
            batch.set("action", action_one_hot.float()) # Phải để float để nhân với Q-values
            # Tính toán Loss
            loss_vals = self.loss_module(batch)
            
            loss_actor = loss_vals["loss_actor"]
            loss_qvalue = loss_vals["loss_qvalue"]
            loss_alpha = loss_vals["loss_alpha"]
            
            # 1. Update Critic
            self.optimizer_critic.zero_grad()
            loss_qvalue.backward()
            torch.nn.utils.clip_grad_norm_(self.qnet_1.parameters(), self.cfg.max_grad_norm)
            self.optimizer_critic.step()

            # 2. Update Actor
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.optimizer_actor.step()

            # 3. Update Alpha (Auto-tuning temperature)
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()

            # 4. Soft Update Target Networks
            if i % self.target_update_interval == 0:
                self.target_updater.step()

            # Logging
            metrics["loss_actor"].append(loss_actor.item())
            metrics["loss_qvalue"].append(loss_qvalue.item())
            metrics["loss_alpha"].append(loss_alpha.item())
            metrics["alpha"].append(loss_vals["alpha"].item())
            metrics["entropy"].append(loss_vals["entropy"].item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def train(self):
        self.actor.train()
        self.qnet_1.train()
        self._eval = False

    def eval(self):
        self.actor.eval()
        self.qnet_1.eval()
        self._eval = True

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "qnet_1": self.qnet_1.state_dict(),
            "loss_module": self.loss_module.state_dict(), # Lưu log_alpha
        }

    def load_state_dict(self, state_dict: Dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.qnet_1.load_state_dict(state_dict["qnet_1"])
        
        # Load alpha param
        if "loss_module" in state_dict:
            self.loss_module.load_state_dict(state_dict["loss_module"])
        
        # Re-init target updater để sync weights mới
        self.target_updater = SoftUpdate(self.loss_module, eps=self.cfg.polyak)