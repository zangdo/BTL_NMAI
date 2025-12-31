from typing import Generator


from torch.optim import Optimizer, Adam, AdamW
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.cuda import _device_t
from torchrl.modules import ProbabilisticActor
from torch.distributions.categorical import Categorical
from torchrl.modules.models import ConvNet, MLP
from torchrl.modules import ValueOperator
from torchrl.data import TensorSpec
from torchrl.modules import (
    DuelingCnnDQNet,
    EGreedyModule,
    QValueActor,
    ActorValueOperator,
    SafeModule,
)
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer,
    LazyTensorStorage,
    Sampler,
)

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict


from omegaconf import DictConfig, OmegaConf
from typing import Callable, Iterable

from gomoku_rl.utils.module import (
    ValueNet,
    ActorNet,
    ResidualTower,
    PolicyHead,
    ValueHead,
    MyDuelingCnnDQNet,
    QNet,
    _PolicyHead,
)
from gomoku_rl.utils.misc import get_kwargs


def make_dqn_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    net_kwargs = get_kwargs(cfg, "num_residual_blocks", "num_channels")
    net = MyDuelingCnnDQNet(
        in_channels=3, out_features=action_spec.space.n, **net_kwargs
    )
    actor = QValueActor(
        net,
        spec=action_spec,
        action_mask_key="action_mask",
    ).to(device)
    return actor


def make_egreedy_actor(
    actor: TensorDictModule,
    action_spec: TensorSpec,
    eps_init: float = 1.0,
    eps_end: float = 0.10,
    annealing_num_steps: int = 1000,
):
    explorative_policy = TensorDictSequential(
        actor,
        EGreedyModule(
            spec=action_spec,
            eps_init=eps_init,
            eps_end=eps_end,
            annealing_num_steps=annealing_num_steps,
            action_mask_key="action_mask",
        ),
    )
    return explorative_policy


def make_ppo_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    actor_net = ActorNet(
        residual_tower=ResidualTower(
            in_channels=3,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(
        module=actor_net, in_keys=["observation", "action_mask"], out_keys=["probs"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    return policy_module


def make_critic(
    cfg: DictConfig,
    device: _device_t,
):
    value_net = ValueNet(
        residual_tower=ResidualTower(
            in_channels=3,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        num_channels=cfg.num_channels,
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module


def make_ppo_ac(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    residual_tower = ResidualTower(
        in_channels=3,
        num_channels=cfg.num_channels,
        num_residual_blocks=cfg.num_residual_blocks,
    ).to(device)

    common_module = SafeModule(
        module=residual_tower, in_keys=["observation"], out_keys=["hidden"]
    )

    policy_head = PolicyHead(
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(
        module=policy_head, in_keys=["hidden", "action_mask"], out_keys=["probs"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_head = ValueHead(
        num_channels=cfg.num_channels,
    ).to(device)

    value_module = ValueOperator(
        module=value_head,
        in_keys=["hidden"],
    )

    return ActorValueOperator(common_module, policy_module, value_module)


def make_dataset_naive(tensordict: TensorDict, batch_size: int) -> Generator[TensorDict, None, None]:
    tensordict = tensordict.reshape(-1)
    assert tensordict.shape[0] >= batch_size
    perm = torch.randperm(
        (tensordict.shape[0] // batch_size) * batch_size,
        device=tensordict.device,
    ).reshape(-1, batch_size)
    for indices in perm:
        yield tensordict[indices]


def get_optimizer(cfg: DictConfig, params: Iterable[Parameter]) -> Optimizer:
    dict_cls: dict[str, Optimizer] = {
        "adam": Adam,
        "adamw": AdamW,
    }
    name: str = cfg.name.lower()
    assert name in dict_cls

    return dict_cls[name](params=params, **cfg.kwargs)
def get_replay_buffer(
    buffer_size: int,
    batch_size: int,
    sampler: Sampler | None = None,
    device: _device_t = None,
):
    storage = LazyTensorStorage(max_size=buffer_size, device=device)
    buffer = TensorDictReplayBuffer(
        storage=storage, batch_size=batch_size, sampler=sampler
    )
    return buffer

class SACActorNet(nn.Module):
    """Mạng Actor dành riêng cho SAC: Trả về Logits thay vì Probs"""
    def __init__(
        self,
        residual_tower: ResidualTower,
        out_features: int,
        num_channels: int = 32,
    ) -> None:
        super().__init__()
        self.residual_tower: nn.Module = residual_tower
        # Dùng _PolicyHead (có gạch dưới) để lấy logits đã mask (-inf)
        # KHÔNG dùng PolicyHead vì nó có Softmax
        self.policy_head: nn.Module = _PolicyHead(
            out_features=out_features,
            num_channels=num_channels,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedding = self.residual_tower(x)
        # Output này là Logits (các ô không hợp lệ sẽ là -inf)
        logits: torch.Tensor = self.policy_head(embedding, mask)
        return logits

def make_sac_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    # SAC Actor output probabilities (Softmax)
    actor_net = SACActorNet(
        residual_tower=ResidualTower(
            in_channels=3,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(
        module=actor_net, 
        in_keys=["observation", "action_mask"], 
        out_keys=["logits"] # SAC Loss thường nhận logits để tính log_prob ổn định hơn
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    return policy_module

def make_sac_critic(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    # Q-Network: Observation -> Q-values (cho tất cả action)
    net_kwargs = get_kwargs(cfg, "num_residual_blocks", "num_channels")
    
    q_net = QNet(
        residual_tower=ResidualTower(
            in_channels=3, 
            num_channels=cfg.num_channels, 
            num_residual_blocks=cfg.num_residual_blocks
        ),
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels
    ).to(device)

    # QValueActor trong TorchRL thực chất là một wrapper giúp xử lý action space
    # Nhưng với SAC Loss, ta chỉ cần nó trả về 'action_value' (Q-values)
    q_module = TensorDictModule(
        module=q_net,
        in_keys=["observation"],
        out_keys=["action_value"], # Key chuẩn mà DiscreteSACLoss mong đợi
    )
    return q_module
