from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy
import torch
import rlkit.torch.pytorch_util as ptu
M = 256
device  = torch.device("cuda:0")
ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

qf = FlattenMlp(
    input_size=10,
    output_size=1,
    hidden_sizes=[M, M]
).to(device)


policy = SkillTanhGaussianPolicy(
        obs_dim=10,
        action_dim=1,
        hidden_sizes=[M, M],
        skill_dim=10
    ).to(device)



optim = torch.optim.Adam(policy.parameters())

new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = policy(
            torch.randn(10,10, device=device), skill_vec=torch.randn(10,10, device=device), reparameterize=True, return_log_prob=True,
        )

optim.zero_grad()
log_pi.mean().backward()
optim.step()
