import argparse
import torch
import token_env
import gymnasium as gym
from encoder import Encoder
from dfa_gym import DFAWrapper
from stable_baselines3 import PPO
from rad_embeddings.utils import TokenEnvFeaturesExtractor, LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from dfa_samplers import ReachSampler, ReachAvoidSampler, RADSampler

parser = argparse.ArgumentParser(description="Train policy for TokenEnv with optional encoder")
parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
parser.add_argument("--encoder-file", type=str, default=None, help="Path to encoder file (optional)")
args = parser.parse_args()

SEED = args.seed
set_random_seed(SEED)
n_envs = 16
env_id = "TokenEnv-v1-fixed"

env = gym.make(env_id)
check_env(env)
n_tokens = env.unwrapped.n_tokens

reach_avoid_sampler = ReachAvoidSampler(n_tokens=n_tokens, max_size=6, p=None, prob_stutter=1.0)

env_kwargs = dict(env_id=env_id, sampler=reach_avoid_sampler, label_f=token_env.TokenEnv.label_f)

env = make_vec_env(DFAWrapper, env_kwargs=env_kwargs, n_envs=n_envs)

if args.encoder_file:
    encoder = Encoder(load_file=args.encoder_file)
else:
    encoder = None

config = dict(
    policy = "MultiInputPolicy",
    env = env,
    n_steps = 128,
    batch_size = 256,
    gamma = 0.99,
    policy_kwargs = dict(
        features_extractor_class=TokenEnvFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=1056, encoder=encoder),
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64]),
        share_features_extractor=True,
        activation_fn=torch.nn.ReLU
    ),
    verbose = 10,
    tensorboard_log = f"exps_no_embed/runs/"
)

model = PPO(**config)

print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)

logger_callback = LoggerCallback(gamma=config["gamma"])

model.learn(1_000_000, callback=[logger_callback])
model.save(f"exps_no_embed/token_env_reach_avoid_policy_seed{SEED}")

