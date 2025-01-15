import numpy as np
from dfa import DFA
from dfa_gym import DFAEnv
from stable_baselines3 import PPO
from utils import dfa2obs
import gymnasium as gym
from utils import DFAEnvFeaturesExtractor, LoggerCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

class Encoder():
    def __init__(self, load_file: str):
        model = PPO.load(load_file)
        model.set_parameters(load_file)
        for param in model.policy.parameters():
            param.requires_grad = False
        model.policy.eval()
        self.obs2rad = model.policy.features_extractor
        self.rad2token = lambda _rad: model.policy.action_net(_rad).argmax(dim=1)
        self.n_tokens = self.obs2rad.n_tokens

    def dfa2rad(self, dfa: DFA) -> np.array:
        assert len(dfa.inputs) == self.n_tokens
        obs = dfa2obs(dfa)
        rad = self.obs2rad(obs)
        return rad

    @staticmethod
    def train(n_tokens: int, train_env: gym.Env, save_dir: str, eval_env: gym.Env | None = None, id: str = "rad"):
        save_dir = save_dir[:-1] if save_dir.endswith("/") else save_dir
        config = dict(
            policy = "MlpPolicy",
            env = train_env,
            learning_rate = 1e-3,
            n_steps = 512,
            batch_size = 1024,
            n_epochs = 2,
            gamma = 0.9,
            gae_lambda = 0.5,
            clip_range = 0.1,
            ent_coef = 1e-2,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            policy_kwargs = dict(
                features_extractor_class = DFAEnvFeaturesExtractor,
                features_extractor_kwargs = dict(features_dim=32, n_tokens=n_tokens),
                net_arch=dict(pi=[], vf=[]),
                share_features_extractor=True,
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/"
        )

        model = PPO(**config)

        print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
        print(model.policy)

        callback_list = []
        logger_callback = LoggerCallback(gamma=config["gamma"])
        callback_list.append(logger_callback)
        if eval_env is not None:
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
            eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
            callback_list.append(eval_callback)
        model.learn(10_000_000, callback=callback_list)
        model.save(f"{save_dir}/{id}")

if __name__ == "__main__":
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.env_checker import check_env

    n_envs = 16
    env_id = "DFAEnv-v1"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    env = gym.make(env_id)
    check_env(env)

    n_tokens = env.unwrapped.sampler.n_tokens

    train_env = make_vec_env(env_id, n_envs=n_envs)
    eval_env = gym.make(env_id)

    Encoder.train(n_tokens=n_tokens, train_env=train_env, eval_env=eval_env, save_dir=save_dir, id=encoder_id)

    sampler = env.unwrapped.sampler
    encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    dfa = sampler.sample()
    print(dfa)

    rad = encoder.dfa2rad(dfa)
    print(rad)

    token = encoder.rad2token(rad)
    print(token)
