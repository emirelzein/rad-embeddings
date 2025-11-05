![Rad Embeddings Logo](https://rad-embeddings.github.io/assets/logo.svg)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://rad-embeddings.github.io/assets/splash.svg">
  <img alt="Rad Embeddings overview" src="https://rad-embeddings.github.io/assets/splash_light.png">
</picture>

This repo contains a Python package for RAD embeddings, see [project webpage](https://rad-embeddings.github.io/) for more information.

# Usage

You can pip-install it.

```
pip install rad-embeddings
```

You train a DFA encoder, which converges pretty quickly.

```
import dfa_gym
import gymnasium as gym
import rad_embeddings as rad
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":

    n_envs = 16
    env_id = "DFAEnv-v0"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    env = gym.make(env_id)
    check_env(env)

    n_tokens = env.unwrapped.sampler.n_tokens

    train_env = make_vec_env(env_id, n_envs=n_envs)
    eval_env = gym.make(env_id)

    rad.Encoder.train(n_tokens=n_tokens, train_env=train_env, eval_env=eval_env, save_dir=save_dir, id=encoder_id)
```

You can then use a trained DFA encoder.

```
import dfa_gym
import gymnasium as gym
import rad_embeddings as rad
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":

    env_id = "DFAEnv-v0"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    env = gym.make(env_id)
    check_env(env)

    n_tokens = env.unwrapped.sampler.n_tokens

    sampler = env.unwrapped.sampler
    encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    dfa = sampler.sample()
    print(dfa)

    rad = encoder.dfa2rad(dfa)
    print(rad)

    token = encoder.rad2token(rad)
    print(token)
```

# Citation (Neurips'24 Paper)

```
@inproceedings{yalcinkayacompositional,
  title = {Compositional Automata Embeddings for Goal-Conditioned Reinforcement Learning},
  author = {Yalcinkaya, Beyazit and Lauffer, Niklas and Vazquez-Chanlatte, Marcell and Seshia, Sanjit A},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

# Citation (Repo)

```
@misc{rad-embd,
  author = {Yalcinkaya, Beyazit and Lauffer, Niklas and Vazquez-Chanlatte, Marcell and Seshia, Sanjit A},
  title = {RAD Embeddings},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RAD-Embeddings/rad-embeddings}},
}
```
