from model import Model
from utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFAEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, model_cls=Model, n_tokens=10):
        super().__init__(observation_space, features_dim)
        in_feat_size = n_tokens + len(feature_inds)
        self.model = model_cls(in_feat_size, features_dim)
        self.n_tokens = n_tokens

    def forward(self, obs):
        feat = obs2feat(obs, n_tokens=self.n_tokens)
        return self.model(feat)
