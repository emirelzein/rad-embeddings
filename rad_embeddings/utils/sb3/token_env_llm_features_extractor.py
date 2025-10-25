import torch
from torch import nn
from typing import List
from dfa import DFA
from dfa.utils import dfa2dict
from sentence_transformers import SentenceTransformer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TokenEnvLLMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor that replaces the learned RAD encoder with an LLM embedding.
    It converts `dfa_obs` to a concise textual description, embeds it using a
    SentenceTransformer, and concatenates the embedding with CNN features from `obs`.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tokens: int = 10,
    ):
        super().__init__(observation_space, features_dim)
        self.n_tokens = n_tokens
        self.embed_model_name = embed_model_name

        # Load embedding model once
        self._embedder = SentenceTransformer(self.embed_model_name)
        emb_dim = self._embedder.get_sentence_embedding_dimension()
        # Project to 32 dims to mirror baseline RAD model output size
        self._proj = nn.Linear(emb_dim, 32)

        # Mirror the baseline image conv
        c, w, h = observation_space["obs"].shape  # CxWxH
        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Cache to avoid recomputing embeddings for identical DFAs
        self._cache: dict[int, torch.Tensor] = {}

    def forward(self, dict_obs):
        device = dict_obs["obs"].device
        obs_img = dict_obs["obs"]
        dfa_obs = dict_obs["dfa_obs"]

        # Ensure batched
        if dfa_obs.ndim == 1:
            dfa_obs = dfa_obs.unsqueeze(0)
        if obs_img.ndim == 3:
            obs_img = obs_img.unsqueeze(0)

        # Build/cache embeddings
        dfa_ints: List[int] = [self._dfa_obs_to_int(x) for x in dfa_obs]
        texts: List[str] = []
        to_compute_idx: List[int] = []
        embs_list: List[torch.Tensor | None] = []

        for i, dfa_int in enumerate(dfa_ints):
            if dfa_int in self._cache:
                embs_list.append(self._cache[dfa_int].to(device))
            else:
                texts.append(self._dfa_int_to_text(dfa_int))
                to_compute_idx.append(i)
                embs_list.append(None)

        if texts:
            with torch.no_grad():
                embs_new = self._embedder.encode(
                    texts, convert_to_tensor=True, device=device
                )
                if embs_new.ndim == 1:
                    embs_new = embs_new.unsqueeze(0)
                embs_new = embs_new.to(device=device, dtype=torch.float32)

            j = 0
            for i in to_compute_idx:
                emb = embs_new[j]
                embs_list[i] = emb
                self._cache[dfa_ints[i]] = emb.detach().cpu()
                j += 1

        llm_embs = torch.stack(embs_list, dim=0)  # [B, D]
        llm_embs = self._proj(llm_embs)  # [B, 32]
        img_feat = self.image_conv(obs_img)  # [B, F]
        return torch.cat((img_feat, llm_embs), dim=1)

    def _dfa_obs_to_int(self, dfa_obs_row: torch.Tensor) -> int:
        bits = "".join(str(int(x)) for x in dfa_obs_row.detach().cpu().view(-1).tolist())
        return int(bits)

    def _dfa_int_to_text(self, dfa_int: int) -> str:
        tokens = list(range(self.n_tokens))
        dfa = DFA.from_int(dfa_int, tokens)
        return repr(dfa)
