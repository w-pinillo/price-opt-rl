import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We need to calculate the features_dim before calling super().__init__
        self.embedding_dim = 16 # Default or from config if passed via features_extractor_kwargs
        
        # Calculate the CORRECT features_dim
        market_features_dim = observation_space["features"].shape[0]
        total_concat_size = self.embedding_dim + market_features_dim # 16 + 18 = 34
        
        # Now call super().__init__ with the correct calculated features_dim
        super().__init__(observation_space, features_dim=total_concat_size)

        # Define the Embedding Layer (now after super().__init__ so it's registered as a submodule)
        self.product_embedding = nn.Embedding(
            num_embeddings=observation_space["product_id"].n, 
            embedding_dim=self.embedding_dim
        )
        
        print(f"DEBUG: Final CustomFeatureExtractor calculated input dim: {self._features_dim}")

    def forward(self, observations) -> th.Tensor:
        product_ids_raw = observations["product_id"].long()
        
        # Squeeze out the extra dimension (if it exists)
        # This transforms (Batch_Size, 1, Num_Products) to (Batch_Size, Num_Products)
        product_ids_one_hot = product_ids_raw.squeeze(1)
        
        # Find the actual product index from the one-hot encoding
        # This converts (Batch_Size, Num_Products) to (Batch_Size,)
        product_indices = th.argmax(product_ids_one_hot, dim=1) # Now dim=1 is the correct one-hot dimension
        
        # Pass through embedding layer: (Batch_Size,) -> (Batch_Size, embedding_dim)
        product_embed = self.product_embedding(product_indices)
        
        # Get market_features from observations
        # This will be (Batch_Size, market_features_dim)
        market_features = observations["features"]

        # Concatenate product embedding and market features
        # (Batch_Size, embedding_dim) + (Batch_Size, market_features_dim)
        # -> (Batch_Size, embedding_dim + market_features_dim)
        return th.cat([product_embed, market_features], dim=1)