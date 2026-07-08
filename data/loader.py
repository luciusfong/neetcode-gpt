import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        ix = torch.randint(len(data) - context_length, (batch_size,))
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        x = torch.stack([data[i:i + context_length] for i in ix])
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        return x, y
