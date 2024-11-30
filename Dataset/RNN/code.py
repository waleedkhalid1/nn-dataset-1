import torch
import torch.nn as nn
from torch.utils.data import Dataset




class DatasetPreparation(Dataset):
    def __init__(self, text_data: str, seq_length: int = 25) -> None:
        self.chars = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)
        # useful way to fetch characters either by index or char
        self.idx_to_char = {i:ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch:i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(text_data)

    @property
    def X_string(self) -> str:
        return self.vector_to_string(self.X)

    def __len__(self) -> int:
        return int(len(self.X) / self.seq_length -1)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length

        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx+1:end_idx+1]).float()
        return X, y

    def string_to_vector(self, name: str) -> list[int]:
        vector = list()
        for s in name:
            vector.append(self.char_to_idx[s])
        return vector

    def vector_to_string(self, vector: list[int]) -> str:
        vector_string = ""
        for i in vector:
            vector_string += self.idx_to_char[i]
        return vector_string


class Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Входной тензор [batch_size, input_size]
        :param hidden_state: Скрытое состояние [batch_size, hidden_size]
        :return: Выход (логиты) [batch_size, output_size] и обновлённое скрытое состояние
        """
        x = self.i2h(x)  # [batch_size, hidden_size]
        hidden_state = self.h2h(hidden_state)  # [batch_size, hidden_size]
        hidden_state = torch.tanh(x + hidden_state)  # [batch_size, hidden_size]
        out = self.h2o(hidden_state)  # [batch_size, output_size]
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)