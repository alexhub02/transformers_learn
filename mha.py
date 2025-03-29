import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(input_dim, num_heads * head_dim)

    def forward(self, hidden_states):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        return query_states, key_states, value_states

# 创建一个示例模型
input_dim = 256
num_heads = 8
head_dim = 32
model = MultiHeadAttention(input_dim, num_heads, head_dim)

# 创建一个示例的隐藏状态张量
batch_size = 32
seq_length = 10
hidden_states = torch.randn(batch_size, seq_length, input_dim)

# 处理隐藏状态
query, key, value = model(hidden_states)
print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")