import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    # 计算注意力分数
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    # 计算注意力权重
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights


def multi_head_attention(Q, K, V, num_heads=8, mask=None):
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0
    depth = d_model // num_heads

    # 将输入划分为多个头
    Q_split = np.array_split(Q, num_heads, axis=-1)
    K_split = np.array_split(K, num_heads, axis=-1)
    V_split = np.array_split(V, num_heads, axis=-1)

    outputs = []
    attention_weights = []
    for i in range(num_heads):
        # 对每个头进行注意力计算
        out, attn_weights = scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i], mask)
        outputs.append(out)
        attention_weights.append(attn_weights)

    # 拼接所有头的输出
    outputs = np.concatenate(outputs, axis=-1)

    return outputs, attention_weights

def attention(seq_length,input_dim):
    X = np.random.rand(seq_length, input_dim)
    Q = X
    K = X
    V = X
    mask = None
    num_heads = 8
    outputs, attention_weights = multi_head_attention(Q, K, V, num_heads, mask)
    for i, attn_weights in enumerate(attention_weights):
        print(f"头 {i + 1} 的注意力权重:")
        print(attn_weights)
        print()

    print("多头注意力的输出:")
    print(outputs)

if __name__ == '__main__':
    attention(10,64)
