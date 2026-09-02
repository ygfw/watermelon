# Transformer 实现笔记

> 按模块拆开：Embedding → Positional Encoding → Multi-Head Attention → FFN → AddNorm → Encoder/Decoder Layer → Encoder/Decoder → Transformer。

需要用到的依赖：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 1. Embedding

Embedding 即把输入的词语转化为向量表示，使用了 `nn` 模块的 `Embedding` 类。

**什么是 Embedding**——Embedding 可以想象成是一个「查字典」的过程，是把整数词元索引转化为词向量的过程。在词向量化任务中，传统常用的方法是 onehot，但该方法占用内存过大，而 Embedding 则是一种更高效的方法，它将词元转化为稠密向量表示。

整数词元索引的原始输入形状为 `[batch_size, seq_len]`，例如 `[2, 4]` 表示输入了 2 句话，每句话包含 4 个词元。这里的每个词元都用一个整数索引来表示，具体的索引源自于词典。

Embedding 需要训练一个权重矩阵，其形状为 `[vocab_size, d_model]`。`vocab_size` 为词典大小（与输入的长度无关），`d_model` 为维度，是人为设定的超参数（类似于通道数量）。经过权重矩阵的计算，输出形状为 `[batch_size, seq_len, d_model]`。

总体而言，权重矩阵相当于一个训练好的向量词典，而 Embedding 就是通过查字典找到每个词的向量表示的过程。

```python
# Embedding层的实现
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        '''
        Args:
            vocab_size: 词典大小
            d_model: 维度
        '''
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        '''
        输入形状: [batch_size, seq_len] - 整数索引序列
        输出形状: [batch_size, seq_len, d_model] - 词嵌入向量

        例如:
        输入: torch.tensor([[1, 2, 3], [4, 5, 0]]) # [2, 3]
        输出: torch.tensor([[[0.1, 0.2, ...], ...]]) # [2, 3, 512]
        '''
        # 细节: 乘以根号d_model (参考论文)
        return self.embedding(x) * math.sqrt(self.d_model)
```

---

## 2. Positional Encoding

位置编码（Positional Encoding）是 Transformer 架构中用于注入序列中词语位置信息的一种方法。由于 Transformer 的自注意力机制本身不具备序列顺序信息，因此需要额外加入位置编码来让模型感知输入序列中词语的位置。

对于位置为 $pos$ 的词语，其位置编码向量的第 $2i$ 个维度和第 $2i+1$ 个维度分别由正弦和余弦函数给出：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- $pos$ 是词语在序列中的位置（从 0 开始）
- $i$ 是维度索引（从 0 开始，对应维度的一半，因为每个位置有两个函数分别处理偶数和奇数索引）
- $d_{model}$ 是模型的维度（即词嵌入的维度）

这样，每个位置都有一个唯一的、确定的且可被模型学习的位置编码向量。此外，正弦和余弦函数具有周期性，可以处理比训练时看到的序列更长的序列。

```python
class PositionalEncoding(nn.Module):
    '''位置编码'''
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        # 在参数数值大多固定的情况下，可以直接在这里定义默认数值，后续调用的时候不需要传入参数。
        # 有默认值的参数必须放在没有默认值的参数后面，否则会报错。
        '''
        Args:
            max_len: 序列的最大长度
            d_model: 模型维度
            dropout: 正则化系数
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)  # 正则化

        # 创建一个空的位置编码矩阵 形状是 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 分子部分 位置索引 生成一个从0到max_len-1的一维张量，形状为[max_len,]
        # unsqueeze(1)是再增加一维成为二维张量[max_len, 1],便于后续广播机制
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 分母部分
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 对偶数列用正弦函数 奇数列用余弦函数 用前面写好的空张量pe来选奇偶维度
        # pe形状为[行,列], pe[:,0::2]表示选择所有行，列从第0列开始到最后一列，步长为2，也就是所有偶数列，代入正弦函数计算
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个批次维度，形状变成[1, max_len, d_model]以便于后续的广播
        pe = pe.unsqueeze(0)

        # 将pe注册为缓冲区，成为模型的一部分但不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: embedding的词向量，形状为 [batch_size, seq_len, d_model]
        Return:
            加上位置编码后的张量，形状同x
        """
        # pe形状为[1, max_len, d_model], 在相加时要去除多余的长度，和seq_len保持一致即可
        # x.size(1)就是取x的第一维的形状 即seq_len
        # pe[:, :x.size(1)]表示 取所有的batch_size，取0到seq_len的行数和所有的d_model维度，最后一个切片可省略不写
        # 最终x的形状仍然为[batch_size, seq_len, d_model], 只是在embedding基础上多加了位置编码的信息
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

---

## 3. Multi-Head Self-Attention

```python
class MultiHeadAttention(nn.Module):
    '''多头注意力机制'''
    def __init__(self, d_model, num_heads, dropout=0.1):
        '''
        Args:
            d_model: 维度
            num_heads: 头的数量 原文中有8个头
            dropout: 正则化系数
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度 512/8 = 64

        # 通过线性变换，将输入的变量映射到Q、K、V三个矩阵中，每个线性变换都有不同的权重，即矩阵包含不同的信息
        # 输入维度和输出维度均为d_model，形状为[batch_size, seq_len, d_model]
        self.w_q = nn.Linear(d_model, d_model)  # Q
        self.w_k = nn.Linear(d_model, d_model)  # K
        self.w_v = nn.Linear(d_model, d_model)  # V
        self.w_o = nn.Linear(d_model, d_model)  # 输出矩阵

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        '''
        Args:
            query: [batch_size, q_seq_len, d_model]
            key: [batch_size, k_seq_len, d_model]
            value: [batch_size, v_seq_len, d_model]
            mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        '''
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        v_seq_len = value.size(1)

        # 首先传入query经过w_q的线性变化，view()用于重塑张量的结构，先把d_model拆成num_heads*d_k，再用transpose调换顺序
        # 如果保持形状为 [batch_size, seq_len, num_heads, d_k], 那么对于每个头，需要从第三个维度中提取数据，这会导致内存访问不连续
        # 而且无法直接使用矩阵乘法同时计算所有头的注意力，所以需要transpose调换顺序
        # 最终QKV的形状是: [batch_size, num_heads, q_seq_len, d_k]
        Q = self.w_q(query).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        '''
        注意力计算流程: 1. 注意力分数 2.应用掩码(可选) 3. 计算权重 4. 加权求和
        '''

        # 计算注意力分数
        # score的形状是[batch_size, num_heads, q_seq_len, k_seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 如果有mask的话 把一部分score替换成极小数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重 在score最后一个维度上应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 注意力权重矩阵的形状是[batch_size, num_heads, q_seq_len, k_seq_len] 其中最后一列已经softmax归一化
        # V的形状是[batch_size, num_heads, k_seq_len, d_k] (注意k_seq_len 必须等于 v_seq_len)
        # 注意力权重和V的后两个维度相乘 得到[batch_size, num_heads, q_seq_len, d_k]
        context = torch.matmul(attention_weights, V)

        # 重塑回原始形状: [batch_size, q_seq_len, d_model]
        # contiguous()确保内存连续存储
        context = context.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )

        # 最终线性变换
        output = self.w_o(context)

        return output, attention_weights
```

---

## 4. PositionwiseFeedForward

`PositionwiseFeedForward` 是 Transformer 中另一个重要组件。它作用在每个位置的特征上，独立地做相同的全连接变换，目的是对每个位置的表示做非线性变换，增加模型表达能力。

具体由两层线性变换中间夹一层 ReLU 组成。第一层把维度从 `d_model` 扩到 `d_ff`（通常 `d_ff` 更大，例如 `d_ff=2048`，`d_model=512`），ReLU 之后第二层再从 `d_ff` 投影回 `d_model`。

$$
\mathrm{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2
$$

输入和输出形状都是 `[batch_size, seq_len, d_model]`，对每个位置的 `d_model` 向量分别处理。因为各位置变换相互独立，所以叫 position-wise。每个位置经过两次线性变换和一次非线性激活，可以学到更复杂的特征。

为什么还需要前馈网络？注意力主要捕捉序列中不同位置之间的关系，前馈网络则对单个位置自己的特征做更深的处理。

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        Args:
            d_model: 维度
            d_ff: 前馈网络的隐藏层维度
            dropout: 正则化系数
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        '''
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        Return:
            输出张量，形状同 x
        '''
        return self.net(x)
```

---

## 5. AddNorm

残差连接和层归一化，放在注意力机制或前馈网络之后。

```python
class AddNorm(nn.Module):
    '''残差连接和层归一化'''
    def __init__(self, d_model, dropout=0.1):
        '''
        Args:
            d_model: 维度
            dropout: 正则化系数
        '''
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)  # 层归一化 直接使用nn的内部实现
        self.dropout = nn.Dropout(dropout)  # 正则化

    def forward(self, x, sublayer):
        '''
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            sublayer: 子层函数，例如注意力机制或前馈网络的输出
        Return:
            输出张量，形状同x
        '''
        # 残差连接 + 层归一化
        return self.norm(x + self.dropout(sublayer))
```

---

## 6. EncoderLayer

每个 EncoderLayer 都包含：MultiHeadAttention → AddNorm1 → FeedForward → AddNorm2

```python
class EncoderLayer(nn.Module):
    '''编码器层'''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        '''
        Args:
            d_model: 维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: 正则化系数
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 多头自注意力机制
        self.add_norm1 = AddNorm(d_model, dropout)  # 残差连接和层归一化1
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈神经网络
        self.add_norm2 = AddNorm(d_model, dropout)  # 残差连接和层归一化2

    def forward(self, x, mask=None):
        '''
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 掩码张量，形状为 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        Return:
            输出张量，形状同x
        '''
        # 多头自注意力机制 + 残差连接和层归一化
        # MultiHeadAttention模块的输出output和weight, _表示忽略第二个输出，只保留output
        # QKV接收同一个输入x
        attn_output, _ = self.self_attn(x, x, x, mask)

        # Attention后连接AddNorm, AddNorm分别接收原始输入x和注意力机制的输出attn_output
        x = self.add_norm1(x, attn_output)

        # 前馈神经网络 + 残差连接和层归一化
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x
```

---

## 7. DecoderLayer

每个 DecoderLayer 都包含：MaskedMultiHeadAttention → AddNorm1 → CrossAttention → AddNorm2 → FeedForward → AddNorm3

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        '''
        Args:
            d_model: 维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
        '''
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        '''
        Args:
            x: 输入张量，形状为 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出张量，形状为 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码张量，形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量，形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        '''
        # 掩码多头自注意力机制 + 残差连接和层归一化
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, attn_output)

        # 交叉注意力机制中的KV来自encoder计算输出，Q来自decoder
        attn_output, _ = self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.add_norm2(x, attn_output)

        # 前馈神经网络 + 残差连接和层归一化
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)
        return x
```

---

## 8. Encoder

```python
class Encoder(nn.Module):
    '''编码器'''
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()

        # 初始化时传入参数并存储
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        '''
        Args:
            src: 源序列张量, 形状为 [batch_size, src_seq_len]
            src_mask: 源序列掩码张量, 形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
        Return:
            编码器输出张量, 形状为 [batch_size, src_seq_len, d_model]
        '''
        # 输入嵌入和位置编码
        x = self.embedding(src)  # [batch_size, src_seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, src_seq_len, d_model]
        x = self.dropout(x)

        # 通过每一层编码器层
        # 只需要传入动态变化的输入和掩码
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
```

---

## 9. Decoder

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()

        # 初始化时传入参数存储
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 输出线性层
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        '''
        Args:
            tgt: 目标序列张量，形状为 [batch_size, tgt_seq_len]，在训练阶段tgt是完整的目标序列，推理阶段是逐步生成的部分序列
            encoder_output: 编码器输出张量，形状为 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码张量，形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量，形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        Return:
            解码器输出张量，形状为 [batch_size, tgt_seq_len, vocab_size]
        '''
        # 输入嵌入和位置编码
        x = self.embedding(tgt)  # [batch_size, tgt_seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, tgt_seq_len, d_model]
        x = self.dropout(x)

        # 通过每一层解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # 输出线性层
        x = self.output_linear(x)  # [batch_size, tgt_seq_len, vocab_size]
        return x
```

---

## 10. Transformer

```python
class Transformer(nn.Module):
    '''完整的Transformer模型'''
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        '''
        Args:
            src: 源序列张量，形状为 [batch_size, src_seq_len]
            tgt: 目标序列张量，形状为 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码张量，形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量，形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        Return:
            Transformer输出张量，形状为 [batch_size, tgt_seq_len, vocab_size]
        '''
        # 编码器输出
        encoder_output = self.encoder(src, src_mask)

        # 解码器输出
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return decoder_output
```
