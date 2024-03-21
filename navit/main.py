```python
from functools import partial  # 导入partial函数，用于固定函数的部分参数，生成新的函数
from typing import List, Union  # 导入类型注解

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数接口
from einops import rearrange, repeat  # 导入einops库，用于更灵活的张量操作
from torch import Tensor, nn  # 导入PyTorch的Tensor和神经网络模块
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence  # 导入RNN的序列填充函数

#helpers

def exists(val):  # 定义exists函数，用于判断值是否存在
    return val is not None

def default(val, d):  # 定义default函数，如果值存在则返回该值，否则返回默认值
    return val if exists(val) else d

def always(val):  # 定义always函数，返回一个始终返回特定值的函数
    return lambda *args: val

def pair(t):  # 定义pair函数，如果输入是元组则直接返回，否则返回一个元素重复的元组
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):  # 定义divisible_by函数，判断一个数是否能被另一个数整除
    return (numer % denom) == 0

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:  # 定义group_images_by_max_seq_len函数，根据最大序列长度对图像进行分组

    calc_token_dropout = default(calc_token_dropout, always(0.))  # 如果没有指定calc_token_dropout，则默认为0

    groups = []  # 初始化groups列表，用于存储分组后的图像
    group = []  # 初始化group列表，用于存储当前分组的图像
    seq_len = 0  # 初始化seq_len，用于记录当前分组的序列长度

    if isinstance(calc_token_dropout, (float, int)):  # 如果calc_token_dropout是浮点数或整数，则将其转换为函数
        calc_token_dropout = always(calc_token_dropout)

    for image in images:  # 遍历图像列表
        assert isinstance(image, Tensor)  # 断言图像是Tensor类型

        image_dims = image.shape[-2:]  # 获取图像的维度
        ph, pw = map(lambda t: t // patch_size, image_dims)  # 计算图像的高度和宽度的块数

        image_seq_len = (ph * pw)  # 计算图像的序列长度
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))  # 计算经过dropout后的序列长度

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'  # 断言序列长度不超过最大序列长度

        if (seq_len + image_seq_len) > max_seq_len:  # 如果当前分组的序列长度加上新图像的序列长度超过最大序列长度，则将当前分组添加到groups列表，并开始新的分组
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)  # 将图像添加到当前分组
        seq_len += image_seq_len  # 更新当前分组的序列长度

    if len(group) > 0:  # 如果最后一个分组不为空，则将其添加到groups列表
        groups.append(group)

    return groups  # 返回分组后的图像列表

#normalization
#they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Module):  # 定义LayerNorm类，实现无偏置的层归一化
    def __init__(self, dim):  # 初始化函数，接收输入维度作为参数
        super().__init__()  # 调用父类的初始化函数
        self.gamma = nn.Parameter(torch.ones(dim))  # 初始化gamma参数，用于缩放
        self.register_buffer('beta', torch.zeros(dim))  # 初始化beta参数，用于偏移

    def forward(self, x):  # 前向传播函数
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)  # 对输入进行层归一化

#they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):  # 定义RMSNorm类，实现查询-键归一化
    def __init__(self, heads, dim):  # 初始化函数，接收头数和输入维度作为参数
        super().__init__()  # 调用父类的初始化函数
        self.scale = dim ** 0.5  # 计算缩放因子
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))  # 初始化gamma参数，用于缩放

    def forward(self, x):  # 前向传播函数
        normed = F.normalize(x, dim = -1)  # 对输入进行归一化
        return normed * self.scale * self.gamma  # 返回缩放后的结果

#feedforward
def FeedForward(dim, hidden_dim, dropout = 0.):  # 定义FeedForward函数，实现前馈网络
    return nn.Sequential(  # 返回一个序列模型
        LayerNorm(dim),  # 层归一化
        nn.Linear(dim, hidden_dim),  # 线性变换
        nn.GELU(),  # GELU激活函数
        nn.Dropout(dropout),  # Dropout层
        nn.Linear(hidden_dim, dim),  # 线性变换
        nn.Dropout(dropout)  # Dropout层
    )

#attention
class Attention(nn.Module):  # 定义Attention类，实现注意力机制
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):  # 初始化函数，接收输入维度、头数、头维度和dropout率作为参数
        super().__init__()  # 调用父类的初始化函数
        inner_dim = dim_head *  heads  # 计算内部维度
        self.heads = heads  # 保存头数
        self.norm = LayerNorm(dim)  # 层归一化

        self.q_norm = RMSNorm(heads, dim_head)  # 查询归一化
        self.k_norm = RMSNorm(heads, dim_head)  # 键归一化

        self.attend = nn.Softmax(dim = -1)  # Softmax函数，用于计算注意力权重
        self.dropout = nn.Dropout(dropout) # Dropout层

        self.to_q = nn.Linear(dim, inner_dim, bias = False) # 线性变换，用于生成查询
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False) # 线性变换，用于生成键值对

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        ) # 线性变换，用于生成输出

    # forward function of Attention 意思是注意力机制的前向传播函数
    def forward(
        self,
        x, # 输入张量
        context = None, # 上下文
        mask = None,# key padding mask 键填充掩码
        attn_mask = None # attention mask 意思是注意力掩码
    ):
        x = self.norm(x)
        kv_input = default(context, x) # 如果没有指定context，则使用x作为键值对输入

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)) # 生成查询、键和值

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv
        ) # 重排查询、键和值的维度

        q = self.q_norm(q) # 查询归一化
        k = self.k_norm(k) # 键归一化

        dots = torch.matmul(q, k.transpose(-1, -2)) # 计算点积

        if exists(mask): # 如果存在mask，则使用mask
            mask = rearrange(mask, 'b j -> b 1 1 j') # 重排mask的维度
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max) # 使用mask填充

        if exists(attn_mask): # 如果存在attn_mask，则使用attn_mask
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max) # 使用attn_mask填充

        attn = self.attend(dots) # 计算注意力权重
        attn = self.dropout(attn) # Dropout层

        out = torch.matmul(attn, v) # 计算加权和
        out = rearrange(out, 'b h n d -> b n (h d)') # 重排输出的维度
        return self.to_out(out)  # 返回输出

#transformer block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([]) # 初始化层列表
        for _ in range(depth): # 循环depth次
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), # 添加注意力机制
                FeedForward(dim, mlp_dim, dropout = dropout) # 添加注意力机制和前馈网络
            ])) # 添加注意力机制和前馈网络

        self.norm = LayerNorm(dim) # 层归一化

    def forward(
        self,
        x, # 输入张量
        mask = None, # key padding mask 键填充掩码
        attn_mask = None # attention mask 意思是注意力掩码
    ):
        for attn, ff in self.layers: # 遍历层列表
            x = attn(x, mask = mask, attn_mask = attn_mask) + x # 注意力机制
            x = ff(x) + x # 前馈网络

        return self.norm(x) # 层归一化

# 从第114行开始添加中文注释
class NaViT(nn.Module):  # 定义NaViT类，继承自nn.Module
    def __init__(
        self,
        *,
        image_size,  # 图像尺寸
        patch_size,  # 块尺寸
        num_classes,  # 类别数量
        dim,  # 维度
        depth,  # 深度
        heads,  # 头数
        mlp_dim,  # MLP维度
        channels = 3,  # 通道数，默认为3
        dim_head = 64,  # 头维度，默认为64
        dropout = 0.,  # dropout率，默认为0
        emb_dropout = 0.,  # 嵌入dropout率，默认为0
        token_dropout_prob = None  # token dropout概率，默认为None
    ):
        super().__init__()  # 调用父类的初始化函数
        image_height, image_width = pair(image_size)  # 获取图像的高度和宽度

        # 计算token的dropout概率
        # 如果token_dropout_prob是函数，则直接使用
        # 如果token_dropout_prob是浮点数或整数，则将其转换为函数
        self.calc_token_dropout = None
        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob
        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # 计算与块相关的参数
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'
        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels  # 保存通道数
        self.patch_size = patch_size  # 保存块尺寸

        # 将块转换为嵌入
        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),  # 层归一化
            nn.Linear(patch_dim, dim),  # 线性变换
            LayerNorm(dim),  # 层归一化
        )

        # 初始化位置嵌入
        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        # 初始化dropout层
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 初始化最后的注意力池化查询
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # 输出到logits
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_classes, bias = False)  # 线性变换
        )

    @property
    def device(self):
        return next(self.parameters()).device # 获取模型参数所在的设备

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], #assume different resolution images already grouped correctly # 批量图像
        group_images = False, # 是否对图像进行分组，默认为False
        group_max_seq_len = 2048 # 最大序列长度，默认为2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout) # 获取块尺寸、通道数、设备和是否有token dropout

        arange = partial(torch.arange, device = device) # 创建一个arange函数，用于生成等差数列
        pad_sequence = partial(orig_pad_sequence, batch_first = True) # 创建一个pad_sequence函数，用于填充序列

        # auto pack if specified
        # 如果指定了自动打包，则对图像进行分组
        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images, # 图像列表
                patch_size = self.patch_size, # 块尺寸
                calc_token_dropout = self.calc_token_dropout, # 计算token的dropout概率
                max_seq_len = group_max_seq_len # 最大序列长度
            )

        #process images into variable lengthed sequences with attention mask

        num_images = [] # 初始化num_images列表
        batched_sequences = [] # 初始化batched_sequences列表
        batched_positions = [] # 初始化batched_positions列表
        batched_image_ids = [] # 初始化batched_image_ids列表

        for images in batched_images:
            num_images.append(len(images)) # 记录图像数量

            sequences = [] # 初始化sequences列表
            positions = [] # 初始化positions列表
            image_ids = torch.empty((0,), device = device, dtype = torch.long)  # 初始化image_ids张量

            for image_id, image in enumerate(images): # 遍历图像列表
                assert image.ndim ==3 and image.shape[0] == c # 断言图像的维度和通道数
                image_dims = image.shape[-2:] # 获取图像的维度
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}' # 断言图像的高度和宽度能被块尺寸整除
                # 计算块的高度和宽度
                ph, pw = map(lambda dim: dim // p, image_dims)
                # 生成块的位置
                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)
                # 重排块的维度
                pos = rearrange(pos, 'h w c -> (h w) c')
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)
                # 重排块的维度
                seq_len = seq.shape[-2]
                # 如果有token dropout，则计算token dropout
                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims) # 计算token的dropout概率
                    num_keep = max(1, int(seq_len * (1 - token_dropout))) # 计算保留的token数量
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices # 生成保留的token索引

                    seq = seq[keep_indices] # 保留token
                    pos = pos[keep_indices] # 保留位置

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id) # 填充image_ids
                sequences.append(seq) # 添加块到sequences列表
                positions.append(pos) # 添加位置到positions列表

            batched_image_ids.append(image_ids) # 添加image_ids到batched_image_ids列表
            batched_sequences.append(torch.cat(sequences, dim = 0)) # 添加sequences到batched_sequences列表
            batched_positions.append(torch.cat(positions, dim = 0)) # 添加positions到batched_positions列表

        #derive key padding mask from sequence lengths and pad to max length of the batch for attention mask calculation # 从序列长度计算键填充掩码

        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)  # 计算键填充掩码
        max_length = arange(lengths.amax().item()) # 计算最大长度
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n') # 生成键填充掩码

        #derive attention mask, and combine with key padding mask from above # 生成注意力掩码，并与上面的键填充掩码结合

        batched_image_ids = pad_sequence(batched_image_ids) # 填充batched_image_ids
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j') # 生成注意力掩码
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j') # 结合键填充掩码

        #combine patched images as well as the patched width / height positions for 2d positional embedding # 生成2D位置嵌入

        patches = pad_sequence(batched_sequences) # 填充块
        patch_positions = pad_sequence(batched_positions) # 填充位置

        #need to know how many images for final attention pooling # 需要知道最终注意力池化的图像数量

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)  # 计算图像数量

        #to patches and add positional embedding # 生成块并添加位置嵌入

        x = self.to_patch_embedding(patches)  # 嵌入块

        #factorized 2d absolute positional embedding # 2D绝对位置嵌入

        h_indices, w_indices = patch_positions.unbind(dim = -1) # 解绑位置

        h_pos = self.pos_embed_height[h_indices] # 获取高度位置嵌入
        w_pos = self.pos_embed_width[w_indices] # 获取宽度位置嵌入

        x = x + h_pos + w_pos # 添加位置嵌入

        #embed dropout # 嵌入dropout

        x = self.dropout(x) # Dropout层

        #attention layers # 注意力层

        x = self.transformer(x, attn_mask = attn_mask) # Transformer层

        #do attention pooling at the end # 最终的注意力池化

        max_queries = num_images.amax().item() # 计算最大查询数

        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0]) # 生成查询

        #attention pool mask # 注意力池化掩码

        image_id_arange = arange(max_queries) # 生成等差数列

        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j') # 生成注意力池化掩码

        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j') # 结合键填充掩码

        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j') # 重排掩码的维度

        #attention pool # 注意力池化

        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries # 注意力池化

        x = rearrange(x, 'b n d -> (b n) d') # 重排输出的维度

        #each batch element may not have same amount of images as the other, so we need to project out to logits # 每个批次元素的图像数量可能不同，因此需要投影到logits

        is_images = image_id_arange < rearrange(num_images, 'b -> b 1') # 生成图像掩码
        is_images = rearrange(is_images, 'b n -> (b n)') # 重排掩码的维度

        x = x[is_images] # 投影到logits

        #project out to logits # 投影到logits

        x = self.to_latent(x) # 投影到logits

        return self.mlp_head(x) # 返回logits logits是指分类器的输出