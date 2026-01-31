import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


from torch.utils.cpp_extension import load_inline
from model.pair_wise_distance_cuda_source import source

print("compile cuda source of sip'function...")
pair_wise_distance_cuda = load_inline(
    "pair_wise_distance", cpp_sources="", cuda_sources=source
)
print("done")


class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, pixel_features, spixel_features, init_spixel_indices, num_spixels_width, num_spixels_height):
        """
        前向传播计算颜色距离和对比度距离矩阵

        参数:
            self: 上下文对象，用于保存反向传播所需张量
            pixel_features: (B, C, H*W) 像素特征
            spixel_features: (B, C, num_spixels) 超像素特征
            init_spixel_indices: (B, H*W) 初始超像素索引
            num_spixels_width: 超像素网格宽度
            num_spixels_height: 超像素网格高度

        返回:
            color_dist: (B, 9, H*W) 颜色距离矩阵
            contrast_dist: (B, 9, H*W) 对比度距离矩阵
        """
        self.num_spixels_width = num_spixels_width
        self.num_spixels_height = num_spixels_height

        # 初始化输出张量
        B, C, _ = pixel_features.shape

        # 保存反向传播所需张量
        self.save_for_backward(pixel_features, spixel_features, init_spixel_indices)

        # 调用CUDA内核
        color_dist, contrast_dist = pair_wise_distance_cuda.forward(
            pixel_features.contiguous(),
            spixel_features.contiguous(),
            init_spixel_indices.contiguous(),
            num_spixels_width,
            num_spixels_height
        )

        return color_dist, contrast_dist

    @staticmethod
    def backward(self, color_dist_grad, contrast_dist_grad):
        """
        反向传播计算梯度

        参数:
            self: 上下文对象，包含保存的张量
            color_dist_grad: (B, 9, H*W) 颜色距离梯度
            contrast_dist_grad: (B, 9, H*W) 对比度距离梯度

        返回:
            pixel_features_grad: (B, C, H*W) 像素特征梯度
            spixel_features_grad: (B, C, num_spixels) 超像素特征梯度
            None: 对init_spixel_indices的梯度(不需要)
            None: 对num_spixels_width的梯度(不需要)
            None: 对num_spixels_height的梯度(不需要)
        """
        pixel_features, spixel_features, init_spixel_indices = self.saved_tensors

        # 初始化梯度张量
        pixel_features_grad = torch.zeros_like(pixel_features)
        spixel_features_grad = torch.zeros_like(spixel_features)

        # 调用CUDA内核计算梯度
        pixel_features_grad, spixel_features_grad = pair_wise_distance_cuda.backward(
            color_dist_grad.contiguous(),
            contrast_dist_grad.contiguous(),
            pixel_features.contiguous(),
            spixel_features.contiguous(),
            init_spixel_indices.contiguous(),
            pixel_features_grad,
            spixel_features_grad,
            self.num_spixels_width,
            self.num_spixels_height
        )

        return pixel_features_grad, spixel_features_grad, None, None, None

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm = norm_layer(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))


def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


def ssn_iter(pixel_features, pixel_contrast_features, stoken_size=[16, 16], n_iter=2):
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    num_spixels_height = height // sheight
    num_spixels_width = width // swidth
    num_spixels = num_spixels_height * num_spixels_width

    combined_features = torch.cat([pixel_features, pixel_contrast_features], dim=1)

    spixel_features, init_label_map = \
        calc_init_centroid(combined_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    combined_features = combined_features.reshape(*combined_features.shape[:2], -1)
    permuted_pixel_features = combined_features.permute(0, 2, 1).contiguous()

    mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)

    with torch.no_grad():
        for k in range(n_iter):
            if k < n_iter - 1:
                color_dist, contrast_dist = PairwiseDistFunction.apply(
                    combined_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)
                dist_matrix = color_dist + 10 * contrast_dist

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                    / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()
            else:
                color_dist, contrast_dist = PairwiseDistFunction.apply(
                    combined_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)
                dist_matrix = color_dist + 10 * contrast_dist

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()

    return abs_affinity, num_spixels


class SIP(nn.Module):
    def __init__(self, n_iter=2):
        super().__init__()
        
        self.n_iter = n_iter
                
    def forward(self, x, x_contrast, stoken_size):
        soft_association, num_spixels = ssn_iter(x, x_contrast, stoken_size, self.n_iter)
        return soft_association, num_spixels


class SGCA(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        # 投影层（Q/K/V/超像素）
        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.sp = nn.Linear(dim, qk_dim, bias=qkv_bias)

        # LEPE卷积和动态残差
        self.get_v_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分离卷积
        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),  # 分组卷积适配多头
            nn.GELU(),
            nn.Conv1d(dim, dim * 9, kernel_size=1, groups=num_heads)  # 生成3x3卷积核权重
        )

        # 归一化和缩放因子
        self.norm = LayerNorm2d(dim)
        self.scale = (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def get_lepe(self, v):
        """位置增强卷积（直接作用于全局特征）"""
        B, C, H, W = v.shape
        lepe = self.get_v_conv(v).reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        return lepe

    def forward_stoken(self, x, affinity_matrix):
        """超像素聚合"""
        x = rearrange(x, 'b c h w -> b (h w) c')
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16)
        return stokens  # (B, num_spixels, C)

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        x = self.norm(x)

        # generate superpixel stoken
        stoken = self.forward_stoken(x, affinity_matrix) # b, k, c

        # stoken projection
        stoken = self.sp(stoken).permute(0,2,1).reshape(B, self.num_heads, self.qk_dim // self.num_heads, num_spixels) # B, H, C, hh*ww

        # q, k, v projection
        q = self.q(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        k = self.k(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        v = self.v(x).reshape(B, self.num_heads, self.dim // self.num_heads, H*W) # B, H, C, N
        lepe = self.get_lepe(self.v(x))  # (B, H*W, C)

        s_attn = k.transpose(-2, -1) @ stoken * self.scale # B, H, H*W, hh*ww
        s_attn = self.attn_drop(F.softmax(s_attn, -2))
        s_out = (v @ s_attn) # B, H, C, hh*ww

        x_attn = stoken.transpose(-2, -1) @ q * self.scale
        x_attn = self.attn_drop(F.softmax(x_attn, -2))
        x_out = (s_out @ x_attn).reshape(B, C, H, W)
        x_out = x_out + lepe.reshape(B, C, H, W)
        return x_out

class DFFFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = 1

        # 频域处理组件
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                             padding=1, groups=in_channels * 2, bias=True)

        # 动态权重生成
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.groups, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 频域通道混合
        self.fdc = nn.Conv2d(in_channels * 2, out_channels * 2 * self.groups,
                             kernel_size=1, groups=self.groups, bias=True)

        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch, c, h, w = x.shape

        # 1. 频域转换
        ffted = torch.fft.rfft2(x, norm='ortho')  # (b,c,h,w//2+1)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (b,c,h,w//2+1,2)

        # 2. 频域特征处理
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w')
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted  # 残差连接

        # 3. 动态权重混合
        dy_weight = self.weight(ffted).unsqueeze(2)  # (b,g,1,h,w//2+1)
        ffted = self.fdc(ffted).view(batch, self.groups, -1, h, w // 2 + 1)
        ffted = torch.einsum('bgchl,bgohl->bchl', ffted, dy_weight)

        # 4. 逆变换恢复
        ffted = F.gelu(ffted)
        ffted = ffted.view(batch, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2)
        output = torch.fft.irfft2(
            torch.view_as_complex(ffted.contiguous()),
            s=(h, w), norm='ortho'
        )

        x = self.x_conv(x)

        return x + output


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Block(nn.Module):
    def __init__(self, dim, out_ch, features, layer_num, stoken_size, heads, qk_dim, mlp_dim):
        super(Block, self).__init__()
        self.layer_num = layer_num
        self.stoken_size = stoken_size
        self.contrast = Conv2d_cd(dim, 1)
        self.sip = SIP(1)
        self.sgca = SGCA(dim, heads, qk_dim)
        self.feedforward = DFFFN(dim, out_ch)


    def forward(self, x):
        x_contrast = self.contrast(x)
        affinity_matrix, num_spixels = self.sip(x, x_contrast, self.stoken_size)

        x = self.sgca(x, affinity_matrix, num_spixels) + x

        out = self.feedforward(x)

        return out

