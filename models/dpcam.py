"""
DPCAM: Dynamic Position-aware Channel-Spatial Attention Module
改进版CBAM，增加位置编码和动态权重机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicChannelAttention(nn.Module):
    """动态通道注意力"""
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        hidden_dim = max(in_planes // ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 主通道注意力
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        )
        
        # 动态权重调制
        self.modulator = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg = self.avg_pool(x)
        max_val = self.max_pool(x)
        
        # 标准通道注意力
        avg_att = self.fc(avg)
        max_att = self.fc(max_val)
        att = avg_att + max_att
        
        # 动态调制
        modulation = self.modulator(avg)
        att = att * modulation
        
        return self.sigmoid(att)


class PositionAttention(nn.Module):
    """位置编码注意力"""
    def __init__(self, in_planes, max_size=256):
        super().__init__()
        pos_dim = max(in_planes // 4, 16)
        
        # 可学习位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, pos_dim, max_size, max_size) * 0.02)
        
        # 位置投影
        self.pos_proj = nn.Sequential(
            nn.Conv2d(pos_dim, in_planes // 2, 1),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 2, 1, 1)
        )
        
        # 特征投影
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, 1),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )
        
        self.interaction = nn.Conv2d(in_planes // 2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 插值位置编码
        pos_enc = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=True)
        pos_att = self.pos_proj(pos_enc)
        
        # 特征驱动的位置注意力
        feat = self.feat_proj(x)
        feat_pos_att = self.interaction(feat)
        
        # 融合
        pos_attention = pos_att + feat_pos_att
        return self.sigmoid(pos_attention)


class DynamicSpatialAttention(nn.Module):
    """动态多尺度空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv_3x3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv_5x5 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv_7x7 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        # 动态权重生成
        self.scale_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 多尺度统计
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg, max_val], dim=1)
        
        # 动态权重
        scale_weights = self.scale_weight_gen(spatial_feat)
        w1, w2, w3 = scale_weights[:, 0:1], scale_weights[:, 1:2], scale_weights[:, 2:3]
        
        # 多尺度注意力
        att_3 = self.conv_3x3(spatial_feat)
        att_5 = self.conv_5x5(spatial_feat)
        att_7 = self.conv_7x7(spatial_feat)
        
        # 融合
        out = w1 * att_3 + w2 * att_5 + w3 * att_7
        return self.sigmoid(out)


class DPCAM(nn.Module):
    """
    DPCAM: Dynamic Position-aware Channel-Spatial Attention Module
    """
    def __init__(self, in_planes, ratio=8, kernel_size=7, use_position=True):
        super().__init__()
        self.use_position = use_position
        
        self.channel_att = DynamicChannelAttention(in_planes, ratio)
        self.spatial_att = DynamicSpatialAttention(kernel_size)
        
        if use_position:
            self.position_att = PositionAttention(in_planes)
        
    def forward(self, x):
        # 串联应用
        ca = self.channel_att(x)
        x = x * ca
        
        sa = self.spatial_att(x)
        x = x * sa
        
        if self.use_position:
            pa = self.position_att(x)
            x = x * pa
        
        return x


class DPCAM_Lite(nn.Module):
    """DPCAM轻量版，接口与CBAM兼容"""
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.dpcam = DPCAM(in_planes, ratio, kernel_size, use_position=True)
    
    def forward(self, x):
        return self.dpcam(x)


class PA_Only(nn.Module):
    """仅位置注意力（PA-only），用于消融实验"""
    def __init__(self, in_planes):
        super().__init__()
        self.position_att = PositionAttention(in_planes)

    def forward(self, x):
        pa = self.position_att(x)
        return x * pa
