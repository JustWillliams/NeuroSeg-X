import torch
import torch.nn as nn
import torch.nn.functional as F

class FAFRM(nn.Module):
    """
    Feature Augmentation - Refinement Module (FA-FRM)
    Refines features by emphasizing channel and spatial context.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        ca = x * self.channel_att(x)
        sa = ca * self.spatial_att(ca)
        return self.refine(sa) + x

class DSHCAT(nn.Module):
    """
    Dual-Scale Hybrid Cross-Attention Transformer (DSHCAT)
    Fuses features from dual scales using cross-attention.
    """
    def __init__(self, high_channels, low_channels):
        super().__init__()
        self.query = nn.Conv2d(high_channels, high_channels, 1)
        self.key = nn.Conv2d(low_channels, high_channels, 1)
        self.value = nn.Conv2d(low_channels, high_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(high_channels, high_channels, 1)

    def forward(self, high_feat, low_feat):
        # Resize low_feat to match high_feat if necessary
        if low_feat.shape[2:] != high_feat.shape[2:]:
            low_feat = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=True)
            
        b, c, h, w = high_feat.shape
        q = self.query(high_feat).view(b, c, -1).permute(0, 2, 1)  # B, N, C
        k = self.key(low_feat).view(b, c, -1)                      # B, C, N
        v = self.value(low_feat).view(b, c, -1).permute(0, 2, 1)   # B, N, C
        
        attn = self.softmax(torch.bmm(q, k) / (c ** 0.5))
        out = torch.bmm(attn, v).permute(0, 2, 1).view(b, c, h, w)
        return self.proj(out) + high_feat

class NeuroSegX(nn.Module):
    """
    NeuroSeg-X: CNN + Swin Transformer with Multi-task head.
    """
    def __init__(self, in_channels=3, seg_classes=4):
        super().__init__()
        # Encoder (Hybrid CNN-Swin style)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU())
        self.frm1 = FAFRM(64)
        self.pool = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.frm2 = FAFRM(128)
        
        self.cat1 = DSHCAT(128, 64)
        
        # Multi-task heads
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, seg_classes, 1)
        )
        
        self.det_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.grad_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # LGG / HGG
        )

    def forward(self, x):
        e1 = self.frm1(self.enc1(x))
        e2_pre = self.enc2(self.pool(e1))
        e2 = self.frm2(e2_pre)
        
        # Dual-scale fusion
        fused = self.cat1(e2, e1)
        
        seg = self.seg_head(fused)
        det = self.det_head(fused)
        grad = self.grad_head(fused)
        
        return {
            'segmentation': seg,
            'detection': det,
            'grading': grad
        }
