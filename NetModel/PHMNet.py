
import torch.nn.functional as F
from utils import init_weights, count_param
import torch
import torch.nn as nn
from thop import profile
import math

class unet2dConv2d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super().__init__()
        layers = []
        for i in range(n):
            conv = nn.Conv2d(in_size, out_size, ks, stride=stride, padding=padding)
            layers.append(conv)
            if is_batchnorm:
                layers.append(nn.BatchNorm2d(out_size))
            layers.append(nn.ReLU(inplace=True))
            in_size = out_size
        self.block = nn.Sequential(*layers)
        init_weights(self.block, init_type='kaiming')
    def forward(self, x):
        return self.block(x)
class SphericalCrossInteraction(nn.Module):

    def __init__(self, in_channels, hidden_dim=None, temperature=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_channels // 2
        self.temp = temperature
        self.proj_t1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_dec = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.reproj_t1 = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.reproj_dec = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        init_weights(self.proj_t1, init_type='kaiming')
        init_weights(self.proj_dec, init_type='kaiming')
        init_weights(self.reproj_t1, init_type='kaiming')
        init_weights(self.reproj_dec, init_type='kaiming')


    def _expmap(self, v):
        return F.normalize(v, dim=-1, eps=1e-8)


    def _logmap(self, x):
        return x

    def forward(self, feat_t1, feat_dec):
        B, C, H, W = feat_t1.shape

        z_t1 = self.proj_t1(feat_t1).flatten(2).transpose(1,2)   # [B,N,D]
        z_dec = self.proj_dec(feat_dec).flatten(2).transpose(1,2)

        s_t1 = self._expmap(z_t1)
        s_dec = self._expmap(z_dec)

        v_t1 = s_t1
        v_dec = s_dec

        sim = torch.bmm(v_t1, v_dec.transpose(1,2))          # [B,N,N]
        attn_t1_to_dec = F.softmax(sim / self.temp, dim=-1)
        attn_dec_to_t1 = F.softmax(sim.transpose(1,2) / self.temp, dim=-1)

        v_fused_t1 = torch.bmm(attn_t1_to_dec, v_dec)        # [B,N,D]
        v_fused_dec = torch.bmm(attn_dec_to_t1, v_t1)

        v_out_t1 = v_t1 + v_fused_t1
        v_out_dec = v_dec + v_fused_dec

        s_out_t1 = self._expmap(v_out_t1)
        s_out_dec = self._expmap(v_out_dec)

        euc_t1 = self.reproj_t1(s_out_t1.transpose(1,2).reshape(B, -1, H, W))
        euc_dec = self.reproj_dec(s_out_dec.transpose(1,2).reshape(B, -1, H, W))
        return euc_t1, euc_dec
class MagnitudePreservingPHOR(nn.Module):
    """
    保留DEC模长的符号不变射影方向表示。

    输入:
        dec: [B, 3, H, W]

    输出:
        phor_dec: [B, 7, H, W]

    PHOR(dec) = [m, m * phi(d)]

    其中:
        m = ||dec||_2
        d = dec / (m + eps)

        phi(d) = [
            dx^2,
            dy^2,
            dz^2,
            sqrt(2) * dx * dy,
            sqrt(2) * dx * dz,
            sqrt(2) * dy * dz
        ]

    满足:
        phi(d) = phi(-d)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.out_channels = 7

    def forward(self, dec):
        if dec.ndim != 4 or dec.shape[1] != 3:
            raise ValueError(
                "DEC input must have shape [B, 3, H, W], "
                f"but got {tuple(dec.shape)}"
            )

        # 清理NaN和Inf
        dec = torch.nan_to_num(
            dec,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # DEC模长，保留可能的FA/方向置信度信息
        magnitude = torch.norm(
            dec,
            p=2,
            dim=1,
            keepdim=True
        )  # [B,1,H,W]

        # 单位方向
        direction = F.normalize(
            dec,
            p=2,
            dim=1,
            eps=self.eps
        )  # [B,3,H,W]

        dx = direction[:, 0:1]
        dy = direction[:, 1:2]
        dz = direction[:, 2:3]

        sqrt_two = 2.0 ** 0.5

        # 六维射影超球面表示
        projective_orientation = torch.cat(
            [
                dx ** 2,
                dy ** 2,
                dz ** 2,
                sqrt_two * dx * dy,
                sqrt_two * dx * dz,
                sqrt_two * dy * dz
            ],
            dim=1
        )  # [B,6,H,W]

        # 将方向表示乘回模长，避免丢失DEC强度信息
        projective_orientation = (
            magnitude * projective_orientation
        )

        # 最终7通道：
        # 1通道模长 + 6通道射影方向
        phor_dec = torch.cat(
            [
                magnitude,
                projective_orientation
            ],
            dim=1
        )  # [B,7,H,W]

        return phor_dec
class HypersphericalPrototypeSegmentationHead(nn.Module):
    """
    Hyperspherical Prototype Segmentation Head (HPSH)

    使用多个可学习球面原型替代普通 Conv 1x1 分类器。

    Args:
        in_channels:
            最后一级解码特征通道数。

        num_classes:
            分割类别数。

        embedding_dim:
            球面嵌入维度，建议 32 或 64。
            不需要设置成 HCI 的 512。

        num_prototypes:
            每个类别的原型数量。
            二分类任务建议先使用 3 或 4。

        temperature:
            球面相似度温度。

        aggregation:
            多原型聚合方式：
            "logsumexp"：推荐，训练平滑；
            "max"：直接选最相似原型。
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        embedding_dim=32,
        num_prototypes=4,
        temperature=0.1,
        aggregation="logsumexp"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.aggregation = aggregation

        self.embedding = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embedding_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=1,
                bias=False
            )
        )

        # [类别数, 每类原型数, 球面维度]
        self.prototypes = nn.Parameter(
            torch.empty(
                num_classes,
                num_prototypes,
                embedding_dim
            )
        )

        nn.init.normal_(
            self.prototypes,
            mean=0.0,
            std=0.02
        )

    def get_normalized_prototypes(self):
        """
        将类别原型约束到单位超球面。
        """
        return F.normalize(
            self.prototypes,
            p=2,
            dim=-1,
            eps=1e-6
        )

    def forward(self, x, return_embedding=False):
        """
        Args:
            x:
                [B, C, H, W]

        Returns:
            logits:
                [B, num_classes, H, W]
        """

        B, _, H, W = x.shape

        embedding = self.embedding(x)

        # [B, D, H, W] -> 单位超球面
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
            eps=1e-6
        )

        prototypes = self.get_normalized_prototypes()

        # 每个像素与每个类别的多个原型计算余弦相似度
        #
        # embedding:
        #   [B, D, H, W]
        #
        # prototypes:
        #   [C, K, D]
        #
        # similarity:
        #   [B, C, K, H, W]
        similarity = torch.einsum(
            "bdhw,ckd->bckhw",
            embedding,
            prototypes
        )

        similarity = similarity / max(
            self.temperature,
            1e-6
        )

        if self.aggregation == "logsumexp":
            # 减去 log(K)，避免仅因原型数量增加而改变 logits 尺度
            logits = torch.logsumexp(
                similarity,
                dim=2
            )

            logits = logits - torch.log(
                torch.tensor(
                    float(self.num_prototypes),
                    device=logits.device,
                    dtype=logits.dtype
                )
            )

        elif self.aggregation == "max":
            logits = similarity.max(dim=2).values

        else:
            raise ValueError(
                f"Unsupported aggregation: "
                f"{self.aggregation}"
            )

        if return_embedding:
            return logits, embedding

        return logits
class PHMNet(nn.Module): # SphericalFusion_PHOR_HPSHNet
    def __init__(self, in_t1=1, in_dir=3, n_classes=5,
                 is_batchnorm=True, temperature=0.1, prototype_temperature=0.1, phor_eps=1e-6):
        super().__init__()
        filters_base = 64
        self.maxpool = nn.MaxPool2d(2)
        # PHOR本身没有可学习参数
        self.phor = MagnitudePreservingPHOR(
            eps=phor_eps
        )
        # ---- 编码器第一层 ----
        self.conv1_t1 = unet2dConv2d(1, filters_base, is_batchnorm)
        self.conv1_dec = unet2dConv2d(7, filters_base, is_batchnorm)

        # ---- 编码器第二层 ----
        self.conv2_t1 = unet2dConv2d(filters_base, filters_base*2, is_batchnorm)
        self.conv2_dec = unet2dConv2d(filters_base, filters_base*2, is_batchnorm)

        # ---- 编码器第三层 ----
        self.conv3_t1 = unet2dConv2d(filters_base*2, filters_base*4, is_batchnorm)
        self.conv3_dec = unet2dConv2d(filters_base*2, filters_base*4, is_batchnorm)
        self.interact3 = SphericalCrossInteraction(filters_base*4, hidden_dim=filters_base*8, temperature=temperature)

        # ---- 编码器第四层 ----
        self.conv4_t1 = unet2dConv2d(filters_base*4, filters_base*8, is_batchnorm)
        self.conv4_dec = unet2dConv2d(filters_base*4, filters_base*8, is_batchnorm)
        self.interact4 = SphericalCrossInteraction(filters_base*8, hidden_dim=filters_base*8, temperature=temperature)

        # ---- 编码器第五层（瓶颈） ----
        self.conv5_t1 = unet2dConv2d(filters_base*8, filters_base*16, is_batchnorm)
        self.conv5_dec = unet2dConv2d(filters_base*8, filters_base*16, is_batchnorm)
        self.interact5 = SphericalCrossInteraction(filters_base*16, hidden_dim=filters_base*8, temperature=temperature)

        self.fusion = unet2dConv2d(filters_base * 32, filters_base * 16, is_batchnorm)

        # ---- 解码器（与 SMOTNet 相同，但跳跃连接使用交互后的特征） ----
        self.up0 = nn.ConvTranspose2d(filters_base*16, filters_base*8, 2, 2)
        self.upconv0 = unet2dConv2d(filters_base*8*3, filters_base*8, is_batchnorm)

        self.up1 = nn.ConvTranspose2d(filters_base*8, filters_base*4, 2, 2)
        self.upconv1 = unet2dConv2d(filters_base*4*3, filters_base*4, is_batchnorm)

        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, 2, 2)
        self.upconv2 = unet2dConv2d(filters_base*2*3, filters_base*2, is_batchnorm)

        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, 2, 2)
        self.upconv3 = unet2dConv2d(filters_base*3, filters_base, is_batchnorm)

        self.final = HypersphericalPrototypeSegmentationHead(
            in_channels=filters_base,
            num_classes=n_classes,
            embedding_dim=32,
            num_prototypes=1,
            temperature=prototype_temperature,
            aggregation="logsumexp"
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs_t1, inputs_dec):
        # ========== 编码器（带交互） ==========
        # Layer 1
        c1_t1 = self.conv1_t1(inputs_t1)
        phor_dec = self.phor(inputs_dec)
        c1_dec = self.conv1_dec(phor_dec)
        p1_t1 = self.maxpool(c1_t1)
        p1_dec = self.maxpool(c1_dec)

        # Layer 2
        c2_t1 = self.conv2_t1(p1_t1)
        c2_dec = self.conv2_dec(p1_dec)
        p2_t1 = self.maxpool(c2_t1)
        p2_dec = self.maxpool(c2_dec)

        # Layer 3
        c3_t1 = self.conv3_t1(p2_t1)
        c3_dec = self.conv3_dec(p2_dec)
        c3_t1, c3_dec = self.interact3(c3_t1, c3_dec)
        p3_t1 = self.maxpool(c3_t1)
        p3_dec = self.maxpool(c3_dec)

        # Layer 4
        c4_t1 = self.conv4_t1(p3_t1)
        c4_dec = self.conv4_dec(p3_dec)
        c4_t1, c4_dec = self.interact4(c4_t1, c4_dec)
        p4_t1 = self.maxpool(c4_t1)
        p4_dec = self.maxpool(c4_dec)

        # Layer 5 (瓶颈，无池化)
        c5_t1 = self.conv5_t1(p4_t1)
        c5_dec = self.conv5_dec(p4_dec)
        c5_t1, c5_dec = self.interact5(c5_t1, c5_dec)

        fusion = torch.cat([c5_t1, c5_dec], 1)
        fusion_feat = self.fusion(fusion)

        # ========== 解码器 ==========
        up0 = self.up0(fusion_feat)                                 # [B,512,16,20]
        cat0 = torch.cat([up0, c4_t1, c4_dec], dim=1)               # 使用交互后的 c4 特征
        upcon0 = self.upconv0(cat0)                                 # [B,512,16,20]

        up1 = self.up1(upcon0)                                     # [B,256,32,40]
        cat1 = torch.cat([up1, c3_t1, c3_dec], dim=1)
        upcon1 = self.upconv1(cat1)                                # [B,256,32,40]

        up2 = self.up2(upcon1)                                     # [B,128,64,80]
        cat2 = torch.cat([up2, c2_t1, c2_dec], dim=1)
        upcon2 = self.upconv2(cat2)                                # [B,128,64,80]

        up3 = self.up3(upcon2)                                     # [B,64,128,160]
        cat3 = torch.cat([up3, c1_t1, c1_dec], dim=1)
        upcon3 = self.upconv3(cat3)                                # [B,64,128,160]

        seg_out = self.final(upcon3)                               # [B,5,128,160]
        return seg_out



if __name__ == '__main__':


    # 是否使用cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = PHMNet(1,3,5).to(device)


    x1 = torch.rand(32, 1, 128, 160).to(device)
    x2 = torch.rand(32, 3, 128, 160).to(device)

    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")
    y = model(x1, x2)
    param = count_param(model)  # 计算参数
    # print('Input shape:', x.shape)
    print('Output shape:', y.shape)
    print('UNet3d totoal parameters: %.2fM (%d)' % (param / 1e6, param))
