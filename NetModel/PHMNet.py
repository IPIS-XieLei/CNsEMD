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

        dec = torch.nan_to_num(
            dec,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        magnitude = torch.norm(
            dec,
            p=2,
            dim=1,
            keepdim=True
        )  # [B,1,H,W]

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

        projective_orientation = (
            magnitude * projective_orientation
        )

        phor_dec = torch.cat(
            [
                magnitude,
                projective_orientation
            ],
            dim=1
        )  # [B,7,H,W]

        return phor_dec
class HypersphericalPrototypeSegmentationHead(nn.Module):

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

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
            eps=1e-6
        )

        prototypes = self.get_normalized_prototypes()

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
class PHMNet(nn.Module):
    def __init__(self, in_t1=1, in_dir=3, n_classes=5,
                 is_batchnorm=True, temperature=0.1, prototype_temperature=0.1, phor_eps=1e-6, hidden_dim=512, num_prototypes=3):
        super().__init__()
        filters_base = 64
        self.maxpool = nn.MaxPool2d(2)

        self.phor = MagnitudePreservingPHOR(
            eps=phor_eps
        )

        self.conv1_t1 = unet2dConv2d(1, filters_base, is_batchnorm)
        self.conv1_dec = unet2dConv2d(7, filters_base, is_batchnorm)


        self.conv2_t1 = unet2dConv2d(filters_base, filters_base*2, is_batchnorm)
        self.conv2_dec = unet2dConv2d(filters_base, filters_base*2, is_batchnorm)


        self.conv3_t1 = unet2dConv2d(filters_base*2, filters_base*4, is_batchnorm)
        self.conv3_dec = unet2dConv2d(filters_base*2, filters_base*4, is_batchnorm)
        self.interact3 = SphericalCrossInteraction(filters_base*4, hidden_dim, temperature=temperature)


        self.conv4_t1 = unet2dConv2d(filters_base*4, filters_base*8, is_batchnorm)
        self.conv4_dec = unet2dConv2d(filters_base*4, filters_base*8, is_batchnorm)
        self.interact4 = SphericalCrossInteraction(filters_base*8, hidden_dim, temperature=temperature)


        self.conv5_t1 = unet2dConv2d(filters_base*8, filters_base*16, is_batchnorm)
        self.conv5_dec = unet2dConv2d(filters_base*8, filters_base*16, is_batchnorm)
        self.interact5 = SphericalCrossInteraction(filters_base*16, hidden_dim, temperature=temperature)

        self.fusion = unet2dConv2d(filters_base * 32, filters_base * 16, is_batchnorm)


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
            num_prototypes=num_prototypes,
            temperature=prototype_temperature,
            aggregation="logsumexp"
        )


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs_t1, inputs_dec):
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

        # Layer 5
        c5_t1 = self.conv5_t1(p4_t1)
        c5_dec = self.conv5_dec(p4_dec)
        c5_t1, c5_dec = self.interact5(c5_t1, c5_dec)

        fusion = torch.cat([c5_t1, c5_dec], 1)
        fusion_feat = self.fusion(fusion)


        up0 = self.up0(fusion_feat)
        cat0 = torch.cat([up0, c4_t1, c4_dec], dim=1)
        upcon0 = self.upconv0(cat0)

        up1 = self.up1(upcon0)
        cat1 = torch.cat([up1, c3_t1, c3_dec], dim=1)
        upcon1 = self.upconv1(cat1)

        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, c2_t1, c2_dec], dim=1)
        upcon2 = self.upconv2(cat2)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, c1_t1, c1_dec], dim=1)
        upcon3 = self.upconv3(cat3)

        seg_out = self.final(upcon3)
        return seg_out



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PHMNet(1, 3, 5).to(device)
    model.eval()

    x1 = torch.rand(1, 1, 128, 160, device=device)
    x2 = torch.rand(1, 3, 128, 160, device=device)

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(x1, x2),
            verbose=False
        )

    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"FLOPs (2 × MACs): {2 * macs / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.4f} M")
    print(f"Exact parameters: {sum(p.numel() for p in model.parameters()):,}")