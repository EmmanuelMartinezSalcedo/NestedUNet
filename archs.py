import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['NestedUNet']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
    
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()
        
        assert dimension in [1, 2, 3]
        
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        
        # Función g: para computar features
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        # Función theta: para query
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        
        # Función phi: para key
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        # Función W: para la proyección final
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels) if bn_layer else nn.Identity()
        )
        
        # Inicializar W con ceros para que el bloque sea una identidad al inicio
        nn.init.constant_(self.W[0].weight, 0)
        if self.W[0].bias is not None:
            nn.init.constant_(self.W[0].bias, 0)
        
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # g_x: (N, inter_channels, H, W) -> (N, inter_channels, H*W)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (N, H*W, inter_channels)
        
        # theta_x: (N, inter_channels, H, W) -> (N, inter_channels, H*W)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # (N, H*W, inter_channels)
        
        # phi_x: (N, inter_channels, H, W) -> (N, inter_channels, H*W)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        # Attention
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        # (N, H*W, H*W) x (N, H*W, inter_channels) -> (N, H*W, inter_channels)
        y = torch.matmul(f_div_C, g_x)
        
        # Reshape
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        # Conexión residual
        z = W_y + x
        
        return z


class ImprovedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.non_local = NonLocalBlock(
            in_channels=out_channels,
            inter_channels=out_channels // 2,
            sub_sample=False,
            bn_layer=True
        )
        
    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.non_local(out)
        return out

class NestedUNet(nn.Module):
    def __init__(self, input_channels=1, deep_supervision=False, dropout_rate=0.3, **kwargs):
        super().__init__()

        # Filtros optimizados para 128x128 - incremento gradual más suave
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.dropout_rate = dropout_rate

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder (columna izquierda) - Ahora con 5 niveles para 128x128
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], dropout_rate=0.1)  # 128x128
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], dropout_rate=0.15)    # 64x64
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], dropout_rate=0.2)     # 32x32
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], dropout_rate=0.25)    # 16x16
        
        # Bottleneck mejorado para el nivel más profundo
        self.bottleneck = ImprovedBottleneck(nb_filter[3], nb_filter[4], dropout_rate=0.1)      # 8x8

        # Attention gates para todas las conexiones skip
        # Para X_1 column
        self.att_gate_3_1 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3]//2)
        self.att_gate_2_1 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2]//2)
        self.att_gate_1_1 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1]//2)
        self.att_gate_0_1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)
        
        # Para X_2 column
        self.att_gate_2_2 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2]//2)
        self.att_gate_1_2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1]//2)
        self.att_gate_0_2 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)
        
        # Para X_3 column
        self.att_gate_1_3 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1]//2)
        self.att_gate_0_3 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)
        
        # Para X_4 column
        self.att_gate_0_4 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)

        # Decoder blocks con estructura nested
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], dropout_rate=0.25)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate=0.2)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=0.15)
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=0.1)

        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate=0.2)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=0.15)
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=0.1)

        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=0.15)
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=0.1)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=0.05)

        # Output layers para segmentación binaria
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        # Encoder path - Ahora con 5 niveles para manejar 128x128
        x0_0 = self.conv0_0(input)           # 128x128 -> 32 channels
        x1_0 = self.conv1_0(self.pool(x0_0)) # 64x64 -> 64 channels
        x2_0 = self.conv2_0(self.pool(x1_0)) # 32x32 -> 128 channels
        x3_0 = self.conv3_0(self.pool(x2_0)) # 16x16 -> 256 channels

        # Bottleneck
        x4_0 = self.bottleneck(self.pool(x3_0)) # 8x8 -> 512 channels

        # Primera columna decoder (X_1)
        x4_0_up = F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)
        x3_0_att = self.att_gate_3_1(x3_0, x4_0_up)
        x3_1 = self.conv3_1(torch.cat([x3_0_att, x4_0_up], 1))

        x3_1_up = F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x2_0_att = self.att_gate_2_1(x2_0, x3_1_up)
        x2_1 = self.conv2_1(torch.cat([x2_0_att, x3_1_up], 1))

        x2_1_up = F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x1_0_att = self.att_gate_1_1(x1_0, x2_1_up)
        x1_1 = self.conv1_1(torch.cat([x1_0_att, x2_1_up], 1))

        x1_1_up = F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)
        x0_0_att_1 = self.att_gate_0_1(x0_0, x1_1_up)
        x0_1 = self.conv0_1(torch.cat([x0_0_att_1, x1_1_up], 1))

        # Segunda columna decoder (X_2)
        x3_1_up_2 = F.interpolate(x3_1, size=x2_1.shape[2:], mode='bilinear', align_corners=True)
        x2_0_att_2 = self.att_gate_2_2(x2_0, x3_1_up_2)
        x2_2 = self.conv2_2(torch.cat([x2_0_att_2, x2_1, x3_1_up_2], 1))

        x2_2_up = F.interpolate(x2_2, size=x1_1.shape[2:], mode='bilinear', align_corners=True)
        x1_0_att_2 = self.att_gate_1_2(x1_0, x2_2_up)
        x1_2 = self.conv1_2(torch.cat([x1_0_att_2, x1_1, x2_2_up], 1))

        x1_2_up = F.interpolate(x1_2, size=x0_1.shape[2:], mode='bilinear', align_corners=True)
        x0_0_att_2 = self.att_gate_0_2(x0_0, x1_2_up)
        x0_2 = self.conv0_2(torch.cat([x0_0_att_2, x0_1, x1_2_up], 1))

        # Tercera columna decoder (X_3)
        x2_2_up_3 = F.interpolate(x2_2, size=x1_2.shape[2:], mode='bilinear', align_corners=True)
        x1_0_att_3 = self.att_gate_1_3(x1_0, x2_2_up_3)
        x1_3 = self.conv1_3(torch.cat([x1_0_att_3, x1_1, x1_2, x2_2_up_3], 1))

        x1_3_up = F.interpolate(x1_3, size=x0_2.shape[2:], mode='bilinear', align_corners=True)
        x0_0_att_3 = self.att_gate_0_3(x0_0, x1_3_up)
        x0_3 = self.conv0_3(torch.cat([x0_0_att_3, x0_1, x0_2, x1_3_up], 1))

        # Cuarta columna decoder (X_4) - Salida final
        x1_3_up_4 = F.interpolate(x1_3, size=x0_3.shape[2:], mode='bilinear', align_corners=True)
        x0_0_att_4 = self.att_gate_0_4(x0_0, x1_3_up_4)
        x0_4 = self.conv0_4(torch.cat([x0_0_att_4, x0_1, x0_2, x0_3, x1_3_up_4], 1))

        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output