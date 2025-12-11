import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['UNet', 'NestedUNet', 'ImprovedNestedUNet']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        if self.dropout:
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
        # x: características del encoder (skip connection)
        # g: características del decoder (gating signal)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Ajustar dimensiones espaciales si no coinciden
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Asegurar que psi tiene el mismo tamaño que x
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return x * psi

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
        
        # g, theta, phi
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        # W
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels) if bn_layer else nn.Identity()
        )
        
        nn.init.constant_(self.W[0].weight, 0)
        if self.W[0].bias is not None:
            nn.init.constant_(self.W[0].bias, 0)
        
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        z = W_y + x  # Residual connection
        
        return z

class EnhancedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.se = SqueezeExcitation(out_channels, reduction=16)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Aplicar SE antes de la suma residual
        out = self.se(out)
        out += identity
        out = self.relu(out)
        
        return out

class ImprovedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.05):
        super().__init__()
        
        self.gelu = nn.GELU()
        
        # Primera convolución con residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Segunda convolución
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # NonLocal block (ya tiene residual interna)
        self.non_local = NonLocalBlock(
            in_channels=out_channels,
            inter_channels=out_channels // 2,
            sub_sample=False,
            bn_layer=True
        )
        
        # Match channels para residual
        self.match_channels = (in_channels != out_channels)
        if self.match_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        # Primera conv con activación
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        # Segunda conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual antes de NonLocal
        if self.match_channels:
            identity = self.downsample(identity)
        out = out + identity
        out = self.gelu(out)
        
        # NonLocal (tiene su propia residual)
        out = self.non_local(out)
        
        return out


class ImprovedNestedUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, deep_supervision=False, 
                 use_attention=True, **kwargs):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.deep_supervision = deep_supervision
        self.use_attention = use_attention
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        # Attention Gates para conexiones skip
        if self.use_attention:
            # Nivel 1
            self.att1_0 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], 
                                        F_int=nb_filter[0]//2)
            
            # Nivel 2
            self.att2_0 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], 
                                        F_int=nb_filter[1]//2)
            self.att1_1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], 
                                        F_int=nb_filter[0]//2)
            
            # Nivel 3
            self.att3_0 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], 
                                        F_int=nb_filter[2]//2)
            self.att2_1 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], 
                                        F_int=nb_filter[1]//2)
            self.att1_2 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], 
                                        F_int=nb_filter[0]//2)
            
            # Nivel 4
            self.att4_0 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3], 
                                        F_int=nb_filter[3]//2)
            self.att3_1 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], 
                                        F_int=nb_filter[2]//2)
            self.att2_2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], 
                                        F_int=nb_filter[1]//2)
            self.att1_3 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], 
                                        F_int=nb_filter[0]//2)
        
        # Decoder con nested connections
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    
    def forward(self, input):
        # Encoder path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Nested decoder with attention
        # Column 1
        if self.use_attention:
            # Primero upsampling, luego attention
            x1_0_up = self.up(x1_0)
            x0_0_att = self.att1_0(x0_0, x1_0_up)
            x0_1 = self.conv0_1(torch.cat([x0_0_att, x1_0_up], 1))
            
            x2_0_up = self.up(x2_0)
            x1_0_att = self.att2_0(x1_0, x2_0_up)
            x1_1 = self.conv1_1(torch.cat([x1_0_att, x2_0_up], 1))
            
            x3_0_up = self.up(x3_0)
            x2_0_att = self.att3_0(x2_0, x3_0_up)
            x2_1 = self.conv2_1(torch.cat([x2_0_att, x3_0_up], 1))
            
            x4_0_up = self.up(x4_0)
            x3_0_att = self.att4_0(x3_0, x4_0_up)
            x3_1 = self.conv3_1(torch.cat([x3_0_att, x4_0_up], 1))
        else:
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # Column 2
        if self.use_attention:
            x1_1_up = self.up(x1_1)
            x0_1_att = self.att1_1(x0_1, x1_1_up)
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1_att, x1_1_up], 1))
            
            x2_1_up = self.up(x2_1)
            x1_1_att = self.att2_1(x1_1, x2_1_up)
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1_att, x2_1_up], 1))
            
            x3_1_up = self.up(x3_1)
            x2_1_att = self.att3_1(x2_1, x3_1_up)
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1_att, x3_1_up], 1))
        else:
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # Column 3
        if self.use_attention:
            x1_2_up = self.up(x1_2)
            x0_2_att = self.att1_2(x0_2, x1_2_up)
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2_att, x1_2_up], 1))
            
            x2_2_up = self.up(x2_2)
            x1_2_att = self.att2_2(x1_2, x2_2_up)
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2_att, x2_2_up], 1))
        else:
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Column 4
        if self.use_attention:
            x1_3_up = self.up(x1_3)
            x0_3_att = self.att1_3(x0_3, x1_3_up)
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3_att, x1_3_up], 1))
        else:
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
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
        
class UNet(nn.Module):
    def __init__(self, input_channels=1, deep_supervision=False, dropout_rate=0.3, **kwargs):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.deep_supervision = deep_supervision
        self.dropout_rate = dropout_rate
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

class NestedUNet(nn.Module):
    def __init__(self, input_channels=1, deep_supervision=False, dropout_rate=0.3, **kwargs):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.deep_supervision = deep_supervision
        self.dropout_rate = dropout_rate
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output