import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['UNet', 'NestedUNet']


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
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma * out + x
        return out
    
class ImprovedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        
        self.self_attention = SelfAttention(out_channels)
        
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.final_dropout = nn.Dropout2d(p=dropout_rate * 0.5)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        out = self.dropout2(out)
        
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        out = out + identity
        
        out = self.self_attention(out)
        
        out = self.final_dropout(out)
        
        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, dropout_rate=0.2, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.dropout_rate = dropout_rate

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], dropout_rate=0)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], dropout_rate=dropout_rate * 0.5)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], dropout_rate=dropout_rate * 0.75)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], dropout_rate=dropout_rate)
        
        self.bottleneck = ImprovedBottleneck(nb_filter[3], nb_filter[4], dropout_rate=dropout_rate)

        self.att_gate_0_1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0]+nb_filter[1], F_int=nb_filter[0])
        self.att_gate_1_1 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1]+nb_filter[2], F_int=nb_filter[1])
        self.att_gate_2_1 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2]+nb_filter[3], F_int=nb_filter[2])
        self.att_gate_3_1 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3]+nb_filter[4], F_int=nb_filter[3])
        
        self.att_gate_0_2 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0]*2+nb_filter[1], F_int=nb_filter[0])
        self.att_gate_1_2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1]*2+nb_filter[2], F_int=nb_filter[1])
        self.att_gate_2_2 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2]*2+nb_filter[3], F_int=nb_filter[2])
        
        self.att_gate_0_3 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0]*3+nb_filter[1], F_int=nb_filter[0])
        self.att_gate_1_3 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1]*3+nb_filter[2], F_int=nb_filter[1])
        
        self.att_gate_0_4 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0]*4+nb_filter[1], F_int=nb_filter[0])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=dropout_rate * 0.5)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=dropout_rate * 0.5)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate=dropout_rate * 0.75)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], dropout_rate=dropout_rate)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=dropout_rate * 0.3)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=dropout_rate * 0.5)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], dropout_rate=dropout_rate * 0.75)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=dropout_rate * 0.2)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], dropout_rate=dropout_rate * 0.3)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], dropout_rate=dropout_rate * 0.1)

        if self.deep_supervision:
            self.final_dropout = nn.Dropout2d(p=dropout_rate * 0.5)
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final_dropout = nn.Dropout2d(p=dropout_rate * 0.5)
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        concat_0_1 = torch.cat([x0_0, self.up(x1_0)], 1)
        x2_0 = self.conv2_0(self.pool(x1_0))
        attended_0_1 = self.att_gate_0_1(concat_0_1, self.up(x1_0))
        x0_1 = self.conv0_1(attended_0_1)

        concat_1_1 = torch.cat([x1_0, self.up(x2_0)], 1)
        x3_0 = self.conv3_0(self.pool(x2_0))
        attended_1_1 = self.att_gate_1_1(concat_1_1, self.up(x2_0))
        x1_1 = self.conv1_1(attended_1_1)
        
        concat_0_2 = torch.cat([x0_0, x0_1, self.up(x1_1)], 1)
        attended_0_2 = self.att_gate_0_2(concat_0_2, self.up(x1_1))
        x0_2 = self.conv0_2(attended_0_2)

        concat_2_1 = torch.cat([x2_0, self.up(x3_0)], 1)
        
        x4_0 = self.bottleneck(self.pool(x3_0))
        
        attended_2_1 = self.att_gate_2_1(concat_2_1, self.up(x3_0))
        x2_1 = self.conv2_1(attended_2_1)
        
        concat_1_2 = torch.cat([x1_0, x1_1, self.up(x2_1)], 1)
        attended_1_2 = self.att_gate_1_2(concat_1_2, self.up(x2_1))
        x1_2 = self.conv1_2(attended_1_2)
        
        concat_0_3 = torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)
        attended_0_3 = self.att_gate_0_3(concat_0_3, self.up(x1_2))
        x0_3 = self.conv0_3(attended_0_3)

        concat_3_1 = torch.cat([x3_0, self.up(x4_0)], 1)
        attended_3_1 = self.att_gate_3_1(concat_3_1, self.up(x4_0))
        x3_1 = self.conv3_1(attended_3_1)
        
        concat_2_2 = torch.cat([x2_0, x2_1, self.up(x3_1)], 1)
        attended_2_2 = self.att_gate_2_2(concat_2_2, self.up(x3_1))
        x2_2 = self.conv2_2(attended_2_2)
        
        concat_1_3 = torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)
        attended_1_3 = self.att_gate_1_3(concat_1_3, self.up(x2_2))
        x1_3 = self.conv1_3(attended_1_3)
        
        concat_0_4 = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)
        attended_0_4 = self.att_gate_0_4(concat_0_4, self.up(x1_3))
        x0_4 = self.conv0_4(attended_0_4)

        if self.deep_supervision:
            output1 = self.final1(self.final_dropout(x0_1))
            output2 = self.final2(self.final_dropout(x0_2))
            output3 = self.final3(self.final_dropout(x0_3))
            output4 = self.final4(self.final_dropout(x0_4))
            return [output1, output2, output3, output4]
        else:
            output = self.final(self.final_dropout(x0_4))
            return output

# class UNet(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()

#         nb_filter = [32, 64, 128, 256, 512]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

#         self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
#         self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
#         self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

#         self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x0_0 = self.conv0_0(input)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x4_0 = self.conv4_0(self.pool(x3_0))

#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

#         output = self.final(x0_4)
#         return output


# class NestedUNet(nn.Module):
#     def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
#         super().__init__()

#         nb_filter = [32, 64, 128, 256, 512]

#         self.deep_supervision = deep_supervision

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

#         self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
#         self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

#         self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

#         self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

#         self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

#         if self.deep_supervision:
#             self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#         else:
#             self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


#     def forward(self, input):
#         x0_0 = self.conv0_0(input)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

#         x4_0 = self.conv4_0(self.pool(x3_0))
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

#         if self.deep_supervision:
#             output1 = self.final1(x0_1)
#             output2 = self.final2(x0_2)
#             output3 = self.final3(x0_3)
#             output4 = self.final4(x0_4)
#             return [output1, output2, output3, output4]

#         else:
#             output = self.final(x0_4)
#             return output