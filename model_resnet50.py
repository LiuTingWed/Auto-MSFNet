import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from utils import drop_path
import genotypes



class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

        out_channel = 128
        self.conv5 = nn.Conv2d(2048, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channel)
        self.conv4 = nn.Conv2d(1024, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(512, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)


    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)
        out5_ = F.relu(self.bn5(self.conv5(out5)), inplace=True)


        return out5_, out2, out3, out4

class Featurefusioncell43(nn.Module):
    def __init__(self, standardShape, channel, op):
        super(Featurefusioncell43, self).__init__()
        self.standardShape = standardShape
        self._ops = op
        self.conv11 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(channel)

    def forward(self, fea, lowfeature):
        f2 = fea[0]
        levelfusion = fea[2]
        f3 = fea[1]
        f4 = fea[3]

        assert levelfusion.size()[3] == self.standardShape
        if lowfeature.size()[2:] != self.standardShape:
            lowfeature = F.interpolate(lowfeature, self.standardShape, mode='bilinear')
        if f2.size()[3] != self.standardShape and f2.size() != torch.Size([]):
            f2 = F.interpolate(f2, self.standardShape, mode='bilinear')
        if f3.size()[3] != self.standardShape and f3.size() != torch.Size([]):
            f3 = F.interpolate(f3, self.standardShape, mode='bilinear')
        if f4.size()[3] != self.standardShape:
            f4 = F.interpolate(f4, self.standardShape, mode='bilinear')

        z1 = f2
        z2 = f3
        z3 = levelfusion
        z4 = f4
        pre_note = [lowfeature]
        states = [z1, z2, z3, z4]
        offset = 0
        for i in range(4):
            if i == 0:
                s0 = states[i]
                s1 = self._ops[offset + i](pre_note[i])
                add = s0 + s1
                pre_note.append(add)
            else:
                p1 = states[i]
                s0 = self._ops[offset + i](pre_note[i])
                s1 = self._ops[offset + i + 1](states[i])
                add = s0 +  s1 + p1
                pre_note.append(add)
                offset += 1

        out = 0
        for i in range(1, 5):
            out += pre_note[i]
        out = F.relu(self.bn11(self.conv11(out)), inplace=True)

        return out


class Featurefusioncell32(nn.Module):
    def __init__(self, standardShape, channel, op):
        super(Featurefusioncell32, self).__init__()
        self.standardShape = standardShape
        self._ops = op
        self.conv11 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(channel)

    def forward(self, fea, lowfeature):
        f2 = fea[0]
        levelfusion = fea[3]
        f3 = fea[1]
        f4 = fea[2]

        assert levelfusion.size()[3] == self.standardShape
        if lowfeature.size()[2:] != self.standardShape:
            lowfeature = F.interpolate(lowfeature, self.standardShape, mode='bilinear')
        if f2.size()[3] != self.standardShape and f2.size() != torch.Size([]):
            f2 = F.interpolate(f2, self.standardShape, mode='bilinear')
        if f3.size()[3] != self.standardShape and f3.size() != torch.Size([]):
            f3 = F.interpolate(f3, self.standardShape, mode='bilinear')
        if f4.size()[3] != self.standardShape:
            f4 = F.interpolate(f4, self.standardShape, mode='bilinear')

        z1 = f2
        z2 = f3
        z3 = f4
        z4 = levelfusion

        pre_note = [lowfeature]
        states = [z1, z2, z3, z4]
        offset = 0
        for i in range(4):
            if i == 0:
                s0 = states[i]
                s1 = self._ops[offset + i](pre_note[i])
                add = s0 + s1
                pre_note.append(add)
            else:
                p1 = states[i]
                s0 = self._ops[offset + i](pre_note[i])
                s1 = self._ops[offset + i + 1](states[i])
                add = s0 +  s1 + p1
                pre_note.append(add)
                offset += 1

        out = 0
        for i in range(1, 5):
            out += pre_note[i]
        out = F.relu(self.bn11(self.conv11(out)), inplace=True)
        return out


class Featurefusioncell54(nn.Module):
    def __init__(self, standardShape, channel, op):
        super(Featurefusioncell54, self).__init__()
        self.standardShape = standardShape
        self._ops = op
        self.conv11 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(channel)

    def forward(self, fea):
        lowfeature = fea[0]
        levelfusion = fea[1]
        f2 = fea[2]
        f3 = fea[3]

        # if levelfusion is not None:
        assert levelfusion.size()[3] == self.standardShape
        if lowfeature.size()[2:] != self.standardShape:
            lowfeature = F.interpolate(lowfeature, self.standardShape, mode='bilinear')
        if f2.size()[3] != self.standardShape and f2.size() != torch.Size([]):
            f2 = F.interpolate(f2, self.standardShape, mode='bilinear')
        if f3.size()[3] != self.standardShape and f3.size() != torch.Size([]):
            f3 = F.interpolate(f3, self.standardShape, mode='bilinear')

        z1 = lowfeature
        z2 = levelfusion
        z3 = f2
        z4 = f3

        pre_note = [z1]
        states = [z2, z3, z4]
        offset = 0
        for i in range(4 - 1):
            if i == 0:
                s0 = states[i]
                s1 = self._ops[offset + i](pre_note[i])
                add = s0 + s1
                pre_note.append(add)
            else:
                p1 = states[i]
                s0 = self._ops[offset + i](pre_note[i])
                s1 = self._ops[offset + i + 1](states[i])
                add = s0 + s1 + p1
                pre_note.append(add)
                offset += 1

        out = 0
        for i in range(1, 4):
            out += pre_note[i]
        out = F.relu(self.bn11(self.conv11(out)), inplace=True)

        return out


class FeatureFusion(nn.Module):

    def __init__(self, genotype_fusion, node=3):
        super(FeatureFusion, self).__init__()

        self._ops = nn.ModuleList()
        self.fnum = 4
        self.node = node
        C = 128

        genotype_ouside = genotype_fusion.normal
        genotype_inside = genotype_fusion.inside
        new_genotype_ouside = sorted(genotype_ouside, key=lambda x: (x[2], x[1]))
        op_name, op_num, _ = zip(*new_genotype_ouside)

        self.op_num = op_num
        offset = 0
        for i in range(self.node):
            for j in range(self.fnum):
                op = OPS[op_name[j + offset]](C, C, 1,False, True)
                self._ops += [op]
            offset += 4

        op_name_inside, op_num_inside = zip(*genotype_inside)

        k = [5, 7, 7]
        noteOper = []
        offset = 0
        for i in range(self.node):
            self._nodes = nn.ModuleList()
            for j in range(k[i]):
                op = OPS[op_name_inside[j + offset]](C, C, 1,False, True)
                self._nodes += [op]
            noteOper.append(self._nodes)
            offset += k[i]

        self.featurefusioncell54 = Featurefusioncell54(16, C, noteOper[0])
        self.featurefusioncell43 = Featurefusioncell43(32, C, noteOper[1])
        self.featurefusioncell32 = Featurefusioncell32(64, C, noteOper[2])

    def forward(self, out5, out2, out3, out4):

        states = [out5, out4, out3, out2]

        # 每一条边的特征权重，遍历完一个节点要clear，每一轮4个，一共12条边
        fea = []
        # 每一个fusion节点输出的tensor字典
        feaoutput = []
        offset = 0
        s = 0

        for i in range(self.node):
            for j, v in enumerate(self.op_num):
                if j == 4:
                    break
                inputFea = states[v]
                x2 = self._ops[offset + j](inputFea)
                fea.append(x2)

            if i == 0:
                new_fea = self.featurefusioncell54(fea)
                feaoutput.append(new_fea)
                fea.clear()

            elif i == 1:
                new_fea = self.featurefusioncell43(fea, feaoutput[0])
                feaoutput.append(new_fea)
                fea.clear()

            elif i == 2:
                new_fea = self.featurefusioncell32(fea, feaoutput[1])
                feaoutput.append(new_fea)
                fea.clear()

            offset += 4

        return feaoutput[2],feaoutput[1],feaoutput[0]

    def _loss(self, input, target):
        logits = self(input)
        logits = logits.squeeze(1)

        return self._criterion(logits, target)

class vgg16(nn.Module):
    def __init__(self,):
        super(vgg16, self).__init__()

        # original image's size = 256*256*3

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)                    # 1/32    4 layers

        out_channel = 128
        self.conv5 = nn.Conv2d(512, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channel)
        self.conv4 = nn.Conv2d(512, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(128, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        h = x

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        # pool1 = h

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        # pool2 = h

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h_nopool3 = h
        h = self.pool3(h)
        # pool3 = h

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h_nopool4 = h
        h = self.pool4(h)

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))

        out2 = F.relu(self.bn2(self.conv2(h_nopool2)), inplace=True)
        out3 = F.relu(self.bn3(self.conv3(h_nopool3)), inplace=True)
        out4 = F.relu(self.bn4(self.conv4(h_nopool4)), inplace=True)
        out5_ = F.relu(self.bn5(self.conv5(h)), inplace=True)

        return out5_, out2, out3, out4

class Network_Resnet50(nn.Module):

    def __init__(self,genotype_fusion):
        super(Network_Resnet50, self).__init__()
        self.resnet = ResNet()
        self.feafusion = FeatureFusion(genotype_fusion)
        self.conv44 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.conv55 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.conv66 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        h_, h_nopool2, h_nopool3, h_nopool4 = self.resnet(input)
        h_nopool2,h_nopool3,h_nopool4 = self.feafusion(h_, h_nopool2, h_nopool3, h_nopool4)
        h_nopool2 = F.interpolate(self.conv44(h_nopool2), size=[256, 256], mode='bilinear')
        h_nopool3 = F.interpolate(self.conv55(h_nopool3), size=[256, 256], mode='bilinear')
        h_nopool4 = F.interpolate(self.conv66(h_nopool4), size=[256, 256], mode='bilinear')
        return h_nopool2,h_nopool3,h_nopool4
