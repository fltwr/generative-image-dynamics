import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat

def get_resnet34_feature_extractor(layers):
    from torchvision.models import resnet34, ResNet34_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    return_nodes = [f"features.{l}" for l in layers]
    return create_feature_extractor(m, return_nodes)

class VGGLoss(nn.Module):
    """The feature reconstruction loss in Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://arxiv.org/abs/1603.08155)."""
    
    def __init__(self, layers=[3, 8, 15, 22]):
        super().__init__()
        self.feature_extractor = get_resnet34_feature_extractor(layers).eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x, y):
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        loss = 0
        for k in x.keys():
            loss += torch.nn.functional.l1_loss(x[k], y[k], reduction="mean")
        return loss / len(x.keys())

class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(self):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip):
                super().__init__()

                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )

                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )

                # end

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                    # end
                # end
            # end

            def forward(self, tenInput):
                if self.boolSkip == False:
                    return self.netMain(tenInput)
                # end

                if self.netShortcut is None:
                    return self.netMain(tenInput) + tenInput

                elif self.netShortcut is not None:
                    return self.netMain(tenInput) + self.netShortcut(tenInput)

                # end
            # end
        # end

        class Downsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Encode(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=32, init=0.25),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=32, init=0.25)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=64, init=0.25),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=64, init=0.25)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=96, init=0.25),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=96, init=0.25)
                )
            # end

            def forward(self, tenInput):
                tenOutput = []

                tenOutput.append(self.netOne(tenInput))
                tenOutput.append(self.netTwo(tenOutput[-1]))
                tenOutput.append(self.netThr(tenOutput[-1]))

                return [torch.cat([tenInput, tenOutput[0]], 1)] + tenOutput[1:]
            # end
        # end

        class Softmetric(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netInput = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                self.netFlow = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)

                for intRow, intFeatures in [(0, 16), (1, 32), (2, 64), (3, 96)]:
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
                # end

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([16, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([32, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([64, 96, 96]))
                # end

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([96, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([64, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([32, 16, 16]))
                # end

                self.netOutput = Basic('conv-relu-conv', [16, 16, 1], True)
            # end

            def forward(self, tenEncone, tenFlow):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([
                    self.netInput(tenEncone[0][:, 0:3, :, :]),
                    self.netFlow(tenFlow),
                ], 1)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                        if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp
                    # end
                # end

                return self.netOutput(tenColumn[0])
            # end
        # end

        class Warp(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = Basic('conv-relu-conv', [3 + 32 + 1, 32, 32], True)
                self.netTwo = Basic('conv-relu-conv', [0 + 64 + 1, 64, 64], True)
                self.netThr = Basic('conv-relu-conv', [0 + 96 + 1, 96, 96], True)
            # end

            def forward(self, tenEncone, tenMetricone, tenForward):
                tenOutput = []

                for intLevel in range(3):
                    if intLevel != 0:
                        tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                        
                        tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    # end
                    
                    tenOutput.append([self.netOne, self.netTwo, self.netThr][intLevel](
                        softsplat(tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1), tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    ))
                # end

                return tenOutput
            # end
        # end

        self.netEncode = Encode()

        self.netSoftmetric = Softmetric()

        self.netWarp = Warp()

        for intRow, intFeatures in [(0, 32), (1, 64), (2, 96)]:
            self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x3' + ' - ' + str(intRow) + 'x4', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x4' + ' - ' + str(intRow) + 'x5', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
        # end

        for intCol in [0, 1, 2]:
            self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([32, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([64, 96, 96]))
        # end

        for intCol in [3, 4, 5]:
            self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([96, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([64, 32, 32]))
        # end

        self.netOutput = Basic('conv-relu-conv', [32, 32, 3], True)
    # end

    def forward(self, tenOne, tenForward):
        tenEncone = self.netEncode(tenOne)
        
        tenMetricone = torch.norm(tenForward, p=2, dim=1, keepdim=True)  

        tenWarp = self.netWarp(tenEncone, tenMetricone, tenForward)

        tenColumn = [None, None, None]

        tenColumn[0] = tenWarp[0]
        tenColumn[1] = tenWarp[1] + self._modules['0x0 - 1x0'](tenColumn[0])
        tenColumn[2] = tenWarp[2] + self._modules['1x0 - 2x0'](tenColumn[1])

        intColumn = 1
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] = tenColumn[intRow] + self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
            # end
        # end

        intColumn = 2
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] = tenColumn[intRow] + self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
            # end
        # end

        intColumn = 3
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        intColumn = 4
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        intColumn = 5
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        return self.netOutput(tenColumn[0])
    # end
# end

@torch.no_grad()
def predict_sequence(src_frame: torch.Tensor, flow: torch.Tensor, model: Synthesis, transforms, return_tensor: bool = True):
    """
    Sequentially predict frames using the model by updating src_frame every step.
    src_frame: initial frame tensor (1, C, H, W)
    flow: (N, 2, H, W) optical flow between frames
    """
    out_frames = [src_frame]
    
    current_frame = src_frame
    for i in range(flow.shape[0]):
        current_flow = flow[i:i+1]  # batch size 1
        
        pred_frame = model(current_frame, current_flow)
        
        # Optionally denormalize or deprocess here, or after loop
        current_frame = pred_frame
        
        out_frames.append(current_frame)
    
    out_frames = torch.cat(out_frames, dim=0)
    
    if return_tensor:
        return transforms.denormalize_frame(out_frames)
    else:
        return transforms.deprocess_frame(out_frames)

@torch.no_grad()
def softsplat_tensor(src_frame: torch.Tensor, flow: torch.Tensor, transforms, weight_type: Union[None, str] = None, return_tensor: bool = True):
    # src_frame and flow should be normalized tensors
    if weight_type == "flow_mag":
        weight = torch.sqrt(torch.square(flow[:, 0, :, :] + flow[:, 1, :, :])).unsqueeze(1)
        mode = "soft"
    else:
        weight = None
        mode = "avg"
    out_frames = softsplat(tenIn=src_frame.repeat(flow.shape[0], 1, 1, 1), tenFlow=flow, tenMetric=weight, strMode=mode)
    if return_tensor:
        return transforms.denormalize_frame(out_frames)
    else:
        return transforms.deprocess_frame(out_frames)
