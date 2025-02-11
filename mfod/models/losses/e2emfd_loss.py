import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from mfod.registry import MODELS

def smoothing(kernel_size, sigma, channels, device):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          torch.tensor(-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance),dtype=torch.float)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    gaussian_filter.to(device)
    return gaussian_filter

def laplacian_pyramid(img, kernels, max_levels=3):
    pyr = []
    for level in range(max_levels):
        filtered = kernels[level](img)
        diff = img - filtered
        pyr.append(diff)
    return pyr

## fusion loss
class LapLoss2(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
        super(LapLoss2, self).__init__()
        self.max_levels = max_levels
        # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.gauss_kernel = smoothing(5, 2, channels, device)
        self.gauss_kernels =[]
        # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.gauss_kernels.append(smoothing(3, 2, channels, device))
        self.gauss_kernels.append(smoothing(5, 2, channels, device))
        self.gauss_kernels.append(smoothing(7, 2, channels, device))

    def forward(self, input, ir, vis):
        pyr_input = laplacian_pyramid(img=input, kernels=self.gauss_kernels, max_levels=self.max_levels)
        pyr_ir = laplacian_pyramid(img=ir, kernels=self.gauss_kernels, max_levels=self.max_levels)
        pyr_vis = laplacian_pyramid(img=vis, kernels=self.gauss_kernels, max_levels=self.max_levels)
        loss = 10. * sum(torch.nn.functional.l1_loss(a, torch.maximum(b,c)) for a, b,c in zip(pyr_input[:-1], pyr_ir[:-1],pyr_vis[:-1] ))
        loss = loss + torch.nn.functional.l1_loss(pyr_input[-1], torch.maximum(pyr_ir[-1],pyr_vis[-1]))
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.subplot(131)
        # plt.axis('off')
        # plt.imshow((pyr_input[0]*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.subplot(132)
        # plt.axis('off')
        # plt.imshow((pyr_ir[0]*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.subplot(133)
        # plt.axis('off')
        # plt.imshow((pyr_vis[0]*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # # plt.savefig("mask.png")
        # plt.savefig("grad.png")
        return loss#,pyr_input,pyr_ir,pyr_vis
    
    
@MODELS.register_module()
class DetcropPixelLoss(nn.Module):
    """Loss function for the pixel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(DetcropPixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()
        self.lap=LapLoss2()

    def forward(self, im_fusion, im_v, im_tir, mask, num_greater=None, total=None):
        """Forward function.
        Args:
            # im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            # im_rgb (Tensor): RGB image with shape (N, C, H, W).
            im_vi (Tensor): RGB image with shape (N, C, H, W).
            im_v (Tensor): HSV(V) image with shape (N, C, H, W).
        """

        image_ir=im_tir[:,:1,:,:]
        # pixel_max = torch.min(im_v, image_ir)
        pixel_max = torch.max(im_v, image_ir) + (im_v + image_ir)/2.0
        # pixel_max = image_ir

        mask_fusion = torch.where(mask>0, im_fusion, mask)
        mask_pixel = torch.where(mask>0, pixel_max.detach(), mask)

        # pixel_mean = (im_v + image_ir)/2.0
        # pixel_mean = torch.min(im_v, image_ir)
        # pixel_mean = torch.max(im_v, image_ir)
        pixel_mean = image_ir

        bg_mask = 1 - mask                                            
        bg_fusion = torch.where(bg_mask>0, im_fusion, bg_mask)
        bg_pixel = torch.where(bg_mask>0, pixel_mean.detach(), bg_mask) 

        mask_loss = self.L1_loss(mask_fusion, mask_pixel)
        bg_loss = self.L1_loss(bg_fusion, bg_pixel)
        pixel_loss = self.loss_weight * (mask_loss + bg_loss)

        loss_grad = self.lap(im_fusion, image_ir, im_v)

        # if num_greater > int(total * 0.8):  #ir权重大
        #     SSIM_loss = (1-ssim(im_fusion, image_ir))/2
        # elif (total - num_greater) > int(total * 0.8):
        #     SSIM_loss = (1-ssim(im_fusion, im_v))/2
        # else:
        #     SSIM_loss = (1-ssim(im_fusion, im_v))/2 + (1-ssim(im_fusion, image_ir))/2
        SSIM_loss = (1-ssim(im_fusion, im_v))/2 + (1-ssim(im_fusion, image_ir))/2
        return SSIM_loss, loss_grad, pixel_loss
    
    
# Matlab style 1D gaussian filter.    
def gaussian(window_size, sigma):    
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# Matlab style n_D gaussian filter.    
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    
    #  I added this for sm    
#    ssim_map = torch.exp(1 + ssim_map)
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
