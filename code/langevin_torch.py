import torch
import torchvision
import os
from math import sqrt
import numpy as np

@torch.no_grad()
def langevin(score_model, x, sigmas, eps, T, save=False, epochs=None, clamp=False, time_str=None, verbose=False):
    # it's better not to clamp
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bs = x.shape[0]
    all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * torch.ones(bs, dtype=torch.long)
        indices = indices.cuda()
        for t in range(T):
            noise = torch.randn_like(x).cuda()
            assert indices.shape == torch.Size([bs,])
            grad = score_model(x, indices)
            x = x + alpha / 2 * grad + sqrt(alpha) * noise
            if clamp:
                x = torch.clamp(x, 0, 1)
            if verbose:
                grad_norm = torch.norm(grad.view(bs, -1), dim=1).mean()
                image_norm = torch.norm(x.view(bs, -1), dim=1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
        if save:
            if i % (len(sigmas) // 10) == (len(sigmas) - 1) % (len(sigmas) // 10):
                all_samples.append(x.clone().cpu())

    if save:
        assert x.shape[0] == 10
        # assert len(sigmas) == 10
        assert epochs is not None
        if time_str is not None:
            save_dir = f'./NCSN/denoising_process/{time_str}/'
        else:
            save_dir = './NCSN/denoising_process/'
        filename = '{:>03d}.png'.format(epochs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # concatenate all samples
        all_samples = all_samples[:10]
        all_samples = torch.cat(all_samples, dim=0)
        # print(all_samples.shape)
        assert all_samples.shape == torch.Size([100, 1, 28, 28])
        # save the image
        grid = torchvision.utils.make_grid(all_samples, nrow=10, padding=2, pad_value=1)
        torchvision.utils.save_image(grid, os.path.join(save_dir, filename))

    return x

@torch.no_grad()
def langevin_masked(score_model, x, sigmas, eps, T, mask, save=False, epochs=None, clamp=False, verbose=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bs = x.shape[0]
    all_samples = []

    reverse_mask = 1 - mask
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * torch.ones(bs, dtype=torch.long)
        indices = indices.cuda()
        for t in range(T):
            noise = torch.randn_like(x).cuda()
            assert indices.shape == torch.Size([bs,])
            grad = score_model(x, indices)
            x = x + (alpha / 2 * grad + sqrt(alpha) * noise) * reverse_mask
            if clamp:
                x = torch.clamp(x, 0, 1)
            if verbose:
                grad_norm = torch.norm(grad.view(bs, -1), dim=1).mean()
                image_norm = torch.norm(x.view(bs, -1), dim=1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
        if save:
            if i % (len(sigmas) // 10) == (len(sigmas) - 1) % (len(sigmas) // 10):
                all_samples.append(x.clone().cpu())

    if save:
        assert x.shape[0] == 10
        # assert len(sigmas) == 10
        assert epochs is not None
        save_dir = './NCSN/denoising_process/'
        filename = '{:>03d}.png'.format(epochs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # concatenate all samples
        all_samples = all_samples[:10]
        all_samples = torch.cat(all_samples, dim=0)
        assert all_samples.shape == torch.Size([100, 1, 28, 28])
        # save the image
        grid = torchvision.utils.make_grid(all_samples, nrow=10, padding=2, pad_value=1)
        torchvision.utils.save_image(grid, os.path.join(save_dir, filename))

    return x