'''
Modified version of inference from https://github.com/krulllab/GAP/tree/main/gap 
'''

import torch as torch
import numpy as np

'''
Samples an image using Conditional Generative Accumulation of Photons (GAP) based on an initial photon image.
If the initial photon image contains only zeros the model samples from scratch. 
If it contains photon numbers, the model performs diversity denoising.
The initial model in the cascade is chosen based on the PSNR of the input image.

        Parameters:
                input_image (torch tensor): the initial photon image, containing integers (batch, channel, y, x).  
                model: list of network used to predict the next phton location.
                max_photons (int): stop sampling when image contains more photons. 
                max_its (int): stop sampling after max_its iterations. 
                max_psnr (float): stop sampling when pseudo PSNR is larger max_psnr
                save_every_n (int): store and return images at every nth step. 
                augment (bool): use 8-fold data augmentation (default is False) 
                beta (float): photon number is increased exponentially by factor beta in each step.
        Returns:
                denoised (numpy array): denoised image at the end of that sampling process.
                photons (numpy array): photon image at the end of the sampling process.
                stack (list): list of numpy arrays containing intermediate results.
                i (int) number of executed iterations.
'''

def sample_image(input_image,
                 models,
                 max_photons = None,
                 max_its = 500000,
                 max_psnr = 30,
                 save_every_n = 5,
                 beta = 0.1,
                 channels = 1,
                ):

    start = input_image[:,-channels:, :, :].clone()
    cond_input = input_image[:,:-channels, :, :].clone()
    photons = start
    photnum = 1

    denoised = None
    stack = []

    psnrs = [-30, -20, -10, 0, 10, 20, 30]
    psnrs = psnrs[:len(models)]
    
    for n in range(len(psnrs)):
        max_psnr = psnrs[n]
        model = models[n]

        for i in range(max_its):
            psnr = np.log10( photons.mean().item() + 1e-50) * 10
            psnr = max(-40, psnr)
                
            if (max_photons is not None) and (photons.sum().item() > max_photons):
                break
                
            if psnr > max_psnr:
                break

            input = torch.cat((cond_input, photons),1)
            denoised = model(input).detach()
    
            denoised = denoised - denoised.max()
            denoised = torch.exp(denoised)   
            denoised = denoised / (denoised.sum(dim=(-1,-2,-3), keepdim = True))
            
            if (save_every_n is not None) and (i%save_every_n == 0):  

                imgsave = denoised[0,:,...].detach().cpu()
                imgsave = imgsave/imgsave.max()
                photsave = photons[0,:,...].detach().cpu()
                photsave = photsave / max(photsave.max(),1)      
                combi = torch.cat((photsave,imgsave),2)
                stack.append([combi.numpy(), psnr])

            # increase photon number    
            photnum = max(beta* photons.sum(),1)
            # draw new photons
            new_photons = torch.poisson(denoised*(photnum))
            
            # add new photons
            photons = photons + new_photons
        
    return denoised[...].detach().cpu().numpy(), photons[...].detach().cpu().numpy(), stack, i