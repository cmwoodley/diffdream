## "Diffream" - Pharmacophore guided generative diffusion model

Carried out as a side project during my PostDoc at the University of Liverpool. Effectively an attempt at trying to replace the autoencoder used in [LigDream](https://doi.org/10.1021/acs.jcim.8b00706) with a 3D Unet.

Adapted a unet structure for medical image segmentation from [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://doi.org/10.1007/978-3-319-46723-8_49)

Trained using 

Implementation uses two methods of guidance for generation:
- Pharmacophore nudging - Inputs noise to generate intermediate representation --> Pharmacophore representation predicted from intermediate rep. --> Calculate BCE loss with true pharmacophore --> adjust input with autograd  --> SMILES predicted with captioning networks from Ligdream
- Embedded pharmacophore conditioning - inputs noise and embedded representation of pharmacophore --> denoised for t timesteps mixing in with original sample --> generated representation captioned by SMILES

## WIP

* Captioning networks are not good --> would like to replace with transformers
