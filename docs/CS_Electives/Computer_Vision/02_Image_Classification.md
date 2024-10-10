# Image Classification

Image --> Class
## Why is it hard?

- Semantic gap between input and output
- Viewpoint variation
  - Translational
  - Rotational
- Illumination variation
- Deformation of object
- Occlusion: object partially hidden
- Background clutter
- Intraclass variation
- Textural variation

## Models

|                      |                          | Disadvantage                               | Robust to variance |
| -------------------- | ------------------------ | ------------------------------------------ | ------------------ |
| $k$ Nearest neighbor | L1 L2 distance of pixels | Inference speed proportional to train size | ❌                  |
| Linear               |                          |                                            | ❌                  |
| FNN                  |                          |                                            | ❌                  |
| CNNs                 |                          |                                            | ✅                  |

## Pre-Processing

- Resize images to the same size
- Does Greyscale work better???
  - Greyscale worsens linear classifier because it can no longer extract colors; linear classifier cannot extract textures well regardless anyways
- Normalize
	- Subtract mean image
	  or
	- Subtract per channel mean
