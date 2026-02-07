# Segment Anything Models (SAM) API

<img src="https://github.com/ksugar/samapi/releases/download/assets/qupath-samapi.gif" width="768">

<img src="https://github.com/ksugar/samapi/releases/download/assets/qupath-sam-multipoint-live.gif" width="768">

A web API for [SAM](https://github.com/facebookresearch/segment-anything) implemented with [FastAPI](https://fastapi.tiangolo.com).

This is a part of the following paper. Please [cite](#citation) it when you use this project. You will also cite the following papers:
-  [the original SAM paper](https://arxiv.org/abs/2304.02643)
-  [the SAM2 paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
-  [the SAM3 paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
-  [the MobileSAM paper](https://arxiv.org/abs/2306.14289).

- Sugawara, K. [*Training deep learning models for cell image segmentation with sparse annotations.*](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1) bioRxiv 2023. doi:10.1101/2023.06.13.544786


## Install

Create a conda environment.

```bash
conda create -n samapi -y python=3.12
conda activate samapi
```

Install PyTorch and torchvision.
# 
```bash
# Windows with CUDA-compatible GPU only
python -m pip install "torch==2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu126

# Windows: also required for SAM3 and SAM2 support
python -m pip install triton-windows
```

Install `samapi` and its dependencies.

```bash
python -m pip install git+https://github.com/ksugar/samapi.git
```

If you are using WSL2, `LD_LIBRARY_PATH` will need to be updated as follows.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Update

If you want to update the samapi server, run the following command in the conda environment.

```bash
python -m pip install -U git+https://github.com/ksugar/samapi.git
```

## Pre-requisites for SAM3

> [!NOTE]
> SAM3 access on Hugging Face is gated by Meta. To use SAM3, you must [request access](https://huggingface.co/facebook/sam3) to the model. However, since v0.7.1, the samapi server can still run without SAM3 using other models (SAM, SAM2, MobileSAM).

The following steps are required only when you want to use SAM3.

### Login to Hugging Face (Optional: required for SAM3)

You need to install the [Hugging Face CLI](https://huggingface.co/docs/huggingface-cli/index) and login to your Hugging Face account to access SAM3 weights.

#### Install Hugging Face CLI on macOS and Linux:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

#### Install Hugging Face CLI on Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

#### Login to Hugging Face:
```bash
hf auth login
```

Enter your token generated at https://huggingface.co/settings/tokens when prompted as follows:

```powershell

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token can be pasted using 'Right-Click'.
Enter your token (input will not be visible):
```

If you are asked to add the token to your Git credentials, you can choose `y`.

## Usage

### Launch a server

Since `v0.4.0`, it is important to launch the server with `--workers 2` (or more) to enable cancellation of a download of a weights file.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1 # Required for running on Apple silicon
uvicorn samapi.main:app --workers 2
```

The command above will launch a server at http://localhost:8000.

```
INFO:     Started server process [21258]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> [!NOTE]
> If you want to access remotely, you may need to launch with `--host 0.0.0.0`.

```bash
uvicorn samapi.main:app --workers 2 --host 0.0.0.0
```

For more information, see [uvicorn documentation](https://www.uvicorn.org/#command-line-options).

### Troubleshooting

If you try to process a large image and receive the following error, you may need to increase the `PIL.Image.MAX_IMAGE_PIXELS` value (default: `89478485`), or completely disable it (i.e. set the variable to the empty valie).

```bash
PIL.Image.DecompressionBombError: Image size (xxxxxxxxx pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.
```

In Linux and MacOS, you can set the environment variable as follows.

```bash
export PIL_MAX_IMAGE_PIXELS="" # or specific value (integer)
```

In Windows, you can set the environment variable as follows.

```cmd
set PIL_MAX_IMAGE_PIXELS="" # or specific value (integer)
```

### Known issues
- SAM3 video predictor does not work with negative bbox prompts. See https://github.com/facebookresearch/sam3/issues/335.
- If you do not have access to SAM3 on HuggingFace, the server will still start and work with other models (SAM, SAM2, MobileSAM), but SAM3-specific endpoints will return a 503 error.

### Request body

#### Endpoint `/sam/` (post)

```python
class SAMBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    bbox: Tuple[int, int, int, int] = Field(example=(0, 0, 0, 0))
    b64img: str
```

| key    | value                                       |
| ------ | ------------------------------------------- |
| type   | One of `vit_h`, `vit_l`, `vit_b` or `vit_t` |
| bbox   | Coordinate of a bbox `(x1, y1, x2, y2)`     |
| b64img | Base64-encoded image data                   |

#### Endpoint `/sam/automask/` (post)

```python
class SAMAutoMaskBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    b64img: str
    points_per_side: Optional[int] = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0
    output_type: str = "Single Mask"
    include_image_edge: bool = False
```

| key                            | value                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| type                           | One of `vit_h`, `vit_l`, or `vit_b`.                                                                                                                                                                                                                                                                                                        |
| b64img                         | Base64-encoded image data.                                                                                                                                                                                                                                                                                                                  |
| points_per_side                | The number of points to be sampled along one side of the image. The total number of points is points_per_side**2.                                                                                                                                                                                                                           |
| points_per_batch               | Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.                                                                                                                                                                                                                            |
| pred_iou_thresh                | A filtering threshold in [0,1], using the model's predicted mask quality.                                                                                                                                                                                                                                                                   |
| stability_score_thresh         | A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.                                                                                                                                                                                                  |
| stability_score_offset         | The amount to shift the cutoff when calculated the stability score.                                                                                                                                                                                                                                                                         |
| box_nms_thresh                 | The box IoU cutoff used by non-maximal suppression to filter duplicate masks.                                                                                                                                                                                                                                                               |
| crop_n_layers                  | If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.                                                                                                                                                                                    |
| crop_nms_thresh                | The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.                                                                                                                                                                                                                                       |
| crop_overlap_ratio             | Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.                                                                                                                                                             |
| crop_n_points_downscale_factor | The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.                                                                                                                                                                                                                                       |
| min_mask_region_area           | If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.                                                                                                                                                                                       |
| output_type                    | If 'Single Mask' is selected, the model will return single masks per prompt. If 'Multi-mask' is selected, the model will return three masks per prompt. 'Multi-mask (all)' keeps all three masks. One of the three masks is kept if the option 'Multi-mask (largest)', 'Multi-mask (smallest)', or 'Multi-mask (best quality)' is selected. |
| include_image_edge             | If True, include a crop area at the edge of the original image.                                                                                                                                                                                                                                                                             |

- [point_grids](https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/automatic_mask_generator.py#L86-L88) is not supported.

### Response body

The response body contains a list of [GeoJSON Feature objects](https://geojson.org).

Supporting other formats is a future work.

#### Endpoint `/sam/version/` (get)

Returns the version of the SAM API.

##### Response body

The version of the SAM API.

```plaintext
0.7.0
```

#### Endpoint `/sam/weights/` (get)

Returns a list of the available weights.

##### Query parameters

| key             | value                                        |
| --------------- | -------------------------------------------- |
| type (Optional) | One of `vit_h`, `vit_l`, `vit_b` or `vit_t`. |

##### Response body

A list of the available weights.

| key  | value                                        |
| ---- | -------------------------------------------- |
| type | One of `vit_h`, `vit_l`, `vit_b` or `vit_t`. |
| name | The name of the registered SAM weights.      |
| URL  | The URL of the registered SAM weights.       |

#### Endpoint `/sam/weights/` (post)

Registers SAM weights.

##### Request body

```python
class SAMWeightsBody(BaseModel):
    type: ModelType
    name: str
    url: str
```

| key  | value                                        |
| ---- | -------------------------------------------- |
| type | One of `vit_h`, `vit_l`, `vit_b` or `vit_t`. |
| name | The name of the SAM weights to register.     |
| URL  | The URL to the SAM weights file to register. |


##### Response body

A message indicating whether the registration is successful.

```plaintext
name https://path/to/weights/file.pth is registered.
```

#### Endpoint `/sam/weights/cancel/` (get)

Cancel the download of the SAM weights.

##### Response body

A message indicating that the cancel signal is sent.

```plaintext
Cancel signal sent
```

#### Endpoint `/sam/progress/` (get)

Returns the progress.

##### Response body

The progress.

| key     | value                              |
| ------- | ---------------------------------- |
| message | A message indicating the progress. |
| percent | Integer value in [0, 100].         |

## Updates

### v0.7.0

- Supprt [SAM3](https://github.com/facebookresearch/sam3) image predictor and video predictor.
- Require Python `3.12` and update `torch`/`torchvision` to newer releases.

### v0.6.0

- Support [SAM2](https://ai.meta.com/sam2/) video predictor.
- Make PIL.Image.MAX_IMAGE_PIXELS adjustable by an environment variable. See details in [Usage > Troubleshooting](#troubleshooting).

### v0.5.0

- Support [SAM2](https://ai.meta.com/sam2/) models.

### v0.4.1

- Update dependencies. Related to: [ksugar/qupath-extension-sam#16](https://github.com/ksugar/qupath-extension-sam/issues/16) by [@halqadasi](https://github.com/halqadasi) and [ksugar/samapi#18](https://github.com/ksugar/samapi/issues/18) by [@ArezooGhodsifard](https://github.com/ArezooGhodsifard).
  - [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit) from `11.7` to `11.8`.
  - [gdown](https://github.com/wkentaro/gdown) from `^4.7.1` to `^5.1.0`.
  - [torch](https://github.com/pytorch/pytorch) from `^1.13.1` to `^2.2.2`.
  - [torchvision](https://github.com/pytorch/vision) from `^0.14.1` to `^0.17.2`.

### v0.4.0

- Support for registering SAM weights from URL. [ksugar/qupath-extension-sam#8](https://github.com/ksugar/qupath-extension-sam/issues/8) [ksugar/samapi#11](https://github.com/ksugar/samapi/pull/11) by [@constantinpape](https://github.com/constantinpape)

### v0.3.0

- Support points and multi-mask output by [@petebankhead](https://github.com/petebankhead)
  
  <img src="https://github.com/ksugar/samapi/releases/download/assets/qupath-sam-multipoint.gif" width="768">

  <img src="https://github.com/ksugar/samapi/releases/download/assets/qupath-sam-rectangle-prompt.gif" width="768">

- Support SamAutomaticMaskGenerator
  
  <img src="https://github.com/ksugar/samapi/releases/download/assets/qupath-sam-automask.gif" width="768">

- Support [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  
- Add opencv-python to dependencies


### v0.2.0

- Support for MPS backend (MacOS) by [@petebankhead](https://github.com/petebankhead)

## Citation

Please cite my paper on [bioRxiv](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1).

```.bib
@article {Sugawara2023.06.13.544786,
	author = {Ko Sugawara},
	title = {Training deep learning models for cell image segmentation with sparse annotations},
	elocation-id = {2023.06.13.544786},
	year = {2023},
	doi = {10.1101/2023.06.13.544786},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning is becoming more prominent in cell image analysis. However, collecting the annotated data required to train efficient deep-learning models remains a major obstacle. I demonstrate that functional performance can be achieved even with sparsely annotated data. Furthermore, I show that the selection of sparse cell annotations significantly impacts performance. I modified Cellpose and StarDist to enable training with sparsely annotated data and evaluated them in conjunction with ELEPHANT, a cell tracking algorithm that internally uses U-Net based cell segmentation. These results illustrate that sparse annotation is a generally effective strategy in deep learning-based cell image segmentation. Finally, I demonstrate that with the help of the Segment Anything Model (SAM), it is feasible to build an effective deep learning model of cell image segmentation from scratch just in a few minutes.Competing Interest StatementKS is employed part-time by LPIXEL Inc.},
	URL = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786},
	eprint = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgements

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [FastAPI](https://fastapi.tiangolo.com)