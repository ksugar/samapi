"""
This is a main.py file for the FastAPI app.
"""

from contextlib import redirect_stderr
from enum import Enum
from io import TextIOWrapper
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Tuple
import warnings

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from geojson import Feature
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from mobile_sam import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_vit_t,
)

from sam2.build_sam import build_sam2, _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

from samapi import __version__
from samapi.hub_extension import load_state_dict_from_url
from samapi.utils import decode_image, mask_to_geometry

logger = logging.getLogger("uvicorn")

SAMAPI_ROOT_DIR = os.getenv("SAMAPI_ROOT_DIR", str(Path.home() / ".samapi"))


SAMAPI_STDERR = SAMAPI_ROOT_DIR + "/samapi.stderr"
SAMAPI_CANCEL_FILE = SAMAPI_ROOT_DIR + "/samapi.cancel"


class ProgressIO(TextIOWrapper):
    """
    For progress bar, we need to redirect stderr to a file.
    """

    def __init__(self):
        super().__init__(sys.__stderr__.buffer, encoding=sys.__stderr__.encoding)

    def write(self, s: str):
        with open(SAMAPI_STDERR, "w", encoding="utf-8") as f:
            f.write(s)
        super().write(s)


progress_io = ProgressIO()


class EndpointFilter(logging.Filter):
    """
    To filter out the log messages from specified FastAPI endpoints.
    Reference: https://github.com/encode/starlette/issues/864#issuecomment-1254987630
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._path = path

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(EndpointFilter(path="/sam/progress/"))
uvicorn_logger.addFilter(EndpointFilter(path="/sam/weights/cancel/"))

app = FastAPI()


class ModelType(str, Enum):
    """
    Model types.
    """

    vit_h = "vit_h"
    vit_l = "vit_l"
    vit_b = "vit_b"
    vit_t = "vit_t"
    sam2_l = "sam2_l"
    sam2_bp = "sam2_bp"
    sam2_s = "sam2_s"
    sam2_t = "sam2_t"


DEFAULT_CHECKPOINT_URLS = {
    ModelType.vit_h: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ModelType.vit_l: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ModelType.vit_b: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ModelType.vit_t: "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    ModelType.sam2_l: "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    ModelType.sam2_bp: "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    ModelType.sam2_s: "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    ModelType.sam2_t: "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
}


def build_sam2_l(checkpoint=None):
    return build_sam2(
        "sam2_hiera_l.yaml",
        ckpt_path=checkpoint,
        device=_get_device(),
    )


def build_sam2_bp(checkpoint=None):
    return build_sam2(
        "sam2_hiera_b+.yaml",
        ckpt_path=checkpoint,
        device=_get_device(),
    )


def build_sam2_s(checkpoint=None):
    return build_sam2(
        "sam2_hiera_s.yaml",
        ckpt_path=checkpoint,
        device=_get_device(),
    )


def build_sam2_t(checkpoint=None):
    return build_sam2(
        "sam2_hiera_t.yaml",
        ckpt_path=checkpoint,
        device=_get_device(),
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "sam2_l": build_sam2_l,
    "sam2_bp": build_sam2_bp,
    "sam2_s": build_sam2_s,
    "sam2_t": build_sam2_t,
}

sam_predictor_registry = {
    "default": SamPredictor,
    "vit_h": SamPredictor,
    "vit_l": SamPredictor,
    "vit_b": SamPredictor,
    "vit_t": SamPredictor,
    "sam2_l": SAM2ImagePredictor,
    "sam2_bp": SAM2ImagePredictor,
    "sam2_s": SAM2ImagePredictor,
    "sam2_t": SAM2ImagePredictor,
}


def get_sam_model(model_type: ModelType, checkpoint_url: Optional[str] = None):
    """
    Returns a SAM model.
    :param model_type: Model type.
    :param checkpoint_url: Checkpoint URL.
    :return: SAM model.
    """
    sam = sam_model_registry[model_type]()
    if checkpoint_url is None:
        checkpoint_url = DEFAULT_CHECKPOINT_URLS[model_type]
    state_dict = load_state_dict_from_url(
        url=checkpoint_url,
        model_dir=str(Path(SAMAPI_ROOT_DIR) / model_type.name),
        cancel_filepath=SAMAPI_CANCEL_FILE,
        map_location=torch.device("cpu"),
    )[0]
    if model_type in (
        ModelType.sam2_l,
        ModelType.sam2_bp,
        ModelType.sam2_s,
        ModelType.sam2_t,
    ):
        state_dict = state_dict["model"]
    missing_keys, unexpected_keys = sam.load_state_dict(state_dict)
    if missing_keys:
        logger.error(missing_keys)
        raise RuntimeError()
    if unexpected_keys:
        logger.error(unexpected_keys)
        raise RuntimeError()
    return sam


def _get_device() -> str:
    """
    Selects the device to use for inference, based on what is available.
    :return: device as str
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            warnings.warn(
                "MPS not available because the current MacOS version is not "
                "12.3+ and/or you do not have an MPS-enabled device on this "
                "machine - using CPU for inference"
            )
    else:
        warnings.warn("No GPU support found - using CPU for inference")

    # Make sure that the device is ready
    if device in ("cuda", "mps"):
        try:
            dummy_input = np.zeros((16, 16, 3), dtype=np.uint8)
            SamPredictor(get_sam_model(ModelType.vit_b).to(device=device)).set_image(
                dummy_input
            )
        except Exception as e:
            warnings.warn(
                f"{device} device found but got the error {str(e)} - using CPU for inference"
            )
            device = "cpu"
    return device


device = _get_device()


def register_state_dict_from_url(model_type: ModelType, url: str, name: str) -> bool:
    """
    Registers a state dict from URL.
    :param model_type: Model type.
    :param url: URL.
    :param name: Name.
    :return: True if registered successfully, False otherwise.
    """
    model_dir = Path(SAMAPI_ROOT_DIR) / model_type.name
    if os.path.exists(SAMAPI_CANCEL_FILE):
        os.remove(SAMAPI_CANCEL_FILE)
    try:
        with redirect_stderr(progress_io):
            _, filepath = load_state_dict_from_url(
                url=url,
                model_dir=str(model_dir),
                cancel_filepath=SAMAPI_CANCEL_FILE,
                map_location=torch.device(device),
            )
    except RuntimeError as e:
        if os.path.exists(SAMAPI_CANCEL_FILE):
            os.remove(SAMAPI_CANCEL_FILE)
            return False
        else:
            raise e
    finally:
        if os.path.exists(SAMAPI_CANCEL_FILE):
            os.remove(SAMAPI_CANCEL_FILE)
    json_file = model_dir / f"{Path(filepath).stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"type": model_type, "name": name, "url": url}, f)
    return True


def _register_default_weights():
    """
    Registers default weights.
    """
    for model_type, checkpoint_url in DEFAULT_CHECKPOINT_URLS.items():
        register_state_dict_from_url(model_type, checkpoint_url, f"default")


# Registers default weights at startup.
_register_default_weights()

# global variables
last_sam_type = ModelType.vit_t
last_checkpoint_url = None
predictor = sam_predictor_registry[last_sam_type](
    get_sam_model(model_type=last_sam_type).to(device=device)
)
last_image = None


@app.get("/sam/version/", response_class=PlainTextResponse)
async def get_version():
    """
    Returns the version of the SAM API.
    :return: Version of the SAM API.
    """
    return __version__


class SAMWeightsBody(BaseModel):
    """
    SAM weights body.
    """

    type: ModelType
    name: str
    url: str


def _get_weights_at(p_model_dir: Path, remove_orphans: bool = True):
    """
    Returns a list of the available weights at the given path.
    :param p_model_dir: Path to the model directory.
    :param remove_orphans: Whether to remove orphan files.
    :return: A list of weights.
    """
    if not p_model_dir.exists():
        logger.warning(f"{p_model_dir} does not exist.")
        return []
    paths = (p for p in p_model_dir.iterdir() if p.suffix != ".json")
    weights = []
    for p in sorted(paths, key=os.path.getmtime):
        p_json = p.parent / f"{p.stem}.json"
        if p_json.exists():
            with open(p_json, encoding="utf-8") as file:
                metadata = json.load(file)
            weights.append(
                {
                    "type": metadata["type"],
                    "name": metadata["name"],
                    "url": metadata["url"],
                }
            )
        else:
            logger.warning(
                f"{p_json} is not found. {'Remove' if remove_orphans else 'Skip'} {p}."
            )
            if remove_orphans:
                p.unlink()
    return weights


@app.get("/sam/weights/")
async def get_weights(type: Optional[ModelType] = None):
    """
    Returns a list of the available weights.
    :param type: Model type.
    :return: A list of weights.
    """
    weights = []
    if type is not None:
        return _get_weights_at(Path(SAMAPI_ROOT_DIR) / type.name)
    else:
        for model_type in ModelType:
            weights.extend(_get_weights_at(Path(SAMAPI_ROOT_DIR) / model_type.name))
    return weights


@app.post("/sam/weights/", response_class=PlainTextResponse)
async def register_weights(body: SAMWeightsBody):
    """
    Registers SAM weights.
    :param body: SAM weights body.
    :return: A message indicating whether the registration is successful.
    """
    weights = _get_weights_at(Path(SAMAPI_ROOT_DIR) / body.type.name)
    for weight in weights:
        if weight["name"] == body.name:
            message = f"Model file with the name '{body.name}' already exists. Skip registration."
            logger.warning(message)
            break
        if weight["url"] == body.url:
            message = f"Model file with the url '{body.url}' already exists. Skip registration."
            logger.warning(message)
            break
    else:
        registered = register_state_dict_from_url(body.type, body.url, body.name)
        if registered:
            message = f"{body.name} ({body.url}) is registered."
        else:
            message = f"Cancelled to register {body.name} ({body.url})."
        logger.info(message)
    return message


@app.get("/sam/weights/cancel/", response_class=PlainTextResponse)
async def cancel_download():
    """
    Cancels the download.
    :return: A message indicating that the cancel signal is sent.
    """
    if os.path.exists(SAMAPI_CANCEL_FILE):
        os.remove(SAMAPI_CANCEL_FILE)
    with open(SAMAPI_CANCEL_FILE, "w", encoding="utf-8") as f:
        f.write("cancel")
    return "Cancel signal sent"


@app.get("/sam/progress/")
async def get_progress():
    """
    Returns the progress.
    :return: The progress.
    """
    with open(SAMAPI_STDERR, encoding="utf-8") as f:
        result = f.read()
        if "| " in result:
            message = result.split("| ")[-1]
        else:
            message = None
        if "%" in result:
            percent = int(result.split("%")[0].split(" ")[-1])
        else:
            percent = -1
        return {"message": message, "percent": percent}


class SAMBody(BaseModel):
    """
    SAM body.
    """

    type: Optional[ModelType] = ModelType.vit_h
    bbox: Optional[Tuple[int, int, int, int]] = Field(example=(0, 0, 0, 0))
    point_coords: Optional[Tuple[Tuple[int, int], ...]] = Field(
        example=((0, 0), (1, 0))
    )
    point_labels: Optional[Tuple[int, ...]] = Field(example=(0, 1))
    b64img: str
    b64mask: Optional[str] = None
    multimask_output: bool = False
    checkpoint_url: Optional[str] = None


@app.post("/sam/")
async def predict_sam(body: SAMBody):
    """
    Predicts SAM with prompts.
    :param body: SAM body.
    """
    global last_sam_type
    global last_checkpoint_url
    global predictor
    global last_image
    if body.type != last_sam_type or body.checkpoint_url != last_checkpoint_url:
        predictor = sam_predictor_registry[body.type](
            get_sam_model(body.type, body.checkpoint_url).to(device=device)
        )
        last_sam_type = body.type
        last_checkpoint_url = body.checkpoint_url
        last_image = None
    if last_image != body.b64img:
        image = _parse_image(body)
        predictor.set_image(image)
        last_image = body.b64img
    else:
        logger.info("Keeping the previous image!")

    start_time = time.time_ns()
    masks, quality, _ = predictor.predict(
        point_coords=_parse_point_coords(body),
        point_labels=_parse_point_labels(body),
        box=_parse_bbox(body),
        mask_input=_parse_mask(body),
        multimask_output=body.multimask_output,
    )
    end_time = time.time_ns()
    logger.info(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")

    features = []
    for obj_int, mask in enumerate(masks):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask),
                properties={
                    "object_idx": index_number,
                    "label": "object",
                    "quality": float(quality[index_number]),
                    "sam_model": body.type,
                },
            )
        )
    return features


class SAMAutoMaskBody(BaseModel):
    """
    SAM auto mask body.
    """

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
    checkpoint_url: Optional[str] = None


@app.post("/sam/automask/")
async def automatic_mask_generator(body: SAMAutoMaskBody):
    """
    Generates masks automatically using SAM.
    :param body: SAM auto mask body.
    """
    global last_sam_type
    global last_checkpoint_url
    global predictor
    global last_image
    if body.type != last_sam_type or body.checkpoint_url != last_checkpoint_url:
        predictor = sam_predictor_registry[body.type](
            get_sam_model(body.type, body.checkpoint_url).to(device=device)
        )
        last_sam_type = body.type
        last_checkpoint_url = body.checkpoint_url
        last_image = None
    if last_image != body.b64img:
        image = _parse_image(body)
        predictor.set_image(image)
        last_image = body.b64img
    else:
        logger.info("Keeping the previous image!")

    mask_generator = SamAutomaticMaskGenerator(
        predictor=predictor,
        points_per_side=body.points_per_side,
        points_per_batch=body.points_per_batch,
        pred_iou_thresh=body.pred_iou_thresh,
        stability_score_thresh=body.stability_score_thresh,
        stability_score_offset=body.stability_score_offset,
        box_nms_thresh=body.box_nms_thresh,
        crop_n_layers=body.crop_n_layers,
        crop_nms_thresh=body.crop_nms_thresh,
        crop_overlap_ratio=body.crop_overlap_ratio,
        crop_n_points_downscale_factor=body.crop_n_points_downscale_factor,
        min_mask_region_area=body.min_mask_region_area,
        output_type=body.output_type,
        include_image_edge=body.include_image_edge,
    )

    image = _parse_image(body)
    start_time = time.time_ns()
    masks = mask_generator.generate(image)
    end_time = time.time_ns()
    logger.info(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")

    features = []
    for obj_int, mask in enumerate(masks):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask["segmentation"]),
                properties={
                    "object_idx": index_number,
                    "label": "object",
                    "quality": mask["predicted_iou"],
                    "sam_model": body.type,
                },
            )
        )
    return features


def _parse_image(body: SAMBody):
    """
    Parses the image.
    :param body: SAM body.
    :return: Image as ndarray.
    """
    image = decode_image(body.b64img)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    return image


def _parse_mask(body: SAMBody):
    """
    Parses the mask.
    :param body: SAM body.
    :return: Mask as ndarray.
    """
    return None if body.b64mask is None else decode_image(body.b64mask)


def _parse_bbox(body: SAMBody):
    """
    Parses the bounding box.
    :param body: SAM body.
    :return: Bounding box as ndarray.
    """
    return None if not body.bbox else np.array(body.bbox)[None]


def _parse_point_labels(body: SAMBody):
    """
    Parses the point labels.
    :param body: SAM body.
    :return: Point labels as ndarray.
    """
    return None if not body.point_labels else np.array(body.point_labels)


def _parse_point_coords(body: SAMBody):
    """
    Parses the point coordinates.
    :param body: SAM body.
    :return: Point coordinates as ndarray.
    """
    return None if not body.point_coords else np.array(body.point_coords)
