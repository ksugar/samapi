"""
This is a main.py file for the FastAPI app.
"""

import base64
from contextlib import redirect_stderr
from enum import Enum
from functools import partial
import io
from io import TextIOWrapper
import json
import logging
import os
import sys

# On macOS with MPS, enable PyTorch MPS CPU fallback for unsupported ops.
# This must be set before any module that imports torch is loaded.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from geojson import Feature
import numpy as np
from PIL import Image
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

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

from samapi import __version__
from samapi.hub_extension import load_state_dict_from_url
from samapi.utils import (
    decode_image,
    mask_to_geometry,
    normalize_bbox,
    prepare_masks_for_visualization,
)

# SAM3 is optional and may not be available due to gating on HuggingFace
SAM3_AVAILABLE = False
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    SAM3_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger_init = logging.getLogger("uvicorn")
    logger_init.warning(
        "SAM3 is not available. The server will continue without SAM3 support."
        f"Error: {e}",
    )

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())
logger = logging.getLogger("uvicorn")
if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG":
    # log all debug messages
    logger.setLevel(logging.DEBUG)

try:
    Image.MAX_IMAGE_PIXELS = int(
        os.getenv("PIL_MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    )
except (TypeError, ValueError):
    logger.warning(
        "PIL.Image.MAX_IMAGE_PIXELS is set to None, potentially exposing the system to "
        + "decompression bomb attacks."
    )
    Image.MAX_IMAGE_PIXELS = None

SAMAPI_ROOT_DIR = os.getenv("SAMAPI_ROOT_DIR", str(Path.home() / ".samapi"))


SAMAPI_STDERR = SAMAPI_ROOT_DIR + "/samapi.stderr"
SAMAPI_CANCEL_FILE = SAMAPI_ROOT_DIR + "/samapi.cancel"

SAM3_BPE_FILE = os.getenv(
    "SAM3_BPE_FILE",
    str(Path(__file__).parent.parent / "sam3_bpes" / "bpe_simple_vocab_16e6.txt.gz"),
)

DEFAULT_WEIGHT_NAME = "default"


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
    sam3 = "sam3"  # Optional; may not be available due to HuggingFace gating


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
if SAM3_AVAILABLE:
    DEFAULT_CHECKPOINT_URLS[ModelType.sam3] = (
        "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "sam2_l": partial(build_sam2, config_file="sam2_hiera_l.yaml", device="cpu"),
    "sam2_bp": partial(build_sam2, config_file="sam2_hiera_b+.yaml", device="cpu"),
    "sam2_s": partial(build_sam2, config_file="sam2_hiera_s.yaml", device="cpu"),
    "sam2_t": partial(build_sam2, config_file="sam2_hiera_t.yaml", device="cpu"),
    "sam2_l_v": partial(
        build_sam2_video_predictor, config_file="sam2_hiera_l.yaml", device="cpu"
    ),
    "sam2_bp_v": partial(
        build_sam2_video_predictor, config_file="sam2_hiera_b+.yaml", device="cpu"
    ),
    "sam2_s_v": partial(
        build_sam2_video_predictor, config_file="sam2_hiera_s.yaml", device="cpu"
    ),
    "sam2_t_v": partial(
        build_sam2_video_predictor, config_file="sam2_hiera_t.yaml", device="cpu"
    ),
}
if SAM3_AVAILABLE:
    # Note: SAM3 model building will be done in get_sam_model()
    pass

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
if SAM3_AVAILABLE:
    sam_predictor_registry["sam3"] = Sam3Processor


def get_sam_model(
    model_type: ModelType, checkpoint_url: Optional[str] = None, is_video: bool = False
):
    """
    Returns a SAM model.
    :param model_type: Model type.
    :param checkpoint_url: Checkpoint URL.
    :return: SAM model.
    """
    if model_type in (ModelType.sam3,):
        if not SAM3_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "SAM3 is not available. Please ensure SAM3 dependencies are "
                    "installed and access to the SAM3 model on HuggingFace has "
                    "been granted."
                ),
            )
        if checkpoint_url is None:
            checkpoint_url = DEFAULT_CHECKPOINT_URLS[model_type]
        checkpoint_path = load_state_dict_from_url(
            url=checkpoint_url,
            model_dir=str(Path(SAMAPI_ROOT_DIR) / model_type.name),
            cancel_filepath=SAMAPI_CANCEL_FILE,
            map_location=torch.device("cpu"),
        )[1]
        if is_video:
            return build_sam3_video_predictor(
                bpe_path=SAM3_BPE_FILE,
                checkpoint_path=checkpoint_path,
                apply_temporal_disambiguation=False,
            )
        else:
            return build_sam3_image_model(
                bpe_path=SAM3_BPE_FILE,
                checkpoint_path=checkpoint_path,
            )
    model_key = model_type.value + ("_v" if is_video else "")
    sam = sam_model_registry[model_key]()
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
                f"{device} device found but got the error {str(e)} "
                + "- using CPU for inference"
            )
            device = "cpu"
    return device


device = _get_device()


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
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                    else:
                        # fallback: attempt unlink and let exception be handled
                        p.unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to remove orphan {p}: {type(e).__name__}: {e}"
                    )
    return weights


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
    logger.info(
        "Registering default weights. This step may take a while for the first time."
    )
    for model_type, checkpoint_url in DEFAULT_CHECKPOINT_URLS.items():
        try:
            register_state_dict_from_url(
                model_type, checkpoint_url, DEFAULT_WEIGHT_NAME
            )
        except Exception as e:
            # Log the error but continue; some models may not be available due to gating
            logger.warning(
                f"Failed to register weights for {model_type}: "
                f"{type(e).__name__}: {e}. "
                f"This model will not be available."
            )


# Registers default weights at startup.
_register_default_weights()

# global variables
last_sam_type = ModelType.vit_t
last_checkpoint_url = None
predictor = sam_predictor_registry[last_sam_type](
    get_sam_model(model_type=last_sam_type).to(device=device)
)
last_image = None
inference_state = None
last_confidence_threshold = 0.5


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
            message = (
                f"Model file with the name '{body.name}' already exists. "
                + "Skip registration."
            )
            logger.warning(message)
            break
        if weight["url"] == body.url:
            message = (
                f"Model file with the url '{body.url}' already exists. "
                + "Skip registration."
            )
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
    bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None, example=(0, 0, 0, 0)
    )
    point_coords: Optional[Tuple[Tuple[int, int], ...]] = Field(
        default=None, example=((0, 0), (1, 0))
    )
    point_labels: Optional[Tuple[int, ...]] = Field(default=None, example=(0, 1))
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


class SAM3Body(BaseModel):
    """
    SAM3 body.
    """

    type: Optional[ModelType] = ModelType.vit_h
    text_prompt: Optional[str] = Field(example="cat")
    positive_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = Field(
        example=[(0, 0, 0, 0)]
    )
    negative_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = Field(
        example=[(0, 0, 0, 0)]
    )
    b64img: str
    checkpoint_url: Optional[str] = None
    reset_prompts: bool = False
    confidence_threshold: float = 0.5


def _to_norm_box_cxcywh(bbox_xywh: Tuple[int, int, int, int], width: int, height: int):
    """
    Converts bbox from xywh to normalized cxcywh.
    :param bbox_xywh: Bbox in xywh format.
    :param width: Image width.
    :param height: Image height.
    :return: Normalized bbox in cxcywh format.
    """
    box_input_xywh = torch.tensor(bbox_xywh).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    return norm_box_cxcywh


@app.post("/sam/sam3/")
async def predict_sam3(body: SAM3Body):
    """
    Predicts SAM with prompts.
    :param body: SAM3 body.
    """
    global last_sam_type
    global last_checkpoint_url
    global predictor
    global last_image
    global inference_state
    global last_confidence_threshold
    if body.type != last_sam_type or body.checkpoint_url != last_checkpoint_url:
        predictor = sam_predictor_registry[body.type](
            get_sam_model(body.type, body.checkpoint_url).to(
                device="cpu" if device == "mps" else device
            )
        )
        last_sam_type = body.type
        last_checkpoint_url = body.checkpoint_url
        last_image = None
    if last_image != body.b64img:
        image_data = base64.b64decode(body.b64img)
        image = Image.open(io.BytesIO(image_data))
        # log image size and type
        logger.info(f"Image size: {image.size}, mode: {image.mode}")
        inference_state = predictor.set_image(image, state=inference_state)
        last_image = body.b64img
    else:
        logger.info("Keeping the previous image!")
    if inference_state is None:
        raise HTTPException(
            status_code=500,
            detail="Inference state is not initialized. Please set the image first.",
        )
    width = inference_state["original_width"]
    height = inference_state["original_height"]

    start_time = time.time_ns()
    if body.reset_prompts:
        predictor.reset_all_prompts(inference_state)
    if last_confidence_threshold != body.confidence_threshold:
        predictor.set_confidence_threshold(
            body.confidence_threshold,
            inference_state,
        )
        last_confidence_threshold = body.confidence_threshold
    inference_state = predictor.set_text_prompt(
        state=inference_state,
        prompt=body.text_prompt,
    )
    logger.info(f"Text prompt set: {body.text_prompt}")
    if body.positive_bboxes is not None:
        logger.info(f"Number of positive boxes: {len(body.positive_bboxes)}")
        for bbox in body.positive_bboxes:
            inference_state = predictor.add_geometric_prompt(
                state=inference_state,
                box=_to_norm_box_cxcywh(bbox, width, height),
                label=True,
            )
    if body.negative_bboxes is not None:
        logger.info(f"Number of negative boxes: {len(body.negative_bboxes)}")
        for bbox in body.negative_bboxes:
            inference_state = predictor.add_geometric_prompt(
                state=inference_state,
                box=_to_norm_box_cxcywh(bbox, width, height),
                label=False,
            )
    end_time = time.time_ns()
    logger.info(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")

    features = []
    logger.info(f"Number of masks: {len(inference_state['masks'])}")

    for obj_int, mask in enumerate(inference_state["masks"]):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask.squeeze(0).cpu()),
                properties={
                    "object_idx": index_number,
                    "label": "object",
                    "quality": float(inference_state["scores"][index_number]),
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

    if body.type in (
        ModelType.sam2_l,
        ModelType.sam2_bp,
        ModelType.sam2_s,
        ModelType.sam2_t,
    ):
        mask_generator = SAM2AutomaticMaskGenerator(
            model=predictor.model,
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
            multimask_output="Multi" in body.output_type,
        )
    else:
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


@app.post("/sam/upload/")
async def video_upload(
    dirname: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Uploads a file for SAM video predictor.
    :param body: SAM upload body.
    """
    file_path = Path(SAMAPI_ROOT_DIR) / "videos" / dirname / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"filename": file.filename}


class SAMVideoBody(BaseModel):
    """
    SAM video predicotr body.
    """

    type: Optional[ModelType] = ModelType.sam2_t
    dirname: Optional[str] = None
    b64imgs: Optional[List[str]] = None
    axes: str = "XYT"
    plane_position: Optional[int] = 0
    start_frame_idx: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None
    objs: Optional[Dict[int, List[Dict]]] = Field(
        example={
            0: {
                "obj_id": 0,
                "point_coords": ((0, 0), (1, 0)),
                "point_labels": (0, 1),
                "bbox": [0, 0, 10, 10],
            }
        }
    )
    checkpoint_url: Optional[str] = None


def get_video_segments(
    predictor: SAM2ImagePredictor,
    video_path: str,
    objs_dict: Dict[int, List[Dict]],
    start_frame_idx: int,
    max_frame_num_to_track: int,
):
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)
    for ann_frame_idx, objs in objs_dict.items():
        for obj in objs:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj.get("obj_id"),
                points=obj.get("point_coords"),
                labels=obj.get("point_labels"),
                box=obj.get("bbox"),
            )
    video_segments = {}
    start_time = time.time_ns()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    end_time = time.time_ns()
    logger.info(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")
    return video_segments


@app.post("/sam/video/")
async def video_predictor(body: SAMVideoBody):
    """
    Generates masks from video automatically using SAM2.
    :param body: SAM video body.
    """
    if body.type not in (
        ModelType.sam2_l,
        ModelType.sam2_bp,
        ModelType.sam2_s,
        ModelType.sam2_t,
    ):
        raise HTTPException(
            status_code=404,
            detail="Only SAM2 models are supported for video prediction.",
        )
    predictor = get_sam_model(
        body.type,
        body.checkpoint_url,
        is_video=True,
    ).to(device=device)

    if body.dirname is not None:
        dir_path = Path(SAMAPI_ROOT_DIR) / "videos" / body.dirname
        try:
            video_segments = get_video_segments(
                predictor,
                video_path=str(dir_path),
                objs_dict=body.objs,
                start_frame_idx=body.start_frame_idx,
                max_frame_num_to_track=body.max_frame_num_to_track,
            )
        finally:
            shutil.rmtree(dir_path)
    elif body.b64imgs is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Saving images to {temp_dir}")
            for idx, b64img in enumerate(body.b64imgs):
                # Decode the base64 image
                image_data = base64.b64decode(b64img)
                image = Image.open(io.BytesIO(image_data))

                # Convert to RGB if the image is grayscale
                if image.mode == "L":
                    image = image.convert("RGB")

                # Save the image as JPEG in the temporary directory
                image_path = os.path.join(temp_dir, f"{idx}.jpg")
                image.save(image_path, "JPEG")
            logger.info(f"Saved {len(body.b64imgs)} images to {temp_dir}")
            video_segments = get_video_segments(
                predictor,
                video_path=temp_dir,
                objs_dict=body.objs,
                start_frame_idx=body.start_frame_idx,
                max_frame_num_to_track=body.max_frame_num_to_track,
            )
    else:
        raise HTTPException(
            status_code=404,
            detail="Either dirname or b64imgs must be provided.",
        )

    features = []
    for frame_idx, masks in video_segments.items():
        if body.axes == "XYZ":
            plane = {
                "c": -1,
                "z": frame_idx,
                "t": body.plane_position,
            }
        if body.axes == "XYT":
            plane = {
                "c": -1,
                "z": body.plane_position,
                "t": frame_idx,
            }
        for obj_id, mask in masks.items():
            if np.any(mask):
                geometry = mask_to_geometry(mask[0])
                geometry["plane"] = plane
                features.append(
                    Feature(
                        geometry=geometry,
                        properties={
                            "object_idx": obj_id,
                            "label": "object",
                            "sam_model": body.type,
                        },
                    )
                )
    return features


class SAM3VideoBody(BaseModel):
    """
    SAM3 video predicotr body.
    """

    type: Optional[ModelType] = ModelType.sam2_t
    dirname: Optional[str] = None
    b64imgs: Optional[List[str]] = None
    axes: str = "XYT"
    plane_position: Optional[int] = 0
    start_frame_idx: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None
    objs: Optional[Dict[int, List[Dict]]] = Field(
        example={
            0: [
                {
                    "obj_id": 0,
                    "text": "cell",
                    "boxes_xywh": [[0, 0, 10, 10], [10, 10, 5, 5]],
                    "box_labels": (0, 1),
                }
            ]
        }
    )
    checkpoint_url: Optional[str] = None


def propagate_in_video(
    predictor,
    session_id,
    start_frame_index=0,
    max_frame_num_to_track=None,
):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="both",
            start_frame_index=start_frame_index,
            max_frame_num_to_track=max_frame_num_to_track,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def get_sam3_video_segments(
    predictor: Sam3VideoPredictor,
    video_path: str,
    objs_dict: Dict[int, List[Dict]],
    start_frame_idx: int,
    max_frame_num_to_track: int,
):
    # Initialize session
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    try:
        # Get image size
        session = predictor._get_session(session_id=session_id)
        inference_state = session["state"]
        width = inference_state["orig_width"]
        height = inference_state["orig_height"]

        # Reset session
        _ = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        # Add prompts
        logger.debug(f"Adding prompts: {objs_dict}")
        for ann_frame_idx, objs in objs_dict.items():
            for obj in objs:
                logger.debug(f"Adding prompt at frame {ann_frame_idx}: {obj}")
                norm_boxes = list(
                    map(
                        lambda box: normalize_bbox(box, width, height),
                        obj.get("boxes_xywh", None),
                    )
                )
                logger.debug(f"Normalized boxes: {norm_boxes}")
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=ann_frame_idx,
                        text=obj.get("text", None),
                        points=None,
                        point_labels=None,
                        bounding_boxes=norm_boxes,
                        bounding_box_labels=obj.get("box_labels", None),
                        obj_id=obj.get("obj_id", None),
                    )
                )
                out = response["outputs"]
                logger.debug(f"Outputs after adding prompt: {out}")

        # Propagate in video
        start_time = time.time_ns()
        outputs_per_frame = propagate_in_video(
            predictor,
            session_id,
            start_frame_idx,
            max_frame_num_to_track,
        )
        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
        logger.debug(f"Outputs per frame: {outputs_per_frame}")
        end_time = time.time_ns()
        logger.info(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")
    finally:
        _ = predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        predictor.shutdown()
    return outputs_per_frame


@app.post("/sam/sam3video/")
async def sam3_video_predictor(body: SAM3VideoBody):
    """
    Generates masks from video automatically using SAM3.
    :param body: SAM3 video body.
    """
    if body.type not in (ModelType.sam3,):
        raise HTTPException(
            status_code=404,
            detail="Only SAM3 models are supported for SAM3 video prediction.",
        )
    predictor = get_sam_model(
        body.type,
        body.checkpoint_url,
        is_video=True,
    )

    predictor.model.score_threshold_detection = 0.5

    if body.dirname is not None:
        dir_path = Path(SAMAPI_ROOT_DIR) / "videos" / body.dirname
        try:
            video_segments = get_sam3_video_segments(
                predictor,
                video_path=str(dir_path),
                objs_dict=body.objs,
                start_frame_idx=body.start_frame_idx,
                max_frame_num_to_track=body.max_frame_num_to_track,
            )
        finally:
            shutil.rmtree(dir_path)
    elif body.b64imgs is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Saving images to {temp_dir}")
            for idx, b64img in enumerate(body.b64imgs):
                # Decode the base64 image
                image_data = base64.b64decode(b64img)
                image = Image.open(io.BytesIO(image_data))

                # Convert to RGB if the image is grayscale
                if image.mode == "L":
                    image = image.convert("RGB")

                # Save the image as JPEG in the temporary directory
                image_path = os.path.join(temp_dir, f"{idx}.jpg")
                image.save(image_path, "JPEG")
            logger.info(f"Saved {len(body.b64imgs)} images to {temp_dir}")
            video_segments = get_sam3_video_segments(
                predictor,
                video_path=str(temp_dir),
                objs_dict=body.objs,
            )
    else:
        raise HTTPException(
            status_code=404,
            detail="Either dirname or b64imgs must be provided.",
        )
    logger.debug(f"Video segments: {video_segments}")
    features = []
    for frame_idx, masks in video_segments.items():
        if body.axes == "XYZ":
            plane = {
                "c": -1,
                "z": frame_idx,
                "t": body.plane_position,
            }
        if body.axes == "XYT":
            plane = {
                "c": -1,
                "z": body.plane_position,
                "t": frame_idx,
            }
        for obj_id, mask in masks.items():
            if np.any(mask):
                geometry = mask_to_geometry(mask)
                geometry["plane"] = plane
                features.append(
                    Feature(
                        geometry=geometry,
                        properties={
                            "object_idx": obj_id,
                            "label": "object",
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
        image = np.stack((image,) * 3, axes=-1)
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
