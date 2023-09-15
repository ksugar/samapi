"""
This module is an extension module of torch.hub.py, in which the download is cancellable.
Original code: https://github.com/pytorch/pytorch/blob/49444c3e546bf240bed24a101e747422d1f8a0ee/torch/hub.py
"""
import errno
import hashlib
import os
import shutil
import sys
import tempfile
from typing import Dict, Optional, Any
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
import warnings
import torch
from torch.hub import get_dir, HASH_REGEX, _is_legacy_zip_format, _legacy_zip_load
from torch.serialization import MAP_LOCATION
from samapi.gdown_extension import download as gdownload

try:
    from tqdm.auto import (
        tqdm,
    )  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore[no-redef]
            def __init__(
                self,
                total=None,
                disable=False,
                unit=None,
                unit_scale=None,
                unit_divisor=None,
            ):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write(
                        "\r{0:.1f}%".format(100 * self.n / float(self.total))
                    )
                sys.stderr.flush()

            def close(self):
                self.disable = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write("\n")


GOOGLE_DRIVE_URL = "https://drive.google.com/"


def download_url_to_file(
    url, dst, hash_prefix=None, progress=True, cancel_filepath=None
):
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
        cancel_filepath (str, optional): Path to a file that, if it exists, signals download cancellation
    Returns:


    Example:
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                if cancel_filepath is not None and os.path.exists(cancel_filepath):
                    sys.stderr.write(
                        'Cancelled download, removing temporary file "{}"'.format(
                            f.name
                        )
                    )
                    sys.stderr.flush()
                    raise RuntimeError("cancelled download")
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    cancel_filepath: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    if url.startswith(GOOGLE_DRIVE_URL):
        output = gdownload(
            url,
            quiet=not progress,
            fuzzy=True,
            cancel_filepath=cancel_filepath,
        )
        filename = os.path.basename(output)
        if file_name is not None:
            filename = file_name
        cached_file = os.path.join(model_dir, filename)
        shutil.move(output, cached_file)
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_name is not None:
            filename = file_name
        cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(
            url,
            cached_file,
            hash_prefix,
            progress=progress,
            cancel_filepath=cancel_filepath,
        )

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    return torch.load(cached_file, map_location=map_location), cached_file
