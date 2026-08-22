import os
import os.path as osp
import base64
import io
import shutil

import numpy as np
import PIL.Image
import PIL.ImageOps

from PyQt6 import QtGui

from ...labeling.logger import logger

EXTRA_IMAGE_EXTENSIONS = (".heic", ".heif")
_PILLOW_HEIF_REGISTERED = False


def ensure_pillow_heif_registered():
    global _PILLOW_HEIF_REGISTERED

    if _PILLOW_HEIF_REGISTERED:
        return True

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        return False

    _PILLOW_HEIF_REGISTERED = True
    return True


def get_supported_image_extensions():
    extensions = {
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    }
    extensions.update(EXTRA_IMAGE_EXTENSIONS)
    return sorted(extensions)


def img_data_to_pil(img_data):
    ensure_pillow_heif_registered()
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_pil_to_data(img_pil):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_data = f.getvalue()
    return img_data


def normalize_image_for_display(img, filename=None):
    """Create a single-channel TIFF preview using logarithmic compression."""
    if img is None:
        return img

    lower_name = (filename or "").lower()
    is_tiff = lower_name.endswith((".tif", ".tiff"))
    if not is_tiff:
        return img

    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = np.asarray(img.convert("L"))
    if arr.ndim != 2:
        return img

    arr = arr.astype(np.float32, copy=False)
    arr = np.log1p(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        normalized = np.zeros(arr.shape, dtype=np.uint8)
    else:
        display_min = float(arr[finite].min())
        display_max = float(arr[finite].max())
        if display_max > display_min:
            normalized = np.clip(
                (arr - display_min)
                / (display_max - display_min)
                * 255,
                0,
                255,
            ).astype(np.uint8)
            normalized[~finite] = 0
        else:
            normalized = np.zeros(arr.shape, dtype=np.uint8)

    return PIL.Image.fromarray(normalized, mode="L")


def pil_to_qimage(img):
    """Convert PIL Image to QImage."""
    if img.mode == "L":
        data = np.asarray(img, dtype=np.uint8)
        height, width = data.shape
        qimage = QtGui.QImage(
            data,
            width,
            height,
            width,
            QtGui.QImage.Format.Format_Grayscale8,
        )
        return qimage

    img = img.convert("RGBA")  # Ensure image is in RGBA format
    data = np.array(img)
    height, width, channel = data.shape
    bytes_per_line = 4 * width
    qimage = QtGui.QImage(
        data,
        width,
        height,
        bytes_per_line,
        QtGui.QImage.Format.Format_RGBA8888,
    )
    return qimage


def img_data_to_qimage(img_data, filename=None):
    is_tiff = bool(filename) and filename.lower().endswith((".tif", ".tiff"))
    if not is_tiff:
        image = QtGui.QImage.fromData(img_data)
        if not image.isNull():
            return image

    if not is_tiff and (
        not filename or not filename.lower().endswith(EXTRA_IMAGE_EXTENSIONS)
    ):
        return QtGui.QImage.fromData(img_data)

    try:
        pil_img = img_data_to_pil(img_data)
        pil_img = normalize_image_for_display(pil_img, filename)
        return pil_to_qimage(pil_img).copy()
    except Exception:
        if is_tiff:
            return QtGui.QImage()
        return QtGui.QImage.fromData(img_data)


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, "PNG")
            f.seek(0)
            return f.read()


def get_pil_img_dim(img_path):
    """
    Get the dimensions of a PIL image.

    Args:
        img_path (str or bytes or PIL.Image.Image): The path to the image file or the image data.

    Returns:
        tuple: The dimensions of the image (width, height).
    """
    try:
        ensure_pillow_heif_registered()
        if isinstance(img_path, str):
            with PIL.Image.open(img_path) as img:
                return img.size[0], img.size[1]
        elif isinstance(img_path, bytes):
            with PIL.Image.open(io.BytesIO(img_path)) as img:
                return img.size[0], img.size[1]
        elif isinstance(img_path, PIL.Image.Image):
            return img_path.size[0], img_path.size[1]
        else:
            raise ValueError(f"Invalid image path type: {type(img_path)}")

    except Exception as e:
        logger.error(
            f"Error reading image dimensions from {img_path}: {str(e)}"
        )
        raise


def check_img_exif(filename):
    """Check if image needs EXIF orientation correction"""
    try:
        ensure_pillow_heif_registered()
        with PIL.Image.open(filename) as img:
            exif = img.getexif()
            orientation = exif.get(0x0112, 1)
            return orientation not in (1, None)

    except Exception:
        return False


def process_image_exif(filename):
    """Process image EXIF orientation."""
    try:
        ensure_pillow_heif_registered()
        with PIL.Image.open(filename) as img:
            exif = img.getexif()
            orientation = exif.get(0x0112, 1)
            if orientation in (1, None):
                return

            corrected_img = PIL.ImageOps.exif_transpose(img)

            backup_dir = osp.join(
                osp.dirname(osp.dirname(filename)),
                "x-anylabeling-exif-backup",
            )
            os.makedirs(backup_dir, exist_ok=True)
            backup_filename = osp.join(backup_dir, osp.basename(filename))
            shutil.copy2(filename, backup_filename)
            corrected_img.save(filename)

    except Exception as e:
        logger.error(f"Error processing EXIF orientation for {filename}: {e}")
