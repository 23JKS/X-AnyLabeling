import unittest
import io
from unittest import mock

import numpy as np
from PIL import Image
from PyQt6 import QtGui

from anylabeling.views.labeling.utils.image import (
    get_supported_image_extensions,
    img_data_to_qimage,
)


class TestImageUtils(unittest.TestCase):

    def test_supported_image_extensions_include_heif_variants(self):
        extensions = get_supported_image_extensions()

        self.assertIn(".heic", extensions)
        self.assertIn(".heif", extensions)

    def test_img_data_to_qimage_falls_back_to_pil(self):
        with mock.patch.object(
            QtGui.QImage,
            "fromData",
            return_value=QtGui.QImage(),
        ):
            with mock.patch(
                "anylabeling.views.labeling.utils.image.img_data_to_pil",
                return_value=Image.new("RGB", (2, 3), "white"),
            ):
                image = img_data_to_qimage(
                    b"not-a-qt-image", "sample.heic"
                )

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 3))

    def test_img_data_to_qimage_skips_pil_for_non_heif(self):
        with mock.patch.object(
            QtGui.QImage,
            "fromData",
            return_value=QtGui.QImage(),
        ):
            with mock.patch(
                "anylabeling.views.labeling.utils.image.img_data_to_pil"
            ) as mocked_img_data_to_pil:
                image = img_data_to_qimage(
                    b"not-a-qt-image", "sample.jpg"
                )

        self.assertTrue(image.isNull())
        mocked_img_data_to_pil.assert_not_called()

    def test_img_data_to_qimage_normalizes_tiff_before_display(self):
        pixels = np.ones((10, 10), dtype=np.uint16) * 100
        pixels[0, 0] = 10000
        pixels[5, 5] = 5000
        pixels[5, 6] = 5000
        pixels[5, 7] = 5000
        source = Image.fromarray(pixels)
        buffer = io.BytesIO()
        source.save(buffer, format="TIFF")

        image = img_data_to_qimage(buffer.getvalue(), "sample.tif")

        self.assertFalse(image.isNull())
        self.assertEqual(image.format(), QtGui.QImage.Format.Format_Grayscale8)
        self.assertGreater(image.pixelColor(5, 5).red(), 200)
        self.assertEqual(image.pixelColor(5, 5).red(), image.pixelColor(5, 5).green())
        self.assertEqual(image.pixelColor(5, 5).red(), image.pixelColor(5, 5).blue())
        self.assertEqual(image.pixelColor(1, 1).red(), 0)
