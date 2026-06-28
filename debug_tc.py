p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

old = (
    '        try:\n'
    '            image_bgr = qt_img_to_rgb_cv_img(image)\n'
    '            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)\n'
    '        except Exception as e:\n'
    '            logger.warning(f"Could not load image: {e}")\n'
    '            return AutoLabelingResult([], replace=self.replace)\n'
    '\n'
    '        try:\n'
    '            sem_prob, embedding = self._sliding_window_infer(image_bgr)\n'
    '        except Exception as e:\n'
    '            logger.warning(f"Inference failed: {e}")\n'
    '            return AutoLabelingResult([], replace=self.replace)'
)

new = (
    '        import traceback\n'
    '        try:\n'
    '            image_bgr = qt_img_to_rgb_cv_img(image)\n'
    '            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)\n'
    '            sem_prob, embedding = self._sliding_window_infer(image_bgr)\n'
    '        except Exception as e:\n'
    '            logger.warning(f"Inference error: {e}\\n{traceback.format_exc()}")\n'
    '            return AutoLabelingResult([], replace=self.replace)'
)

c = c.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(c)
print("OK")
