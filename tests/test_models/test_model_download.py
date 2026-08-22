import ssl

import pytest

from anylabeling.services.auto_labeling import model as auto_model
from anylabeling.services.auto_labeling.model import Model


class DummyModel(Model):
    def predict_shapes(self, image, filename=None):
        return None

    def unload(self):
        pass


class FakeResponse:
    headers = {"Content-Length": "6"}

    def __init__(self):
        self.chunks = [b"secure", b""]

    def read(self, _):
        return self.chunks.pop(0)


def test_download_with_retry_keeps_tls_certificate_verification(
    tmp_path, monkeypatch
):
    captured_contexts = []

    def fake_urlopen(req, timeout, context=None):
        captured_contexts.append(context)
        if context is not None:
            assert context.check_hostname
            assert context.verify_mode == ssl.CERT_REQUIRED
        return FakeResponse()

    monkeypatch.setattr(auto_model.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(auto_model, "get_config", lambda: {})

    model = DummyModel({}, lambda _: None)
    model.MAX_RETRIES = 1
    dest_path = tmp_path / "model.onnx"

    assert model.download_with_retry(
        "https://example.com/model.onnx", str(dest_path), None
    )
    assert dest_path.read_bytes() == b"secure"
    assert captured_contexts


def test_get_model_abs_path_raises_exception_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_model, "get_config", lambda: {})

    model = DummyModel({}, lambda _: None)
    config = {"config_file": str(tmp_path / "cfg.yaml"), "weights": "missing.onnx"}

    with pytest.raises(ValueError, match="Model path not found"):
        model.get_model_abs_path(config, "weights")


def test_get_model_abs_path_resolves_dist_bundle_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_model, "get_config", lambda: {})
    monkeypatch.setattr(auto_model.sys, "_MEIPASS", "", raising=False)

    bundle_dir = tmp_path / "dist"
    bundle_file = (
        bundle_dir
        / "anylabeling"
        / "services"
        / "auto_labeling"
        / "models"
        / "plate_yolo"
        / "best.pt"
    )
    bundle_file.parent.mkdir(parents=True)
    bundle_file.write_bytes(b"plate-model")
    monkeypatch.setattr(
        auto_model.sys,
        "executable",
        str(bundle_dir / "X-AnyLabeling.exe"),
        raising=False,
    )

    model = DummyModel({}, lambda _: None)
    config = {
        "config_file": str(tmp_path / "cfg.yaml"),
        "plate_model_path": "anylabeling/services/auto_labeling/models/plate_yolo/best.pt",
    }

    assert model.get_model_abs_path(config, "plate_model_path") == str(bundle_file)
