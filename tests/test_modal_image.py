"""Regression for the live FlashInfer nvcc startup failure."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys
from flexsds.server.modal_image import serving_image, CUDA_IMAGE


def test_cuda_compiler_base_and_pinned_vllm_are_shared(monkeypatch):
    image = Mock()
    for method in ('entrypoint','uv_pip_install','pip_install'):
        getattr(image,method).return_value = image
    factory = Mock(return_value=image)
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(Image=SimpleNamespace(from_registry=factory)))
    assert serving_image() is image
    assert 'devel' in CUDA_IMAGE
    factory.assert_called_once_with(CUDA_IMAGE,add_python='3.11')
    image.entrypoint.assert_called_once_with([])
    image.uv_pip_install.assert_called_once_with('vllm==0.30.0')
