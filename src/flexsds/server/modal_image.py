"""Shared CUDA development runtime for serving and empirical calibration."""
CUDA_IMAGE = 'nvidia/cuda:13.0.3-devel-ubuntu22.04'


def serving_image():
    import modal
    # Match the M4 runtime base/layers; FlashInfer JIT needs nvcc, not only
    # the CUDA runtime libraries distributed with the torch wheel.
    return (modal.Image.from_registry(CUDA_IMAGE, add_python='3.11')
            .entrypoint([])
            .uv_pip_install('vllm==0.30.0')
            .pip_install('psutil'))
