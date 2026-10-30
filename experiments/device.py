"""Device placement for the fan-in runners (NFI_DEVICE=cuda; default cpu, which changes nothing).

The pilots build everything on the CPU and move numpy data through torch.from_numpy and Tensor.numpy. On the GPU:
- models are built on the CPU, so initial weights and masks equal the CPU runs', then moved (`build`);
- torch.from_numpy returns device tensors and Tensor.numpy copies back to the host;
- the default device is the GPU, so tensors created in the update (indices, targets, solver matrices) live there;
- checkpoints are saved as CPU tensors, so the analysis scripts load them unchanged;
- cuDNN and cuBLAS are set to deterministic algorithms (a nondeterministic op raises instead of running).
GPU results do not equal CPU results; compare arms only within one device.

Call `install()` in each worker process, after the fork (CUDA must not be initialised before the pool forks).
"""
import os

import torch

NAME = os.environ.get('NFI_DEVICE', 'cpu')
ACTIVE = NAME != 'cpu'
if ACTIVE:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # read when cuBLAS first initialises


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def install():
    if not ACTIVE:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.set_default_device(NAME)
    if getattr(torch, '_nfi_device_installed', False):
        return
    from_numpy, numpy, save = torch.from_numpy, torch.Tensor.numpy, torch.save
    torch.from_numpy = lambda a: from_numpy(a).to(NAME)
    torch.Tensor.numpy = lambda self, *a, **k: numpy(self.cpu(), *a, **k)
    torch.save = lambda obj, *a, **k: save(to_cpu(obj), *a, **k)
    torch._nfi_device_installed = True


def build(make):
    """make() -> (enc, model), built on the CPU as before and then moved to the device."""
    with torch.device('cpu'):
        enc, model = make()
    if ACTIVE:
        for m in (*enc.values(), model):
            m.to(NAME)
    return enc, model


def info():
    if not ACTIVE:
        return {'device': 'cpu'}
    return {'device': NAME, 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'deterministic_algorithms': True, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
