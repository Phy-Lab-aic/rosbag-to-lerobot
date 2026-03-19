"""Minimal torch stub for import compatibility.

Only provides the API surface needed by lerobot's data layer:
- Tensor class (isinstance checks, type annotations)
- dtype/device classes
- Basic tensor creation functions (tensor, stack, cdist, full, zeros, ones, cat)
- nn.Module (isinstance/issubclass checks by datasets library)
- CUDA/MPS/XPU availability checks (always False)
"""

from torch import nn  # noqa: F401

__version__ = "0.0.0+stub"


# --- dtype ---

class dtype:
    def __init__(self, name=""):
        self._name = name

    def __repr__(self):
        return f"torch.{self._name}"

    def __eq__(self, other):
        return isinstance(other, dtype) and self._name == other._name

    def __hash__(self):
        return hash(self._name)


float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
bool = dtype("bool")
bfloat16 = dtype("bfloat16")
long = int64
float = float32
double = float64
half = float16


# --- device ---

class device:
    def __init__(self, type_or_str="cpu", index=None):
        if isinstance(type_or_str, device):
            self.type = type_or_str.type
            self.index = type_or_str.index
        else:
            parts = str(type_or_str).split(":")
            self.type = parts[0]
            self.index = int(parts[1]) if len(parts) > 1 else index

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        return False

    def __hash__(self):
        return hash((self.type, self.index))

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type


# --- Tensor ---

class Tensor:
    def __init__(self, *args, **kwargs):
        self._data = None

    def numpy(self):
        import numpy as np
        return np.array([])

    def item(self):
        return 0

    def tolist(self):
        return []

    def type(self, dtype=None):
        if dtype is None:
            return "torch.Tensor"
        return Tensor()

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def long(self):
        return self

    def int(self):
        return self

    def clone(self):
        return Tensor()

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, *shape):
        return Tensor()

    def reshape(self, *shape):
        return Tensor()

    def unsqueeze(self, dim):
        return Tensor()

    def squeeze(self, dim=None):
        return Tensor()

    def permute(self, *dims):
        return Tensor()

    def size(self, dim=None):
        if dim is not None:
            return 0
        return (0,)

    @property
    def shape(self):
        return (0,)

    @property
    def dtype(self):
        return float32

    @property
    def device(self):
        return device("cpu")

    @property
    def ndim(self):
        return 0

    @property
    def T(self):
        return self

    def dim(self):
        return 0

    def __len__(self):
        return 0

    def __getitem__(self, key):
        return Tensor()

    def __repr__(self):
        return "tensor([])"

    def __add__(self, other):
        return Tensor()

    def __radd__(self, other):
        return Tensor()

    def __sub__(self, other):
        return Tensor()

    def __rsub__(self, other):
        return Tensor()

    def __mul__(self, other):
        return Tensor()

    def __rmul__(self, other):
        return Tensor()

    def __truediv__(self, other):
        return Tensor()

    def __rtruediv__(self, other):
        return Tensor()

    def __neg__(self):
        return Tensor()

    def __bool__(self):
        return False

    def __eq__(self, other):
        return Tensor()

    def __ne__(self, other):
        return Tensor()

    def __lt__(self, other):
        return Tensor()

    def __le__(self, other):
        return Tensor()

    def __gt__(self, other):
        return Tensor()

    def __ge__(self, other):
        return Tensor()


class BoolTensor(Tensor):
    pass


class Generator:
    """Stub for torch.Generator (needed by datasets library pickling)."""

    def __init__(self, device=None):
        pass

    def manual_seed(self, seed):
        return self

    def seed(self):
        return 0


# --- Tensor creation functions ---

def tensor(data=None, dtype=None, device=None, requires_grad=False):
    return Tensor()

def zeros(*size, dtype=None, device=None, requires_grad=False, out=None):
    return Tensor()

def ones(*size, dtype=None, device=None, requires_grad=False, out=None):
    return Tensor()

def full(size, fill_value, dtype=None, device=None, requires_grad=False, out=None):
    return Tensor()

def empty(*size, dtype=None, device=None, requires_grad=False, out=None):
    return Tensor()

def stack(tensors, dim=0, out=None):
    return Tensor()

def cat(tensors, dim=0, out=None):
    return Tensor()

def cdist(x1, x2, p=2.0):
    return Tensor()

def from_numpy(ndarray):
    return Tensor()

def arange(*args, **kwargs):
    return Tensor()

def linspace(*args, **kwargs):
    return Tensor()

def no_grad():
    import contextlib
    return contextlib.nullcontext()

def is_tensor(obj):
    return isinstance(obj, Tensor)

def as_tensor(data, dtype=None, device=None):
    return Tensor()

def set_default_dtype(d):
    pass

def manual_seed(seed):
    pass

def set_num_threads(n):
    pass

def get_num_threads():
    return 1

def compile(model=None, **kwargs):
    if model is not None:
        return model
    def decorator(fn):
        return fn
    return decorator


# --- Sub-modules (namespace stubs) ---

class _CudaModule:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def set_device(device):
        pass

    @staticmethod
    def manual_seed(seed):
        pass

    @staticmethod
    def manual_seed_all(seed):
        pass

    class amp:
        @staticmethod
        def autocast(enabled=True, dtype=None):
            import contextlib
            return contextlib.nullcontext()

        class GradScaler:
            def __init__(self, *args, **kwargs):
                pass
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                pass
            def update(self):
                pass


class _BackendsModule:
    class mps:
        @staticmethod
        def is_available():
            return False

    class cudnn:
        enabled = False
        benchmark = False
        deterministic = False


class _XpuModule:
    @staticmethod
    def is_available():
        return False


cuda = _CudaModule()
backends = _BackendsModule()
xpu = _XpuModule()
