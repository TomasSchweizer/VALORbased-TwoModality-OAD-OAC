import gc
import pynvml
import torch

def gpu_memory_usage(device=0):
    return torch.cuda.memory_allocated(device) / 1024.0**3


def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)


def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3


def memory_cleanup(device):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        #yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        
        if (mem := gpu_memory_usage(device)) > 16.0:
            print("GPU memory usage still high!")
            cnt = 0
            for obj in get_tensors():
                print(obj.name)
                obj.detach()
                obj.grad = None
                obj.storage().resize_(0)
                cnt += 1
            gc.collect()
            torch.cuda.empty_cache()
            usage, cache, misc = gpu_memory_usage_all(device)
            print(
                f"  forcibly cleared {cnt} tensors: {mem:.03f}GB -> {usage:.03f}GB (+{cache:.03f}GB cache, +{misc:.03f}GB misc)"       
            )


def get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue
            
            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue