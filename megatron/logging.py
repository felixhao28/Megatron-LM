
import torch

def print_with_rank(msg: str):
    from megatron.core.parallel_state import get_tensor_model_parallel_rank
    print(f"rank_{get_tensor_model_parallel_rank()} {msg}")

import os

_log_aix = os.environ.get("AIXDEBUG", "") != ""
def log_tensor_aix(input: torch.Tensor, msg: str, bias=None,
                   sequence_parallel=False, tensor_parallel=False, print_value=False, only_on_rank_0=False):
    if not _log_aix:
        return
    from megatron.core.parallel_state import get_tensor_model_parallel_rank
    if sequence_parallel:
        # merge sequence_parallel
        assert not tensor_parallel
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_world_size,
            get_tensor_model_parallel_group,
            get_global_memory_buffer,
        )

        # log_tensor_aix(input, "not reduced:" + msg, bias)

        world_size = get_tensor_model_parallel_world_size()
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = \
            get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        torch.distributed.all_gather_into_tensor(
            all_gather_buffer,
            input,
            group=get_tensor_model_parallel_group())
        tensor = all_gather_buffer
    elif tensor_parallel:
        # merge tensor_parallel
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
        tensor = gather_from_tensor_model_parallel_region(input)
    else:
        tensor = input
    rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if only_on_rank_0 and not rank0:
        return
    shape = tensor.shape
    if bias is not None:
        tensor = tensor + bias.view([1] * (len(tensor.shape) - 1) + [-1])
    t = tensor.double()
    mean = float(torch.mean(t))
    max = float(torch.max(t))
    t = t.flatten()
    idx = t.shape[0] // 3
    if sequence_parallel or tensor_parallel:
        from megatron.utils import print_rank_0
        print_rank_0(f"rank_{get_tensor_model_parallel_rank()} {msg}: mean={mean} max={max} [{idx}]={t[idx]} shape={shape}")
        if print_value:
            print_rank_0(tensor)
    else:
        print(f"rank_{get_tensor_model_parallel_rank()} {msg}: mean={mean} max={max} [{idx}]={t[idx]} shape={shape}")
        if print_value:
            print(tensor)
