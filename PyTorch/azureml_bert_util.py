from horovod.torch.mpi_ops import allreduce
from horovod.torch.compression import Compression
import horovod.torch as hvd
import torch
import time

from collections import OrderedDict
try: 
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        print("Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.")
        warned_flatten = True
    from torch._utils import _flatten_dense_tensors as flatten
    from torch._utils import _unflatten_dense_tensors as unflatten


def adjust_gradient_accumulation_steps(x, initial_steps, target_steps, warmup):
    return min(max(int(x/warmup*target_steps), initial_steps), target_steps)


def init_communicator(local_rank, process_count_per_node):
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:4321',
                            world_size=process_count_per_node, 
                            rank=local_rank)

    node_info_tensor = torch.IntTensor(2).fill_(-1).cuda(local_rank)
    if local_rank == 0:
        try:
            hvd.init()
            node_info_tensor[0].fill_(hvd.size())
            node_info_tensor[1].fill_(hvd.rank())
        except:
            node_info_tensor[0].fill_(1)
            node_info_tensor[1].fill_(0)
    
    torch.distributed.broadcast_multigpu([node_info_tensor], 0)

    node_count = node_info_tensor[0].item()
    node_rank = node_info_tensor[1].item()
    
    world_size = process_count_per_node * node_count
    rank = local_rank + node_rank * process_count_per_node
    return node_count, world_size, node_rank, rank


def sync_grads(model, local_rank, node_count, ngpu, fp16=False):
    if node_count == 1 and ngpu == 1: return

    grads = [param.grad.data for param in model.parameters() if param.grad is not None]
    buckets = OrderedDict()
    for tensor in grads:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
  
    compression = hvd.Compression.fp16 if fp16 else hvd.Compression.none
    for tp in buckets:
        bucket = buckets[tp]
        coalesced = flatten(bucket) / ngpu / node_count 
        if node_count > 1:
            # intra-node grad reduce to device 0
            torch.distributed.reduce_multigpu([coalesced], 0)
            # inter-node grad all-reduce in device 0
            if local_rank == 0:
                coalesced = allreduce(tensor=coalesced, average=False, compression=compression)
            # intra-node grad broadcast from device 0
            torch.distributed.broadcast_multigpu([coalesced], 0)
        else:
            torch.distributed.all_reduce_multigpu([coalesced])
        for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
            buf.copy_(synced)


def broadcast_parameters(model, local_rank, node_count):
    if local_rank == 0 and node_count > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    for param in model.parameters():
        torch.distributed.broadcast_multigpu([param], 0)


def wait_for_all_wokrers(local_rank):
    if local_rank == 0:
        allreduce(torch.IntTensor(1).fill_(-1))
    torch.distributed.broadcast_multigpu([torch.IntTensor(1).fill_(-1).cuda(local_rank)], 0)
    time.sleep(5)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
