from time import perf_counter

import torch

# benchmark a+b


def add_ubench(N, repeat, dtype):
    warmup = 1

    size_bytes = None
    if dtype == torch.float32:
        # 4 bytes per float; 2 input tensors
        size_bytes = 8 * N
    elif dtype == torch.bfloat16:
        size_bytes = 4 * N

    latency = 0.0

    for i in range(repeat + warmup):
        input1 = torch.rand([N], dtype=dtype).cuda()
        input2 = torch.rand([N], dtype=dtype).cuda()

        start = perf_counter()

        output = input1 + input2
        torch.cuda.synchronize()

        if i >= warmup:
            latency += perf_counter() - start

        torch.cuda.empty_cache()

    latency = latency / repeat
    gb = size_bytes / 1e9
    gb_per_sec = gb / latency

    print(f"dtype: {dtype}, N: {N}, {gb} GB, latency: {latency} sec, {gb_per_sec} GB/s")
    # print(f"input1: {input1}")
    # print(f"input2: {input2}")
    # print(f"output: {output}")

    return {
        "latency": latency,
        "gb_per_sec": gb_per_sec,
        "gb": gb,
    }


for N in [16777216, 33554432]:
    for dtype in [torch.float32, torch.bfloat16]:
        add_ubench(N=N, repeat=20, dtype=dtype)
