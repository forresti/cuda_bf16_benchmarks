# cuda_bf16_benchmarks


how to run:

```
make
./main
```

Here is the report it prints on an NVIDIA A10 GPU:
```
run_add() cpu latency: 0.0226018 sec
run_add(): type=fp32, N: 16777216, 0.134218 GB, latency: 0.0018558 sec, 72.3233 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 16777216, 0.0671089 GB, latency: 0.0026032 sec, 25.7794 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.000286508 sec, 234.231GB/s, 0.0671089 GB

run_add() cpu latency: 0.0454628 sec
run_add(): type=fp32, N: 33554432, 0.268435 GB, latency: 0.00368755 sec, 72.7951 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 33554432, 0.134218 GB, latency: 0.00518212 sec, 25.9001 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.000558591 sec, 240.279GB/s, 0.134218 GB

run_add() cpu latency: 0.0648241 sec
run_add(): type=fp32, N: 50331648, 0.402653 GB, latency: 0.00552037 sec, 72.9396 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 50331648, 0.201327 GB, latency: 0.00777392 sec, 25.8977 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.000833702 sec, 241.485GB/s, 0.201327 GB
```



