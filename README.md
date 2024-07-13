# cuda_bf16_benchmarks


how to run:

```
make
./main
```

Here is the report it prints on an NVIDIA A10 GPU:
```
run_add() cpu latency: 0.0229251 sec
run_add(): type=fp32, N: 16777216, 0.134218 GB, latency: 0.00185399 sec, 72.394 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 16777216, 0.0671089 GB, latency: 0.00259881 sec, 25.8229 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.00111141 sec, 241.526GB/s, 0.268435 GB

run_add() cpu latency: 0.0462849 sec
run_add(): type=fp32, N: 33554432, 0.268435 GB, latency: 0.00368159 sec, 72.913 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 33554432, 0.134218 GB, latency: 0.00518906 sec, 25.8655 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.0022037 sec, 243.622GB/s, 0.536871 GB

run_add() cpu latency: 0.065717 sec
run_add(): type=fp32, N: 50331648, 0.402653 GB, latency: 0.00551865 sec, 72.9622 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
run_add(): type=bf16, N: 50331648, 0.201327 GB, latency: 0.00777798 sec, 25.8842 GB/s, 
    diff_arrays(), diff: 0 avg_diff: 0
memcpy_ubench(): 0.00330279 sec, 243.826GB/s, 0.805306 GB
```



