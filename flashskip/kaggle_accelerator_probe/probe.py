import json, time, torch

assert torch.cuda.is_available()
device = torch.device("cuda")
info = {
    "name": torch.cuda.get_device_name(0),
    "count": torch.cuda.device_count(),
    "capability": torch.cuda.get_device_capability(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
for _ in range(10):
    x @ x
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    x @ x
torch.cuda.synchronize()
info["matmul_iterations_per_second"] = 100 / (time.perf_counter() - start)
print("ACCELERATOR_PROBE", json.dumps(info, sort_keys=True), flush=True)
