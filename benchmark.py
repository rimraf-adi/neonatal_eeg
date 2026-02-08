import time
import torch
import torch.nn as nn
import torch.optim as optim

def run_benchmark(device, batch_size=64, iters=200):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1000),
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # synthetic data
    x = torch.randn(batch_size, 4096, device=device)
    y = torch.randn(batch_size, 1000, device=device)

    # warm-up
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if device != "cpu":
            torch.mps.synchronize()

    # timed section
    start = time.time()
    for _ in range(iters):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if device != "cpu":
            torch.mps.synchronize()
    end = time.time()

    return end - start


# RUN TESTS
cpu_t = run_benchmark("cpu")
mps_t = None

if torch.backends.mps.is_available():
    mps_t = run_benchmark("mps")

print("CPU time:", cpu_t)
print("MPS time:", mps_t)
if mps_t is not None:
    print("Speedup:", cpu_t / mps_t)
