import torch
import time
from statistics import mean, stdev
import extension_cpp

import fire

# ================================================================
# 1. Setup
# ================================================================
device = "cuda"
dtype = torch.float32

def sample_inputs(device, b, l, n, k, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    def make_nondiff_tensor(*size):
        return torch.randn(size, device=device, requires_grad=False)

    def make_zero_tensor(*size):
        return torch.zeros(size, device=device, requires_grad=requires_grad)

    return make_tensor(b, l, n), make_tensor(1, 1, k), make_tensor(b, 1, n)


# ================================================================
# 2. Define your operators
# ================================================================

# ---- Reference PyTorch implementation ----
def reference_op(*args):
    # print(len(args[0]))
    return extension_cpp.ops.reference_convrnn(*args[0])


# ---- Custom CUDA operator wrapper ----
# For example if your extension is: custom_ops.my_conv1d
def custom_op(*args):
    return extension_cpp.ops.convrnn_interface(*args[0])


# ================================================================
# 3. Benchmark helpers
# ================================================================

def time_gpu(fn, *args, repeat=20, do_backward=False):
    """Return list of execution times in milliseconds."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        out = fn(*args)
        if do_backward:
            g = torch.randn_like(out)
            out.backward(g, retain_graph=True)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        out = fn(*args)
        if do_backward:
            g = torch.randn_like(out)
            out.backward(g, retain_graph=True)

        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))  # ms

    return times


def report(name, times):
    print(f"{name:20s}:  mean {mean(times):.3f} ms   std {stdev(times):.3f} ms")




def run_benchmark(batch_size=32, seq_length=128, kernel_size=3, n=16):
    # ================================================================
    # 4. Run benchmarks
    # ================================================================
    print("\n--- Inference (forward only) ---")
    ref_fwd_times = time_gpu(reference_op,
                             sample_inputs("cuda",
                                                batch_size, seq_length, n, kernel_size,
                                                requires_grad=True),
                             do_backward=False)
    custom_fwd_times = time_gpu(custom_op,
                                sample_inputs("cuda",
                                                    batch_size, seq_length, n, kernel_size,
                                                    requires_grad=True),
                                do_backward=False)

    report("Reference forward", ref_fwd_times)
    report("Custom forward", custom_fwd_times)

    print("\n--- Training (forward + backward) ---")
    ref_train_times = time_gpu(reference_op,
                               sample_inputs("cuda",
                                                    batch_size, seq_length, n, kernel_size,
                                                    requires_grad=True),
                               do_backward=True)
    custom_train_times = time_gpu(custom_op,
                                  sample_inputs("cuda",
                                                      batch_size, seq_length, n, kernel_size,
                                                      requires_grad=True),
                                  do_backward=True)

    report("Reference train", ref_train_times)
    report("Custom train", custom_train_times)


    # ================================================================
    # 5. Compute speedups
    # ================================================================
    def speedup(ref, custom):
        return mean(ref) / mean(custom)


    print("\n--- Speedup ---")
    print(f"Forward speedup: {speedup(ref_fwd_times, custom_fwd_times):.2f}x")
    print(f"Train speedup:   {speedup(ref_train_times, custom_train_times):.2f}x")

if __name__ == "__main__":
    fire.Fire(run_benchmark)
