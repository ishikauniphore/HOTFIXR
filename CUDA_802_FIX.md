# Fix: CUDA Error 802 (cudaErrorSystemNotReady) on 8× A100 SXM4

## Symptom

Any attempt to initialize CUDA — whether in Python via `torch.cuda.init()`, `torch.cuda.is_available()`, or through vllm's `LLM(...)` constructor — failed with:

```
RuntimeError: CUDA error: system not ready
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

Or at the C level: `cuInit(0)` returned error code **802** (`CUDA_ERROR_SYSTEM_NOT_READY`).

This happened even in the main thread (not just in FastAPI threadpool workers), and even after enabling NVIDIA persistence mode (`sudo nvidia-smi -pm 1`).

---

## Root Cause

The machine has **8× NVIDIA A100 SXM4 GPUs** connected via **NVSwitch** (NVLink fabric). On NVSwitch-based systems, CUDA **cannot initialize on any GPU** unless the **NVIDIA Fabric Manager** service is running.

The Fabric Manager is responsible for:
- Discovering and configuring all GPUs and NVSwitch chips
- Establishing the NVLink routing topology between GPUs

Without it, `cuInit()` returns 802 regardless of persistence mode or any Python-level workarounds.

The service was simply not installed on this machine.

---

## Fix

Install and start the NVIDIA Fabric Manager matching the installed driver version (580.x):

```bash
sudo apt-get install -y nvidia-fabricmanager-580
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
```

Verify it's running and configured the NVSwitch fabric:

```bash
systemctl status nvidia-fabricmanager
```

Expected output includes:
```
Active: active (running)
...
nv-fabricmanager: Successfully configured all the available GPUs and NVSwitches to route NVLink traffic.
```

Then confirm CUDA works:

```bash
conda run -n verl python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
# Expected: True, 8
```

---

## Notes

- **A reboot is NOT required** if the NVIDIA driver is already loaded. The Fabric Manager can configure the NVSwitch topology on a live system.
- **Persistence mode** (`nvidia-persistenced`) is a separate service from Fabric Manager and alone is not sufficient for NVSwitch systems.
- The versioned package (`nvidia-fabricmanager-580`) must match the installed driver version exactly. Check your driver version with `nvidia-smi`.
- To find the right package name: `apt-cache search nvidia-fabricmanager`
- This fix applies to any NVSwitch-based NVIDIA system (A100 SXM4, H100 SXM, etc.) — not just AWS EC2.
- AWS officially documents this requirement for P4d/P4de (A100) and P5 (H100) instance types in their EC2 NVIDIA driver install guide.

---

## Why Persistence Mode Alone Wasn't Enough

| Service | Purpose | Required for NVSwitch? |
|---|---|---|
| `nvidia-persistenced` | Keeps GPU driver state loaded between processes | No (but recommended) |
| `nvidia-fabricmanager` | Configures NVSwitch routing so GPUs can communicate | **Yes — mandatory** |

On single-GPU or PCIe multi-GPU systems (no NVSwitch), Fabric Manager is not needed. On SXM form-factor systems with NVSwitch (A100 SXM4, H100 SXM5, etc.), it is required for CUDA to initialize at all.
