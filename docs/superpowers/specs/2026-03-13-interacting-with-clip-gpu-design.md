# Interacting_with_CLIP GPU Notebook Design

## Summary
Update `tutorials/Interacting_with_CLIP.ipynb` to require a CUDA GPU and run all model/tensor operations on `cuda:0`, while keeping full precision. The notebook will error clearly if CUDA is unavailable. Existing `CUDA_VISIBLE_DEVICES="0"` behavior remains unchanged.

## Goals
- Require CUDA availability and fail fast with a clear error if not present.
- Run model inference and tensor computations on `cuda:0`.
- Keep full precision (no autocast or mixed precision).
- Preserve notebook flow and outputs with minimal edits.

## Non-Goals
- Changing model architecture, preprocessing, or tutorial narrative.
- Adding performance optimizations beyond GPU placement.
- Supporting multiple GPUs or device selection UI.

## Current State
- The notebook sets `CUDA_VISIBLE_DEVICES="0"` but does not move the model or tensors to GPU.
- Model inference runs on CPU by default, and tensors are created on CPU.

## Proposed Design

### Device Enforcement
- Add a device setup cell after `import torch`:
  - Assert `torch.cuda.is_available()` and raise a `RuntimeError` with a clear message if false.
  - Define `device = torch.device("cuda:0")`.
- Keep the existing `CUDA_VISIBLE_DEVICES="0"` cell unchanged.
- Move the model to GPU immediately after creation, before `model.eval()`.

### Tensor Placement
- Ensure all model inputs are on GPU:
  - `image_input` created on GPU (or `.to(device)` after creation).
  - `text_tokens` moved to GPU after tokenization.
- All `encode_image` and `encode_text` calls run on GPU.

### CPU Interop for NumPy/Plotting
- Move tensors to CPU before NumPy and matplotlib usage:
  - `similarity` computed from `.cpu().numpy()` inputs.
  - `top_probs`, `top_labels` moved to CPU before plotting and label indexing.

### CIFAR100 Flow
- Keep dataset loading on CPU (unchanged).
- Compute text features on GPU and move to CPU as needed for plotting.

## Error Handling
- If CUDA is unavailable, raise a `RuntimeError` with guidance to install CUDA/PyTorch GPU support.

## Testing
- Manual notebook run on a CUDA-enabled environment to confirm:
  - It fails fast on non-CUDA environments.
  - GPU tensors/model are used without device mismatch errors.
  - Output plots and results match expected behavior.
