# 🧠 Dream Diffusion — GGUF Inference UI

> **Fork of [DreamLM/Dream](https://github.com/DreamLM/Dream)**  
> The original repo targets BF16 full-precision weights (~20 GB VRAM). This fork replaces the Python inference stack with [`llama-diffusion-cli`](https://github.com/ggml-org/llama.cpp) to load **4-bit GGUF-quantized** weights, bringing the memory requirement down to **~5–8 GB VRAM**.

---

## Differences from upstream

| | [DreamLM/Dream](https://github.com/DreamLM/Dream) (original) | This fork |
|---|---|---|
| **Weights format** | BF16 HuggingFace checkpoint | GGUF (quantized, e.g. Q4_K_M) |
| **VRAM required** | 20 GB+ | ~5–8 GB |
| **Stop / Retry** | ❌ | ✅ |

---

## ✨ Features

- 🔁 **Real-time diffusion visualization** — watch tokens get revealed step-by-step
- 🧩 **System prompt** — set a persistent persona for the model
- ⏹ **Stop button** — cancel mid-generation instantly
- 🔁 **Retry button** — regenerate the last response without retyping
- 📊 **Stats panel** — elapsed time, step count, word count per response
- 🎨 **Dark UI** — custom dark-themed Gradio interface
- 🖥️ **Auto GPU detection** — header shows your actual GPU name

---

## 🖥️ Requirements

| Component | Requirement |
|---|---|
| OS | Windows 10/11 (Linux should work with path changes) |
| GPU | NVIDIA GPU with CUDA support (8 GB VRAM recommended) |
| CUDA | 12.4 (`cu124`) |
| Python | 3.11 |
| Conda | Anaconda / Miniconda |

### External dependencies (not in pip)

- **`llama-diffusion-cli.exe`** — Build from the `diffusion` branch of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **GGUF model file** — Download from HuggingFace:  
  [`bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF`](https://huggingface.co/bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF)  
  Recommended quant: `Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf`

---

## 🚀 Installation

### 1. Clone this repo

```bash
git clone https://github.com/your-username/Dream-main.git
cd Dream-main
```

### 2. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate dream_diff_env
```

> **Tip:** If PyTorch isn't picking up CUDA, reinstall with the correct index:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

### 3. Build `llama-diffusion-cli`

This is the C++ binary that actually runs inference. It is **not on PyPI** — you must compile it from the `llama.cpp` source.

#### Prerequisites

| Tool | Notes |
|---|---|
| [Git](https://git-scm.com/) | For cloning the repo |
| [Visual Studio 2022](https://visualstudio.microsoft.com/vs/community/) | Select **"Desktop development with C++"** workload; make sure **"C++ CMake tools for Windows"** is included |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | Match your driver version; CUDA 12.x recommended |
| CMake 3.21+ | Included with VS2022, or install from [cmake.org](https://cmake.org/download/) |

#### Build steps (PowerShell / Developer Command Prompt)

```powershell
# 1. Clone llama.cpp (main branch — diffusion support is merged in)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 2. Configure with CMake — enable CUDA
cmake -B build -S . `
    -DGGML_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release

# 3. Build (adjust -j to your CPU core count)
cmake --build build --config Release -j 4

# 4. The binary will be at:
#    build\bin\Release\llama-diffusion-cli.exe
```

> **Tip:** If CMake can't find CUDA, open a **Developer Command Prompt for VS 2022** (Start menu) instead of a plain PowerShell — it sets the required MSVC and CUDA environment variables automatically.

#### Verify the build

```powershell
.\build\bin\Release\llama-diffusion-cli.exe --help
```

You should see usage output listing `--diffusion-steps`, `--diffusion-algorithm`, etc.

---

### 4. Edit paths in `app_gguf.py`

```python
LLAMA_DIFFUSION_CLI_PATH = r"C:\path\to\llama-diffusion-cli.exe"
MODEL_PATH               = r"C:\path\to\Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf"
```

### 4. Run

```bash
python app_gguf.py
# → http://localhost:7860
```

---

## ⚙️ Generation Parameters

| Parameter | Description | Recommended |
|---|---|---|
| Max New Tokens | Maximum tokens to generate | 256–512 |
| Diffusion Steps | More steps → higher quality, slower | 16–48 |
| Temperature | Randomness (0 = greedy) | 0.05–0.3 |
| Top-p | Nucleus sampling cutoff | 0.95 |
| Top-k | Top-k sampling (0 = auto) | 0 |
| Diffusion Algorithm | Token selection strategy | `low_confidence` |
| Algorithm Temperature | Mask selection aggressiveness | 0.1 |

---

## 🗂️ Project Structure

```
Dream-main/
├── app_gguf.py               # Main Gradio application (this fork)
├── environment.yaml     # Conda environment spec
├── README.md            # This file
├── debug.py             # Standalone debug/test script
├── demo_*.py            # Original upstream demo scripts
├── src/                 # Source utilities (upstream)
└── eval/                # Evaluation scripts (upstream)
```

---

## 📄 License

See [LICENSE](LICENSE). Original model and code by [DreamLM](https://github.com/DreamLM/Dream).
