# VLM Multi-Token Prediction

A PyTorch-based Vision-Language Model (VLM) implementation for Image Q&A with support for multi-token prediction. This project combines OpenCLIP vision encoders with transformer-based language modeling to enable efficient image understanding and text generation.

## Overview

This project implements a **BasicVLM** model that:
- 🖼️ Encodes images using OpenCLIP (ViT-B-32)
- 🔗 Projects image features to language embedding space
- 📝 Generates text through next-token prediction
- 📊 Supports TensorBoard visualization with image predictions
- ⚡ Optimized for GPU training (12GB+ VRAM recommended)

## Features

✅ **Core Architecture**
- OpenCLIP vision encoder (frozen or fine-tuned)
- Linear image projector with dimension alignment
- Transformer decoder with 24 layers, 8 attention heads
- Causal masking for autoregressive generation
- Sinusoidal positional embeddings

✅ **Training Infrastructure**
- Gradient clipping and learning rate scheduling (Cosine Annealing)
- Comprehensive gradient monitoring and statistics
- Automatic checkpoint saving (best model + epoch-wise)

✅ **Logging & Visualization**
- TensorBoard integration for training metrics
- Per-layer gradient and weight histograms
- Ground truth vs. predicted token comparison with color-coded accuracy
- Position-wise analysis for debugging predictions

✅ **Data Handling**
- COCO Caption dataset integration via LAVIS
- Configurable batch size and sequence length
- Proper next-token prediction target construction
- Data validation for input/target alignment

## Project Structure

```
vlm-multi_token_prediction/
├── models/                          # Core model components
│   ├── vlm.py                      # BasicVLM model definition
│   ├── vision_encoder.py           # OpenCLIP encoder wrapper
│   ├── image_proj.py               # Image-to-text projection
│   ├── transformer.py              # Transformer blocks (attention, FFN)
│   ├── positional_embeddings.py    # Sinusoidal PE implementation
│   ├── lm_head.py                  # Language modeling head
│   ├── open_clipencoder.py         # OpenCLIP wrapper
│   └── CLIP.py                     # CLIP model utilities
├── train.py                         # Main trainer class with TensorBoard logging
├── dataloader.py                    # Data loading and preprocessing
├── config/                          # Configuration files
├── utils/                           # Utility functions
├── docs/                            # Documentation
└── README.md                        # This file
```

## Installation

### 1. Environment Setup with `uv`

```bash
# Install uv package manager (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc for macOS
uv --version
```

### 2. Clone Repository & Install Dependencies

```bash
git clone <repository-url>
cd vlm-multi_token_prediction

# Install dependencies using uv
```

### 3. Download COCO Dataset (Optional)

```bash
# Create cache directory
mkdir -p ~/.cache/lavis/coco

# Download COCO dataset and check by using below command
python -c "from lavis.datasets.builders import load_dataset; \
           dataset = load_dataset('coco_caption')"
```

### Data Loading Issues
```bash
# Set LAVIS cache directory
export LAVIS_CACHE_DIR=/path/to/cache
```

## Future Work
- [ ] fix minor bugs
- [ ] Multi-token  prediction with speculative decoding
- [ ] Mixed precision training (FP16/BF16)
- [ ] CLIP style pretraining (not part of this repo)
- [ ] LLM style pretraining (not part of this repo)
