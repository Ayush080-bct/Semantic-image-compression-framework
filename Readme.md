# not_text_processor

Vision-side extraction module for the **Semantic Image Compression Framework**. This branch handles everything *except* OCR/text extraction — it's responsible for turning a raw image into structured semantic information using Florence-2.

## What this does

Instead of feeding raw image tokens directly to an LLM, this module extracts structured visual semantics (captions, objects, regions) using Microsoft's Florence-2 vision-language model. The output feeds into the Semantic Mapper, which combines it with OCR output (handled on a separate branch) to produce the final XML document.

## Pipeline position

```
Input Image
    ├── Vision-Language Model (this branch) ──┐
    └── OCR Engine (text_processor branch) ────┼──> Semantic Mapper ──> XML Document ──> LLM
```

## Setup

### Requirements
- Python 3.12
- NVIDIA GPU with CUDA 12.1+ (CPU fallback supported but slow)
- Conda

### Environment

```bash
conda create -n semantic-compression python=3.12
conda activate semantic-compression

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.38.0 Pillow einops timm
pip install flash_attn --no-build-isolation
```

> **Note:** `flash_attn` requires a matching CUDA toolkit (`nvcc`). If not available system-wide, install it into the conda env: `conda install -c nvidia cuda-toolkit=12.1`

## Usage

```python
from florence_extractor import FlorenceExtractor

extractor = FlorenceExtractor()
result = extractor.extract("path/to/image.jpg")

print(result["caption"])  # Detailed scene description
print(result["objects"])  # Bounding boxes + labels
```

## Florence-2 tasks used

| Task | Purpose |
|---|---|
| `<DETAILED_CAPTION>` | Scene-level semantic description |
| `<OD>` | Object detection (bounding boxes + labels) |

### Known limitations
- `<OD>` is tuned for natural photos — performs weakly on diagrams, screenshots, and UI content. Consider `<DENSE_REGION_CAPTION>` or `<REGION_PROPOSAL>` for non-photographic inputs.
- OCR task (`<OCR>`) is intentionally **not** used here — that responsibility belongs to the `text_processor` branch (EasyOCR).

## Output format

```python
{
    "caption": {"<DETAILED_CAPTION>": "..."},
    "objects": {"<OD>": {"bboxes": [[x1, y1, x2, y2], ...], "labels": ["..."]}}
}
```

This dict is passed downstream to the Semantic Mapper for merging with OCR output and conversion into the final XML schema.

## Next steps
- [ ] Evaluate `<DENSE_REGION_CAPTION>` for diagram-heavy inputs
- [ ] Benchmark inference latency (GPU vs CPU)
- [ ] Define interface contract with Semantic Mapper module
