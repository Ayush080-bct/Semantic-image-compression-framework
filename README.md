# Semantic Image Compression Framework For LLM-Based Visual Reasoning

## Overview

The Semantic Image Compression Framework is a research-oriented system designed to reduce the computational and context-window costs associated with Large Language Model (LLM) visual reasoning.

Instead of sending high-resolution document images directly to multimodal models, the framework extracts semantic information such as text, document hierarchy, layout structure, tables, and sections, then compiles them into a lightweight XML/Text representation.

This approach significantly reduces token consumption while preserving semantic fidelity.

---

## Objectives

* Extract textual and structural information from document images.
* Remove redundant visual information that does not contribute to reasoning.
* Convert document layouts into machine-readable semantic representations.
* Reduce LLM context window utilization.
* Improve inference speed and lower operational costs.

---
## Architecture

A detailed description of the framework architecture, processing layers, data flow, and semantic compilation pipeline is available in:

```text
docs/ARCHITECTURE.md
```

The framework follows a four-tier architecture:

```text
Data Input Layer
        ↓
Preprocessing Engine
        ↓
Core Inference Layer
(PaddleOCR + Florence-2)
        ↓
Semantic Compilation Layer
        ↓
Compressed XML Representation
        ↓
Downstream LLM Pipeline
```

For implementation details, component responsibilities, data journeys, and compression strategies, refer to `docs/ARCHITECTURE.md`.

---

## Project Structure

```text
Semantic-image-compression-framework/
│
├── datasets/
│   └── sample.jpg
│
├── docs/
│   └── ARCHITECTURE.md
│
├── Notebooks/
│   └── prototype.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── extractors.py
│   └── compiler.py
│
├── .gitignore
├── environment.yml
└── README.md
```

---

## Environment Setup

### Create Environment

```bash
conda create -n minpro python=3.12 -y
```

### Activate Environment

```bash
conda activate minpro
```

### Export Environment

```bash
conda env export --no-builds | grep -v "^prefix: " > environment.yml
```

### Synchronize Environment

```bash
conda env update -n minpro --file environment.yml --prune
```

## Streamlit Dashboard

Run the report dashboard from the repository root:

```bash
streamlit run Dashboard/streamlit_app.py
```

Upload an image to inspect the route, OCR boxes, Florence caption, generated XML, processing time, and representation-size measurements. XML byte and token values are proxies for semantic representation size; they do not by themselves establish semantic fidelity or downstream LLM cost savings.

---

## Core Modules

### preprocessing.py

Performs image enhancement operations:

* Grayscale conversion
* Noise removal
* Thresholding
* Contrast enhancement
* Deskewing

### extractors.py

Responsible for:

* OCR extraction using PaddleOCR
* Layout analysis
* Bounding box generation
* Semantic region detection

### compiler.py

Converts extracted information into structured semantic formats such as:

```xml
<document>
    <title>Research Paper</title>

    <section>
        <heading>Introduction</heading>
        <paragraph>
            Content...
        </paragraph>
    </section>
</document>
```

---

## Example Workflow

1. Load document image.
2. Apply preprocessing operations.
3. Extract text and layout metadata.
4. Reconstruct semantic document hierarchy.
5. Generate compressed XML representation.
6. Send compressed output to downstream LLM.

---

## Expected Benefits

| Metric            | Traditional Vision Input | Semantic Compression |
| ----------------- | ------------------------ | -------------------- |
| Input Size        | High                     | Low                  |
| Token Usage       | High                     | Reduced              |
| Processing Cost   | High                     | Lower                |
| Latency           | Higher                   | Faster               |
| Semantic Fidelity | High                     | High                 |

---

## Future Enhancements

* Table structure reconstruction
* Mathematical formula extraction
* Multi-page PDF support
* JSON schema export
* Token reduction benchmarking
* Evaluation metrics for semantic preservation
* Integration with multimodal LLM pipelines

---

## Research Motivation

Modern multimodal language models often consume significant computational resources when processing document images. Much of the visual information contained in documents is redundant for reasoning tasks.

This framework investigates whether document images can be transformed into compact semantic representations that preserve meaning while dramatically reducing context-window requirements and inference costs.

---
