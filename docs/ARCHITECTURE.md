# System Architecture: Semantic Image Compression Framework

## Overview

The **Semantic Image Compression Framework for LLM-Based Visual Reasoning** is designed to transform high-resolution document images into compact, semantically structured representations while preserving their complete informational content.

The framework eliminates redundant visual information, extracts textual and structural semantics, and compiles the result into a lightweight XML-based representation. This significantly reduces Large Language Model (LLM) context consumption and inference costs without sacrificing semantic fidelity.

---

## End-to-End Pipeline Architecture

Data flows through four sequential processing layers:

```text
[ Tier 1: Data Input Layer ]
                │
                ▼
      Raw Images / File Paths

[ Tier 2: Preprocessing Engine ]
                │
                ▼
   Normalized Image Matrices

[ Tier 3: Core Inference Layer ]
        ├── PaddleOCR
        │     └── Text Extraction & Bounding Boxes
        │
        └── Florence-2
              └── Layout Understanding & Reading Order
                │
                ▼
      Structured Semantic Metadata

[ Tier 4: Semantic Compilation Layer ]
                │
                ▼

      Compressed XML Representation
                │
                ▼

        Downstream LLM Pipeline
```

---

## 1. Data Input Layer

### Purpose

The Data Input Layer serves as the entry point to the framework, providing a reliable and platform-independent mechanism for document acquisition.

### Implementation

To ensure portability across operating systems, file locations are resolved dynamically using Python's `pathlib` library rather than hardcoded absolute paths.

### Data Flow

**Input**

* Relative document references (e.g., `datasets/sample.jpg`)

**Processing**

* Dynamic path resolution
* Cross-platform filesystem abstraction

**Output**

* Validated absolute file path objects

---

## 2. Preprocessing Engine

### Purpose

The Preprocessing Engine prepares raw document images for efficient inference by reducing visual complexity and computational overhead.

### Implementation

Using OpenCV, the engine performs lightweight image normalization operations to improve OCR accuracy and reduce memory consumption.

### Processing Steps

* Color-to-grayscale conversion
* Contrast enhancement
* Resolution normalization
* Image downscaling using `cv2.INTER_AREA`

### Data Flow

**Input**

* Raw image pixel arrays

**Output**

* Optimized NumPy image matrices suitable for inference

---

## 3. Core Inference Layer

### Purpose

This layer performs parallel semantic extraction using specialized vision models.

The framework separates document understanding into two complementary tasks:

1. Fine-grained text recognition
2. Global layout interpretation

### Route A: PaddleOCR

#### Responsibility

Extract textual content and spatial localization information.

#### Output Example

```json
[
  {
    "text": "Total:",
    "bbox": [102, 540, 160, 555]
  }
]
```

#### Generated Information

* Recognized text tokens
* Word-level coordinates
* Line segmentation metadata

---

### Route B: Florence-2

#### Responsibility

Analyze the document holistically to infer structural organization and reading order.

#### Generated Information

* Headers
* Paragraphs
* Tables
* Lists
* Section boundaries
* Reading-order relationships

### Combined Output

Both inference routes generate structured semantic metadata that is asynchronously merged before compilation.

---

## 4. Semantic Compilation Layer

### Purpose

The Semantic Compilation Layer performs the framework's primary compression operation by converting OCR and layout metadata into a compact semantic representation.

### Implementation

A custom compiler integrates textual content with structural annotations while eliminating visually redundant information.

### Compression Operations

* Removal of raw pixel dependencies
* Elimination of whitespace regions and margins
* Reduction of coordinate-heavy metadata
* Conversion of geometric layouts into logical document structures

### Data Flow

**Input**

* OCR text streams
* Structural layout annotations
* Reading-order metadata

**Output**

```xml
<header>Report</header>
<table>
    <row>
        <cell>Total:</cell>
    </row>
</table>
```

---

## Final Output

The framework produces a highly compressed XML document that preserves the original document's semantic meaning while dramatically reducing token footprint.

### Key Benefits

* Significant LLM context-window reduction
* Lower inference cost
* Faster downstream processing
* Preservation of document semantics
* Model-agnostic output format
* Improved scalability for large document collections

```

The resulting XML payload can be directly consumed by downstream LLMs for reasoning, retrieval, summarization, or information extraction tasks.
```
