"""
Dimension 1: Token & Size Compression Efficiency Benchmark Script
------------------------------------------------------------------
This script tests the performance of the Semantic Image Compression Framework
along Dimension 1 (Token & Payload Size Compression Efficiency):

  - Estimated Visual Tokens (VLM input cost for raw image)
  - XML Text Tokens (downstream LLM cost for compressed XML)
  - Token Compression Ratio (Visual Tokens / Text Tokens)
  - Token Savings Percentage (%)
  - Raw Image File Size (KB) vs XML Output Size (KB)
  - Byte Compression Ratio & Byte Savings Percentage (%)
  - Pipeline Execution Latency (seconds)

This script does NOT modify any existing source code in the repository.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

from PIL import Image

# Try importing tiktoken for accurate LLM token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def estimate_gpt4o_visual_tokens(
    image_width: int, image_height: int, low_res: bool = False
) -> int:
    """
    Estimates visual tokens for OpenAI GPT-4o / GPT-4-Vision.
    Low-res mode: 85 tokens.
    High-res mode:
      1. Scale image to fit within 2048x2048.
      2. Scale shortest side to 768px.
      3. Count 512x512 tiles: (170 tokens * num_tiles) + 85 base tokens.
    """
    if low_res:
        return 85

    w, h = image_width, image_height

    # Step 1: Scale down to fit within 2048x2048
    max_dim = max(w, h)
    if max_dim > 2048:
        scale = 2048.0 / max_dim
        w, h = int(w * scale), int(h * scale)

    # Step 2: Scale such that shortest side is 768px
    min_dim = min(w, h)
    if min_dim > 768:
        scale = 768.0 / min_dim
        w, h = int(w * scale), int(h * scale)

    # Step 3: Count 512x512 tiles
    tiles_w = math.ceil(w / 512.0)
    tiles_h = math.ceil(h / 512.0)
    total_tiles = tiles_w * tiles_h

    return (total_tiles * 170) + 85


def estimate_claude_visual_tokens(image_width: int, image_height: int) -> int:
    """
    Estimates visual tokens for Anthropic Claude 3 / 3.5 Sonnet.
    Formula approximately: (width * height) / 750 tokens, max 1600 tokens.
    """
    tokens = math.ceil((image_width * image_height) / 750.0)
    return max(80, min(1600, tokens))


def estimate_llava_visual_tokens(image_width: int, image_height: int) -> int:
    """
    Estimates visual tokens for open-source patch-based models (e.g. LLaVA 1.5/1.6, Qwen2-VL).
    Standard patch grid: 576 visual tokens per 336x336 image tile.
    """
    tiles_w = math.ceil(image_width / 336.0)
    tiles_h = math.ceil(image_height / 336.0)
    return tiles_w * tiles_h * 576


def estimate_visual_tokens(img_path: str, vlm_model: str = "gpt4o") -> int:
    """Returns estimated visual tokens based on target VLM model architecture."""
    with Image.open(img_path) as img:
        w, h = img.size

    model_key = vlm_model.lower().replace("-", "").replace("_", "")
    if "gpt4olow" in model_key or "lowres" in model_key:
        return estimate_gpt4o_visual_tokens(w, h, low_res=True)
    elif "claude" in model_key:
        return estimate_claude_visual_tokens(w, h)
    elif "llava" in model_key or "qwen" in model_key or "patch" in model_key:
        return estimate_llava_visual_tokens(w, h)
    else:  # Default to GPT-4o High-Res
        return estimate_gpt4o_visual_tokens(w, h, low_res=False)


def count_text_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Counts LLM text tokens using tiktoken, or falls back to ~4 characters per token estimate."""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            print(e)

    # Fallback heuristic: ~4 characters per token
    return max(1, math.ceil(len(text) / 4.0))


class Dimension1Benchmarker:
    def __init__(
        self, vlm_model: str = "gpt4o", tokenizer_encoding: str = "cl100k_base"
    ):
        self.vlm_model = vlm_model
        self.tokenizer_encoding = tokenizer_encoding

        print("Initializing Semantic Image Pipeline...")
        from run_pipeline import PipelineManager

        self.pipeline_manager = PipelineManager()
        print("Pipeline initialized successfully.\n")

    def run_benchmark_on_image(self, image_path: str) -> dict:
        """Executes compression pipeline on a single image and computes Dimension 1 metrics."""
        image_path = str(Path(image_path).resolve())

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get raw image metadata
        raw_size_bytes = os.path.getsize(image_path)
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Estimate Visual Tokens
        est_visual_tokens = estimate_visual_tokens(image_path, vlm_model=self.vlm_model)

        # Run pipeline and measure latency
        start_time = time.perf_counter()
        xml_output = self.pipeline_manager.run(image_path)
        latency_sec = time.perf_counter() - start_time

        # Measure XML metadata
        xml_size_bytes = len(xml_output.encode("utf-8"))
        xml_tokens = count_text_tokens(
            xml_output, encoding_name=self.tokenizer_encoding
        )

        # Dimension 1 Ratios & Savings
        token_cr = est_visual_tokens / xml_tokens if xml_tokens > 0 else 0.0
        token_savings_pct = (
            (1.0 - (xml_tokens / est_visual_tokens)) * 100.0
            if est_visual_tokens > 0
            else 0.0
        )

        byte_cr = raw_size_bytes / xml_size_bytes if xml_size_bytes > 0 else 0.0
        byte_savings_pct = (
            (1.0 - (xml_size_bytes / raw_size_bytes)) * 100.0
            if raw_size_bytes > 0
            else 0.0
        )

        return {
            "filename": os.path.basename(image_path),
            "filepath": image_path,
            "resolution": f"{img_width}x{img_height}",
            "raw_size_kb": round(raw_size_bytes / 1024.0, 2),
            "xml_size_kb": round(xml_size_bytes / 1024.0, 2),
            "visual_tokens": est_visual_tokens,
            "xml_text_tokens": xml_tokens,
            "token_compression_ratio": round(token_cr, 2),
            "token_savings_pct": round(token_savings_pct, 2),
            "byte_compression_ratio": round(byte_cr, 2),
            "byte_savings_pct": round(byte_savings_pct, 2),
            "latency_sec": round(latency_sec, 3),
            "xml_output": xml_output,
        }


def print_summary_table(results: list, vlm_model: str):
    """Prints a formatted ASCII summary table for Dimension 1 metrics."""
    print("=" * 105)
    print(
        f"  DIMENSION 1: TOKEN & SIZE COMPRESSION EFFICIENCY BENCHMARK  (VLM Target: {vlm_model.upper()})"
    )
    print("=" * 105)
    header = f"{'Filename':<18} | {'Resolution':<11} | {'Raw(KB)':<8} | {'XML(KB)':<8} | {'VisTokens':<9} | {'XMLTokens':<9} | {'TokCR':<7} | {'TokSave%':<8} | {'Latency':<7}"
    print(header)
    print("-" * 105)

    total_vis_tokens = 0
    total_xml_tokens = 0
    total_raw_kb = 0.0
    total_xml_kb = 0.0
    total_latency = 0.0

    for r in results:
        total_vis_tokens += r["visual_tokens"]
        total_xml_tokens += r["xml_text_tokens"]
        total_raw_kb += r["raw_size_kb"]
        total_xml_kb += r["xml_size_kb"]
        total_latency += r["latency_sec"]

        line = (
            f"{r['filename'][:18]:<18} | "
            f"{r['resolution']:<11} | "
            f"{r['raw_size_kb']:<8.1f} | "
            f"{r['xml_size_kb']:<8.1f} | "
            f"{r['visual_tokens']:<9d} | "
            f"{r['xml_text_tokens']:<9d} | "
            f"{r['token_compression_ratio']:<7.2f} | "
            f"{r['token_savings_pct']:<7.1f}% | "
            f"{r['latency_sec']:<6.2f}s"
        )
        print(line)

    print("-" * 105)
    n = len(results)
    if n > 0:
        avg_tok_cr = (
            total_vis_tokens / total_xml_tokens if total_xml_tokens > 0 else 0.0
        )
        avg_tok_save = (
            (1.0 - (total_xml_tokens / total_vis_tokens)) * 100.0
            if total_vis_tokens > 0
            else 0.0
        )
        avg_byte_cr = total_raw_kb / total_xml_kb if total_xml_kb > 0 else 0.0
        avg_byte_save = (
            (1.0 - (total_xml_kb / total_raw_kb)) * 100.0 if total_raw_kb > 0 else 0.0
        )

        summary_line = (
            f"{'TOTAL / OVERALL':<18} | "
            f"{f'{n} image(s)':<11} | "
            f"{total_raw_kb:<8.1f} | "
            f"{total_xml_kb:<8.1f} | "
            f"{total_vis_tokens:<9d} | "
            f"{total_xml_tokens:<9d} | "
            f"{avg_tok_cr:<7.2f} | "
            f"{avg_tok_save:<7.1f}% | "
            f"{total_latency:<6.2f}s"
        )
        print(summary_line)
        print("=" * 105)
        print(f"Overall Token Compression Ratio: {avg_tok_cr:.2f}x reduction")
        print(
            f"Overall Token Savings:            {avg_tok_save:.2f}% fewer tokens sent to LLM"
        )
        print(
            f"Overall Payload Byte Reduction:   {avg_byte_cr:.2f}x smaller ({avg_byte_save:.2f}% byte savings)"
        )
        print(
            f"Average Processing Time:          {(total_latency / n):.3f} seconds / image"
        )
        print("=" * 105)


def export_results(results: list, output_file: str):
    """Saves benchmark results to JSON or CSV."""
    ext = os.path.splitext(output_file)[1].lower()

    # Exclude raw xml_output string from summary report if exporting unless needed
    clean_results = []
    for r in results:
        item = {k: v for k, v in r.items() if k != "xml_output"}
        clean_results.append(item)

    if ext == ".csv":
        if clean_results:
            keys = clean_results[0].keys()
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(clean_results)
            print(f"\nResults successfully exported to CSV: {output_file}")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults successfully exported to JSON: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Dimension 1: Token & Size Compression Efficiency Benchmark"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="images.jpeg",
        help="Path to an input image file or directory containing images (default: 'images.jpeg')",
    )
    parser.add_argument(
        "-v",
        "--vlm-model",
        default="gpt4o",
        choices=["gpt4o", "gpt4o-low", "claude35", "llava15"],
        help="Target VLM model architecture for visual token baseline estimation (default: 'gpt4o')",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="cl100k_base",
        help="Tiktoken encoding name for XML text token counting (default: 'cl100k_base')",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path to export results (JSON or CSV, e.g. results.json or results.csv)",
    )
    parser.add_argument(
        "--save-xml-dir",
        help="Optional directory path to save generated compressed XML files",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Gather image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = []

    if input_path.is_file():
        image_files.append(input_path)
    elif input_path.is_dir():
        for root, _, files in os.walk(input_path):
            for file in sorted(files):
                if Path(file).suffix.lower() in valid_extensions:
                    image_files.append(Path(root) / file)

    if not image_files:
        print(
            f"Error: No valid image files ({', '.join(valid_extensions)}) found at '{input_path}'."
        )
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to benchmark.")
    if not TIKTOKEN_AVAILABLE:
        print(
            "Note: 'tiktoken' package is not installed. Falling back to character-based token estimation (~4 chars/token)."
        )

    # Initialize Benchmarker
    benchmarker = Dimension1Benchmarker(
        vlm_model=args.vlm_model, tokenizer_encoding=args.tokenizer
    )

    # Save XML setup if requested
    if args.save_xml_dir:
        os.makedirs(args.save_xml_dir, exist_ok=True)

    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Benchmarking: {img_file.name}...")
        try:
            res = benchmarker.run_benchmark_on_image(str(img_file))
            results.append(res)

            if args.save_xml_dir:
                xml_filename = f"{img_file.stem}_compressed.xml"
                xml_path = os.path.join(args.save_xml_dir, xml_filename)
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(res["xml_output"])

        except Exception as e:
            print(f"   [ERROR] Failed to process {img_file.name}: {e}")

    if not results:
        print("No images were successfully processed.")
        sys.exit(1)

    # Output ASCII Table Summary
    print()
    print_summary_table(results, vlm_model=args.vlm_model)

    # Export if requested
    if args.output:
        export_results(results, args.output)


if __name__ == "__main__":
    main()
