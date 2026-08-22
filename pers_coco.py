import os
import json
import tempfile
import tiktoken
from datasets import load_dataset
from run_pipeline import PipelineManager  # Make sure this matches your actual import

# Where the per-image outputs (image + xml) get saved
OUTPUT_DIR = "benchmark_outputs"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
XML_DIR = os.path.join(OUTPUT_DIR, "xml")


def calculate_image_tokens(width, height, detail="high"):
    """
    Simulates multimodal LLM token consumption for a raw image.
    Uses standard baseline logic (e.g., OpenAI's Vision model token pricing).
    """
    if detail == "low":
        return 85

    # Simplified tile calculation: 85 base tokens + 170 per 512x512 tile
    tiles_w = (width + 511) // 512
    tiles_h = (height + 511) // 512
    return 85 + (170 * tiles_w * tiles_h)


def calculate_text_tokens(text_data, model="gpt-4o"):
    """Calculates token count for the generated XML."""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text_data))


def run_semantic_compression_benchmark(num_images=100):
    """
    Streams COCO dataset online, processing a specific subset of images
    to benchmark XML semantic compression token savings. Saves each
    image and its corresponding XML representation to disk.
    """
    print(f"Streaming first {num_images} images from COCO...")

    # Make sure output folders exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(XML_DIR, exist_ok=True)

    # Load dataset in streaming mode to avoid downloading the massive archive
    dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)

    # Take only the requested number of images
    subset = dataset.take(num_images)

    # Instantiate the pipeline ONCE outside the loop to avoid reloading models
    print("Loading PipelineManager models (Florence/EasyOCR)...")
    pipeline = PipelineManager()

    results = []
    total_raw_tokens = 0
    total_xml_tokens = 0

    for i, sample in enumerate(subset):
        temp_path = None
        try:
            image = sample["image"]
            width, height = image.size

            # Use a stable, human-readable id for filenames
            image_id = sample.get("image_id", i)
            safe_id = str(image_id)

            image_save_path = os.path.join(IMAGES_DIR, f"{safe_id}.jpg")
            xml_save_path = os.path.join(XML_DIR, f"{safe_id}.xml")

            # Save the streamed image directly to its permanent location.
            # (Convert to RGB to ensure smooth JPEG saving.)
            image.convert("RGB").save(image_save_path, format="JPEG")

            # The pipeline still needs a file path to run against — reuse the
            # saved image instead of writing a separate temp copy.
            temp_path = image_save_path

            # 1. Baseline: Calculate raw image tokens
            raw_tokens = calculate_image_tokens(width, height, detail="high")

            # 2. Execution: Run the compression pipeline with the FILE PATH -> XML
            xml_representation = pipeline.run(temp_path)

            # 3. Persist the XML representation to disk
            with open(xml_save_path, "w", encoding="utf-8") as xml_file:
                xml_file.write(xml_representation)

            # 4. Measurement: Calculate XML tokens
            xml_tokens = calculate_text_tokens(xml_representation)

            # 5. Analysis
            savings = raw_tokens - xml_tokens
            reduction_percent = (savings / raw_tokens) * 100 if raw_tokens > 0 else 0

            total_raw_tokens += raw_tokens
            total_xml_tokens += xml_tokens

            results.append({
                "image_id": image_id,
                "image_path": image_save_path,
                "xml_path": xml_save_path,
                "raw_tokens": raw_tokens,
                "xml_tokens": xml_tokens,
                "reduction_percent": reduction_percent,
            })

            print(f"[Image {i+1:03d}] Raw: {raw_tokens:4d} | XML: {xml_tokens:4d} | "
                  f"Reduction: {reduction_percent:.2f}% | Saved: {image_save_path}, {xml_save_path}")

        except Exception as e:
            print(f"[Image {i+1:03d}] Failed to process. Error: {e}")

        # No finally-block cleanup anymore — we intentionally KEEP the image
        # file on disk instead of deleting it, since it's now a saved output.

    # Final Aggregation
    overall_reduction = ((total_raw_tokens - total_xml_tokens) / total_raw_tokens) * 100 if total_raw_tokens > 0 else 0

    print("-" * 50)
    print("BENCHMARK COMPLETE")
    print(f"Total Raw Tokens: {total_raw_tokens}")
    print(f"Total XML Tokens: {total_xml_tokens}")
    print(f"Overall Token Reduction: {overall_reduction:.2f}%")
    print(f"Images saved to: {os.path.abspath(IMAGES_DIR)}")
    print(f"XML saved to:    {os.path.abspath(XML_DIR)}")

    # Save a manifest summarizing all results + file locations
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({
            "num_images": num_images,
            "total_raw_tokens": total_raw_tokens,
            "total_xml_tokens": total_xml_tokens,
            "overall_reduction_percent": overall_reduction,
            "results": results,
        }, manifest_file, indent=2)
    print(f"Manifest written to: {os.path.abspath(manifest_path)}")

    return results


if __name__ == "__main__":
    run_semantic_compression_benchmark(num_images=10)