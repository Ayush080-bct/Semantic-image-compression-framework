import os
import tempfile
import tiktoken
from datasets import load_dataset
from run_pipeline import PipelineManager  # Make sure this matches your actual import

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
    to benchmark XML semantic compression token savings.
    """
    print(f"Streaming first {num_images} images from COCO...")
    
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
            
            # Save the streamed image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
                # Convert to RGB to ensure smooth JPEG saving
                image.convert('RGB').save(temp_path, format="JPEG")
            
            # 1. Baseline: Calculate raw image tokens
            raw_tokens = calculate_image_tokens(width, height, detail="high")
            
            # 2. Execution: Run the compression pipeline with the FILE PATH -> XML
            xml_representation = pipeline.run(temp_path) 
            
            # 3. Measurement: Calculate XML tokens
            xml_tokens = calculate_text_tokens(xml_representation)
            
            # 4. Analysis
            savings = raw_tokens - xml_tokens
            reduction_percent = (savings / raw_tokens) * 100 if raw_tokens > 0 else 0
            
            total_raw_tokens += raw_tokens
            total_xml_tokens += xml_tokens
            
            results.append({
                "image_id": sample.get("image_id", i),
                "raw_tokens": raw_tokens,
                "xml_tokens": xml_tokens,
                "reduction_percent": reduction_percent
            })
            
            print(f"[Image {i+1:03d}] Raw: {raw_tokens:4d} | XML: {xml_tokens:4d} | Reduction: {reduction_percent:.2f}%")
            
        except Exception as e:
            print(f"[Image {i+1:03d}] Failed to process. Error: {e}")
            
        finally:
            # Clean up the temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        
    # Final Aggregation
    overall_reduction = ((total_raw_tokens - total_xml_tokens) / total_raw_tokens) * 100 if total_raw_tokens > 0 else 0
    
    print("-" * 50)
    print("BENCHMARK COMPLETE")
    print(f"Total Raw Tokens: {total_raw_tokens}")
    print(f"Total XML Tokens: {total_xml_tokens}")
    print(f"Overall Token Reduction: {overall_reduction:.2f}%")
    
    return results

if __name__ == "__main__":
    run_semantic_compression_benchmark(num_images=100)