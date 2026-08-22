from PIL import Image, ImageDraw
from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor

def visualize_dense_regions(image_path, dense_result, output_path="dense_visualization.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    regions = dense_result.get("<DENSE_REGION_CAPTION>", {})
    for bbox, label in zip(regions.get("bboxes", []), regions.get("labels", [])):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        draw.text((x1, y1 - 12), label[:30], fill="blue")
    img.save(output_path)
    print(f"Saved: {output_path}")

florence = FlorenceExtractor()
image = Image.open("images.jpeg").convert("RGB")
dense = florence.run_task(image, "<DENSE_REGION_CAPTION>")
visualize_dense_regions("images.jpeg", dense)