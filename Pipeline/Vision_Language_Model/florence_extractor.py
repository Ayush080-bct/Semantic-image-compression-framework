from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

class FlorenceExtractor:
    def __init__(self):
        model_id = "microsoft/Florence-2-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading Florence-2 on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        print("Florence-2 ready!")

    def run_task(self, image: Image.Image, task: str) -> str:
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                do_sample=False
            )
        result = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(result, task=task, image_size=image.size)
        return parsed

    def extract(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        return {
            "caption": self.run_task(image, "<DETAILED_CAPTION>"),
            "objects": self.run_task(image, "<OD>"),
            "dense_regions": self.run_task(image, "<DENSE_REGION_CAPTION>"),
            "ocr": self.run_task(image, "<OCR>"),
        }


if __name__ == "__main__":
    extractor = FlorenceExtractor()
    result = extractor.extract("test.png")
    for key, value in result.items():
        print(f"\n--- {key.upper()} ---")
        print(value)