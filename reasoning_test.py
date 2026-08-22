import requests

diagram_xml = """<ImageSemantic type="document">
  <VLM>
    <Caption>The image shows a flowchart diagram of a computer system with arrows indicating data flow between components.</Caption>
  </VLM>
  <OCR>
    <text bbox="181,26,251,43" confidence="0.999" quality="high">Input Image</text>
    <text bbox="72,86,129,104" confidence="0.998" quality="high">EasyOCR</text>
    <text bbox="303,87,363,101" confidence="0.817" quality="high">Florence 2</text>
    <text bbox="179,169,253,183" confidence="0.967" quality="high">Text coverage</text>
    <text bbox="135,211,157,225" confidence="0.825" quality="high">yes</text>
    <text bbox="55,257,147,271" confidence="0.972" quality="high">Document Route</text>
    <text bbox="297,257,367,271" confidence="0.993" quality="high">Photo Route</text>
    <text bbox="169,320,265,338" confidence="0.899" quality="high">Semantic Mapper</text>
    <text bbox="181,383,253,397" confidence="0.905" quality="high">XML Output</text>
  </OCR>
</ImageSemantic>"""

questions = [
    "What component processes the semantic representation before generating the final output?",
    "How many high quality text detections are there in the document?",
    "What are all the components mentioned in this pipeline from input to output in order?",
]

for q in questions:
    payload = {
        "model": "llama3.2",
        "prompt": f"""You are analyzing an image represented as a semantic XML document.

{diagram_xml}

Answer this question based only on the XML above. Be concise.
Question: {q}""",
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    answer = response.json()["response"]
    print(f"Q: {q}")
    print(f"A: {answer.strip()}")
    print()