import easyocr
import gradio as gr
import numpy as np
from PIL import Image
from functools import lru_cache

# ----------------------------
# Load OCR model (cached)
# ----------------------------
@lru_cache(maxsize=1)
def load_model():
    reader = easyocr.Reader(['en'], model_storage_directory='.')
    return reader


reader = load_model()


# ----------------------------
# OCR Function
# ----------------------------
def extract_text(image: Image.Image):
    if image is None:
        return []

    image_np = np.array(image)
    result = reader.readtext(image_np)

    extracted_text = [text[1] for text in result]
    return extracted_text


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="Easy OCR - Extract Text from Images") as demo:

    gr.Markdown("# 🧠 Easy OCR - Extract Text from Images")
    gr.Markdown(
        "## Optical Character Recognition using `easyocr` + `gradio`\n\n"
        "🔗 **Original Streamlit App:** "
        "[image-to-text-app on 🤗 Spaces](https://huggingface.co/spaces/Amrrs/image-to-text-app)"
    )

    with gr.Row():
        image_input = gr.Image(
            label="Upload your image here",
            type="pil"
        )

    extract_button = gr.Button("🔍 Extract Text")

    output_text = gr.JSON(
        label="Extracted Text"
    )

    extract_button.click(
        fn=extract_text,
        inputs=image_input,
        outputs=output_text
    )


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    demo.launch()
