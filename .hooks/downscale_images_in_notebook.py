import base64
import io
import os
import sys

import nbformat
from PIL import Image

# --- CONFIGURATION ---
MAX_DIM = 800  # Resize images larger than this
CONVERT_PNG_TO_JPEG = True  # Set to True to convert PNG to JPEG (lossy)


# --- HELPERS ---
def compress_image(encoded_image, image_format, original_mime):
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert all to RGB

    orig_width, orig_height = image.size
    if max(orig_width, orig_height) > MAX_DIM:
        # Resize
        image.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

    # Re-encode
    buffer = io.BytesIO()
    if CONVERT_PNG_TO_JPEG and original_mime == "image/png":
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        new_mime = "image/jpeg"
    else:
        image.save(buffer, format=image_format, optimize=True)
        new_mime = original_mime

    new_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Add display size metadata
    display_metadata = {new_mime: {"width": orig_width, "height": orig_height}}

    return new_data, new_mime, display_metadata


# --- MAIN PROCESSING FUNCTION ---
def process_notebook(path: str) -> bool:
    changed = False
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type != "display_data":
                continue

            data = output.get("data", {})
            for mime in ["image/png", "image/jpeg"]:
                if mime in data:
                    image_format = "PNG" if mime == "image/png" else "JPEG"
                    new_image, new_mime, meta = compress_image(
                        data[mime], image_format, mime
                    )

                    if new_mime != mime:
                        del data[mime]  # Remove old MIME if format changed
                    data[new_mime] = new_image
                    if meta:
                        output.setdefault("metadata", {}).update(meta)

                    changed = True
                    break  # Only handle one image per output

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    return changed


# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    for notebook in sys.argv[1:]:
        if notebook.endswith(".ipynb") and os.path.isfile(notebook):
            if os.path.getsize(notebook) < 512 * 1024:
                print(f"Skipping {notebook} (smaller than 512KB)")
                continue  # Skip small notebooks
            if process_notebook(notebook):
                print(f"✔ Optimized images in {notebook}")
