from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
import shutil
import json # For reading json file content
import os
import uuid
import logging
from PIL import Image
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
    PDFPopplerTimeoutError # Added for completeness
)

from chat import DOLPHIN
# Functions from demo_page.py and utils.py will be called
# We need to ensure they are available in the Python path or copy/adapt them.
# For now, attempting direct import path assuming they are structured as modules.
from demo_page import process_page
# setup_output_dirs is called by save_outputs, so not directly needed here.
from utils.utils import prepare_image, parse_layout_string, process_coordinates, save_outputs, ImageDimensions, map_to_original_coordinates, adjust_box_edges
from utils.markdown_utils import MarkdownConverter

# Setup logging
logging.basicConfig(level=logging.INFO) # Configure basic logging
logger = logging.getLogger(__name__)

# Load configuration
logger.info("Loading DOLPHIN model configuration...")
cfg = OmegaConf.load("./config/Dolphin.yaml")

# Create FastAPI app instance, disabling default docs
app = FastAPI(docs_url=None, redoc_url=None, title="DOLPHIN API")

# Initialize DOLPHIN model
logger.info("Initializing DOLPHIN model...")
model = DOLPHIN(cfg)
logger.info("DOLPHIN model initialized.")

# Mount static files directory for Swagger UI
# This should be done after app initialization and before routes that might conflict.
# However, standard practice is often to mount it early.
# For serving Swagger files, it must be available when /docs is hit.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url, # Use app.openapi_url
        title=app.title + " - Swagger UI", # Use app.title
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png"
    )

# Example of how to potentially store in app state (though not strictly necessary for global)
# app.state.model = model

TEMP_UPLOADS_DIR = "temp_uploads"
RESULTS_DIR = "results"
DEFAULT_MAX_BATCH_SIZE = 4

@app.on_event("startup")
async def startup_event():
    """Create temporary and results directories on startup."""
    logger.info(f"Creating temporary directory: {TEMP_UPLOADS_DIR}")
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    logger.info(f"Creating results directory: {RESULTS_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/process/")
async def process_file_endpoint(
    file: UploadFile = File(...),
    output_format: str = Query("json", enum=["json", "markdown"], description="Format for the output: 'json' or 'markdown'")
):
    """
    Accepts an image file (PNG, JPG, JPEG, BMP, TIFF) or a PDF file for processing
    using the DOLPHIN model.

    - For **image files**, the image is processed directly.
    - For **PDF files**, each page is individually converted to an image and then processed.
      PDF processing may take longer.

    The results are returned in the specified format (JSON or Markdown).

    - **file**: The image or PDF file to process.
    - **output_format**: Query parameter to specify the desired output format.
        - **'json'**:
            - For image files: Returns a JSON object with detailed recognition results for the image.
            - For PDF files: Returns a JSON array, where each element is a JSON object
              representing the recognition results for a single page of the PDF.
        - **'markdown'**:
            - For image files: Returns the recognized content in Markdown format as a plain text response.
            - For PDF files: Returns a single Markdown string concatenating the Markdown output
              of all pages, separated by a page break marker (`--- Page Break ---`).
    """
    upload_file_path = None
    request_id = str(uuid.uuid4()) # Generate request_id early for logging

    logger.info(f"Processing request {request_id}. Output format: {output_format}. Input filename: {file.filename}")

    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower() # Ensure lowercase for comparison
        # Allowed file extensions (images and PDF)
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']

        if file_extension not in allowed_extensions:
            logger.warning(f"Request {request_id}: Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid file type. Supported types: {', '.join(allowed_extensions)}. Got: {file_extension}")

        unique_upload_filename = f"{request_id}{os.path.splitext(file.filename)[1]}" # Preserve original case for filename
        upload_file_path = os.path.join(TEMP_UPLOADS_DIR, unique_upload_filename)

        logger.info(f"Request {request_id}: Saving uploaded file to {upload_file_path}")
        try:
            with open(upload_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to save uploaded file {upload_file_path}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        pil_images_from_pdf = [] # To store images if PDF is uploaded

        if file_extension == '.pdf':
            logger.info(f"Request {request_id}: Uploaded file is a PDF. Attempting conversion to images.")
            try:
                pil_images_from_pdf = convert_from_path(upload_file_path, dpi=200)
                if not pil_images_from_pdf: # Should not happen if convert_from_path succeeds with no error
                    logger.error(f"Request {request_id}: PDF conversion resulted in no images for {upload_file_path}.")
                    raise HTTPException(status_code=500, detail="PDF conversion failed, no images produced.")
                logger.info(f"Request {request_id}: Successfully converted PDF {upload_file_path} to {len(pil_images_from_pdf)} image(s).")
                # For now, we just log. Subsequent steps will process these images.
                # The original `upload_file_path` for a PDF will point to the PDF itself,
                # and `process_page` expects an image path. This will need adjustment later.
            except PDFInfoNotInstalledError:
                logger.error(f"Request {request_id}: PDFInfoNotInstalledError - Poppler poppler-utils not installed or not in PATH.", exc_info=True)
                raise HTTPException(status_code=500, detail="PDF processing error: Poppler (poppler-utils) not installed or not found in PATH. Please ensure it is installed.")
            except PDFPageCountError:
                logger.error(f"Request {request_id}: PDFPageCountError - Could not determine page count for {upload_file_path}.", exc_info=True)
                raise HTTPException(status_code=400, detail="Invalid or corrupted PDF: Could not determine page count.")
            except PDFSyntaxError:
                logger.error(f"Request {request_id}: PDFSyntaxError - PDF syntax error in {upload_file_path}.", exc_info=True)
                raise HTTPException(status_code=400, detail="Invalid or corrupted PDF: Syntax error.")
            except PDFPopplerTimeoutError:
                logger.error(f"Request {request_id}: PDFPopplerTimeoutError - Poppler timed out processing {upload_file_path}.", exc_info=True)
                raise HTTPException(status_code=500, detail="PDF processing error: Poppler timed out.")
            except Exception as e: # Catch any other pdf2image errors or general errors
                logger.error(f"Request {request_id}: An unexpected error occurred during PDF conversion of {upload_file_path}. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to convert PDF to images: {str(e)}")
        else:
            # This is an image file, proceed with existing logic (which uses upload_file_path)
            logger.info(f"Request {request_id}: Uploaded file is an image: {upload_file_path}")
            # No changes here for now, process_page will be called with upload_file_path later

        # Define save directory for results for this specific file
        output_save_dir = os.path.join(RESULTS_DIR, request_id)
        logger.info(f"Request {request_id}: Creating output directory {output_save_dir}")
        try:
            os.makedirs(output_save_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to create output directory {output_save_dir}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create output directory: {str(e)}")


        if pil_images_from_pdf:
            logger.info(f"Request {request_id}: Processing {len(pil_images_from_pdf)} images converted from PDF.")
            all_pages_recognition_results = []
            all_pages_markdown_paths = []

            for i, page_image in enumerate(pil_images_from_pdf):
                page_num = i + 1
                temp_page_filename = f"{request_id}_page_{page_num}.png"
                temp_page_image_path = os.path.join(TEMP_UPLOADS_DIR, temp_page_filename)

                logger.info(f"Request {request_id}: Saving temporary image for page {page_num} to {temp_page_image_path}")
                try:
                    page_image.save(temp_page_image_path, "PNG")
                except Exception as e:
                    logger.error(f"Request {request_id}: Failed to save temporary image for page {page_num}. Error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to save image for page {page_num}: {str(e)}")

                # Define a specific output subdirectory for this page's results within the main request's output_save_dir
                # This helps in keeping page-specific outputs organized if needed, though process_page uses unique names based on input.
                # For `save_outputs` to work as expected (creating unique filenames in `output_save_dir/markdown` and `output_save_dir/recognition_json`),
                # we pass `output_save_dir` directly. `process_page` uses the basename of `temp_page_image_path`.

                page_specific_output_basename = os.path.splitext(temp_page_filename)[0]

                try:
                    logger.info(f"Request {request_id}: Processing page {page_num} ({temp_page_image_path}) with DOLPHIN model.")
                    # `process_page` saves its own JSON and MD files based on the input image path's basename.
                    page_json_path, page_recognition_results = process_page(
                        image_path=temp_page_image_path,
                        model=model,
                        save_dir=output_save_dir, # Main output directory for the request
                        max_batch_size=DEFAULT_MAX_BATCH_SIZE
                    )
                    all_pages_recognition_results.append(page_recognition_results)

                    # Construct expected markdown path based on how save_outputs works
                    expected_md_filename = f"{page_specific_output_basename}.md"
                    page_markdown_path = os.path.join(output_save_dir, "markdown", expected_md_filename)
                    all_pages_markdown_paths.append(page_markdown_path)

                    logger.info(f"Request {request_id}: Page {page_num} processed. JSON: {page_json_path}, MD: {page_markdown_path}")

                except Exception as e:
                    logger.error(f"Request {request_id}: Error processing page {page_num} ({temp_page_image_path}). Error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error processing page {page_num} of PDF: {str(e)}")
                finally:
                    # Clean up temporary page image file
                    if os.path.exists(temp_page_image_path):
                        try:
                            os.remove(temp_page_image_path)
                            logger.info(f"Request {request_id}: Successfully removed temporary page image {temp_page_image_path}")
                        except Exception as e:
                            logger.error(f"Request {request_id}: Failed to remove temporary page image {temp_page_image_path}. Error: {e}", exc_info=True)

            # Aggregated results processing
            if output_format == "markdown":
                logger.info(f"Request {request_id}: Aggregating Markdown output for {len(all_pages_markdown_paths)} pages.")
                combined_markdown = []
                for md_path in all_pages_markdown_paths:
                    if not os.path.exists(md_path):
                        logger.warning(f"Request {request_id}: Markdown file {md_path} not found for page. Skipping.")
                        combined_markdown.append(f"\n\n--- Error: Page Markdown not found at {os.path.basename(md_path)} ---\n\n")
                        continue
                    try:
                        with open(md_path, "r", encoding="utf-8") as md_file:
                            combined_markdown.append(md_file.read())
                    except Exception as e:
                        logger.error(f"Request {request_id}: Failed to read markdown file {md_path}. Error: {e}", exc_info=True)
                        combined_markdown.append(f"\n\n--- Error reading page Markdown: {os.path.basename(md_path)} ---\n\n")
                logger.info(f"Request {request_id}: Successfully aggregated Markdown. Returning content.")
                return PlainTextResponse(content="\n\n---\nPage Break\n---\n\n".join(combined_markdown), media_type="text/markdown")
            else: # Default to JSON
                logger.info(f"Request {request_id}: Returning aggregated JSON results for {len(all_pages_recognition_results)} pages.")
                return JSONResponse(content=all_pages_recognition_results)

        else: # This is for direct image uploads (existing logic)
            logger.info(f"Request {request_id}: Starting DOLPHIN model processing for image {upload_file_path}")
            try:
                # `process_page` saves its own JSON and MD files.
                json_path, _ = process_page(
                    image_path=upload_file_path,
                    model=model,
                    save_dir=output_save_dir,
                    max_batch_size=DEFAULT_MAX_BATCH_SIZE
                )
                logger.info(f"Request {request_id}: DOLPHIN model processing completed for image. JSON path: {json_path}")
            except FileNotFoundError as e:
                logger.error(f"Request {request_id}: File not found during DOLPHIN model processing. Error: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"File not found during model processing: {str(e)}")
            except Exception as e:
                logger.error(f"Request {request_id}: Error during DOLPHIN model processing for image. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error during DOLPHIN model processing: {str(e)}")

            # Determine output based on output_format for single image
            if output_format == "markdown":
                base_filename = os.path.splitext(unique_upload_filename)[0]
                markdown_file_path = os.path.join(output_save_dir, "markdown", f"{base_filename}.md")
                logger.info(f"Request {request_id}: Attempting to read Markdown output from {markdown_file_path}")
                if not os.path.exists(markdown_file_path):
                    logger.error(f"Request {request_id}: Markdown file not found at {markdown_file_path}")
                    raise HTTPException(status_code=404, detail=f"Markdown file not found: {markdown_file_path}")
                try:
                    with open(markdown_file_path, "r", encoding="utf-8") as md_file:
                        markdown_content = md_file.read()
                    logger.info(f"Request {request_id}: Successfully read Markdown file. Returning content.")
                    return PlainTextResponse(content=markdown_content, media_type="text/markdown")
                except Exception as e:
                    logger.error(f"Request {request_id}: Failed to read Markdown file {markdown_file_path}. Error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to read Markdown file: {str(e)}")
            else: # Default to JSON for single image
                logger.info(f"Request {request_id}: Attempting to read JSON output from {json_path}")
                if not os.path.exists(json_path):
                    logger.error(f"Request {request_id}: JSON results file not found at {json_path}")
                    raise HTTPException(status_code=404, detail=f"JSON results file not found: {json_path}")
                try:
                    with open(json_path, "r", encoding="utf-8") as f_json:
                        json_content = json.load(f_json)
                    logger.info(f"Request {request_id}: Successfully read and parsed JSON file. Returning content.")
                    return JSONResponse(content=json_content)
                except Exception as e:
                    logger.error(f"Request {request_id}: Failed to read or parse JSON results file {json_path}. Error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to read or parse JSON results file: {str(e)}")

    except HTTPException as e: # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Request {request_id}: An unexpected error occurred. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if file:
            try:
                file.file.close()
            except Exception as e:
                logger.warning(f"Request {request_id}: Error closing uploaded file stream. Error: {e}", exc_info=True)

        # Clean up the uploaded temporary file
        if upload_file_path and os.path.exists(upload_file_path):
            try:
                os.remove(upload_file_path)
                logger.info(f"Request {request_id}: Successfully removed temporary upload file {upload_file_path}")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to remove temporary upload file {upload_file_path}. Error: {e}", exc_info=True)
                max_batch_size=DEFAULT_MAX_BATCH_SIZE
            )
            logger.info(f"Request {request_id}: DOLPHIN model processing completed. JSON path: {json_path}")
        except FileNotFoundError as e: # Specific to process_page if it expects files that aren't there
            logger.error(f"Request {request_id}: File not found during DOLPHIN model processing. Error: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"File not found during model processing: {str(e)}")
        except Exception as e:
            logger.error(f"Request {request_id}: Error during DOLPHIN model processing. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during DOLPHIN model processing: {str(e)}")

        # Determine output based on output_format
        if output_format == "markdown":
            base_filename = os.path.splitext(unique_upload_filename)[0]
            markdown_file_path = os.path.join(output_save_dir, "markdown", f"{base_filename}.md")
            logger.info(f"Request {request_id}: Attempting to read Markdown output from {markdown_file_path}")

            if not os.path.exists(markdown_file_path):
                logger.error(f"Request {request_id}: Markdown file not found at {markdown_file_path}")
                raise HTTPException(status_code=404, detail=f"Markdown file not found: {markdown_file_path}")
            try:
                with open(markdown_file_path, "r", encoding="utf-8") as md_file:
                    markdown_content = md_file.read()
                logger.info(f"Request {request_id}: Successfully read Markdown file. Returning content.")
                return PlainTextResponse(content=markdown_content, media_type="text/markdown")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to read Markdown file {markdown_file_path}. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to read Markdown file: {str(e)}")

        else: # Default to JSON
            logger.info(f"Request {request_id}: Attempting to read JSON output from {json_path}")
            if not os.path.exists(json_path):
                logger.error(f"Request {request_id}: JSON results file not found at {json_path}")
                raise HTTPException(status_code=404, detail=f"JSON results file not found: {json_path}")
            try:
                with open(json_path, "r", encoding="utf-8") as f_json:
                    json_content = json.load(f_json)
                logger.info(f"Request {request_id}: Successfully read and parsed JSON file. Returning content.")
                return JSONResponse(content=json_content)
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to read or parse JSON results file {json_path}. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to read or parse JSON results file: {str(e)}")

    except HTTPException as e: # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Request {request_id}: An unexpected error occurred. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if file:
            try:
                file.file.close()
            except Exception as e:
                logger.warning(f"Request {request_id}: Error closing uploaded file stream. Error: {e}", exc_info=True)

        # Clean up the uploaded temporary file
        if upload_file_path and os.path.exists(upload_file_path):
            try:
                os.remove(upload_file_path)
                logger.info(f"Request {request_id}: Successfully removed temporary upload file {upload_file_path}")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to remove temporary upload file {upload_file_path}. Error: {e}", exc_info=True)
        # Optionally, clean up the uploaded temporary file if it's no longer needed
        # For now, keeping it for debugging, but in production, you might delete it:
        # if upload_file_path and os.path.exists(upload_file_path):
        #     os.remove(upload_file_path)


@app.get("/")
async def root():
    return {"message": "DOLPHIN API is running"}

# Further endpoints will be added in subsequent steps.
