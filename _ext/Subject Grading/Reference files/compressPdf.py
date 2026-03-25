# compressPdf.py
#
# PDF Compression Module
#
# COMPRESSION FLOW (Simple Explanation):
# =======================================
#
# Step 1: CHECK FILE SIZE
#   - Read the PDF file and check its size in megabytes (MB)
#   - If the file is smaller than 10MB, do nothing and exit
#   - If the file is 10MB or larger, proceed to compression
#
# Step 2: BACKUP THE ORIGINAL FILE
#   - Rename the original PDF to a temporary backup file (e.g., "output.pdf.tmp")
#   - This keeps the original safe until compression is successful
#   - If anything goes wrong, we can restore the original from this backup
#
# Step 3: COMPRESS EACH PAGE
#   - Open the backup PDF file
#   - For each page in the PDF:
#     a) Convert the page to an image (rasterize it)
#     b) If the image is too large (bigger than 2000 pixels), resize it down
#     c) Compress the image using JPEG format with quality level 75
#     d) Create a new PDF page from the compressed image
#   - Build a new PDF document with all the compressed pages
#
# Step 4: REPLACE THE ORIGINAL FILE
#   - Save the new compressed PDF to the original file path
#   - The original file is now replaced with the smaller compressed version
#
# Step 5: CLEANUP
#   - If compression succeeded: Delete the temporary backup file
#   - If compression failed: Restore the original file from the backup
#
# OPTIONAL RETRY:
#   - If the compressed file is still too large (>= 10MB), try again with:
#     * Lower JPEG quality (more compression)
#     * Smaller maximum image size (more resizing)
#   - This is called "aggressive compression"
#
# RESULT:
#   - The PDF file is now smaller (ideally 5-8MB)
#   - The original file is safely preserved if compression fails
#   - Image quality may be slightly reduced, but the PDF remains readable

import os
import tempfile
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import io


def compress_pdf_if_needed(
    pdf_path: str,
    target_size_mb: float = 10.0,
    max_quality: int = 75,
    max_dimension: int = 2000,
    aggressive: bool = False,
) -> bool:
    """
    Compress a PDF file if its size is >= target_size_mb.
    
    Args:
        pdf_path: Path to the PDF file to compress
        target_size_mb: File size threshold in MB to trigger compression (default: 10.0)
        max_quality: JPEG quality for image compression (1-100, default: 75)
        max_dimension: Maximum width/height in pixels for compressed images (default: 2000)
        aggressive: If True, use more aggressive compression settings (default: False)
    
    Returns:
        True if compression was performed, False otherwise
    """
    if not os.path.exists(pdf_path):
        print(f"  Warning: PDF file not found: {pdf_path}")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"  PDF file size: {file_size_mb:.2f} MB")
    
    if file_size_mb < target_size_mb:
        print(f"  PDF size ({file_size_mb:.2f} MB) is below threshold ({target_size_mb} MB). No compression needed.")
        return False
    
    print(f"  PDF size ({file_size_mb:.2f} MB) exceeds threshold ({target_size_mb} MB). Starting compression...")
    
    # Adjust compression settings if aggressive mode is enabled
    quality = max_quality
    dimension = max_dimension
    if aggressive:
        quality = max(30, max_quality - 20)  # Lower quality
        dimension = max(1500, max_dimension - 500)  # Smaller dimension
        print(f"  Using aggressive compression settings: quality={quality}, max_dimension={dimension}")
    
    # Create temporary backup file
    temp_backup = pdf_path + ".tmp"
    try:
        # Backup original PDF
        os.rename(pdf_path, temp_backup)
        print(f"  Backed up original PDF to: {temp_backup}")
        
        # Open the backup PDF
        doc = fitz.open(temp_backup)
        
        # Create new PDF document for compressed output
        new_doc = fitz.open()
        
        # Process each page
        total_pages = len(doc)
        print(f"  Compressing {total_pages} pages...")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Get page dimensions
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
            
            # Calculate scale factor if page exceeds max_dimension
            scale = 1.0
            if page_width > max_dimension or page_height > max_dimension:
                scale = min(max_dimension / page_width, max_dimension / page_height)
                new_width = int(page_width * scale)
                new_height = int(page_height * scale)
            else:
                new_width = int(page_width)
                new_height = int(page_height)
            
            # Rasterize page to image (matrix for scaling)
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary (JPEG doesn't support transparency)
            if pil_image.mode in ("RGBA", "LA", "P"):
                rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                if pil_image.mode == "P":
                    pil_image = pil_image.convert("RGBA")
                rgb_image.paste(pil_image, mask=pil_image.split()[3] if pil_image.mode == "RGBA" else None)
                pil_image = rgb_image
            elif pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # Compress image to JPEG
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="JPEG", quality=quality, optimize=True)
            img_buffer.seek(0)
            
            # Create new page in output document with compressed image
            new_page = new_doc.new_page(width=page_width, height=page_height)
            
            # Insert compressed image
            img_rect = fitz.Rect(0, 0, page_width, page_height)
            new_page.insert_image(img_rect, stream=img_buffer.getvalue())
            
            # Clean up
            pix = None
            pil_image = None
            img_buffer.close()
            
            if (page_num + 1) % 10 == 0:
                print(f"    Processed {page_num + 1}/{total_pages} pages...")
        
        # Save compressed PDF to original path
        new_doc.save(pdf_path)
        new_doc.close()
        doc.close()
        
        # Check new file size
        new_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        compression_ratio = (1 - (new_size_mb / file_size_mb)) * 100
        
        print(f"  Compression complete!")
        print(f"    Original size: {file_size_mb:.2f} MB")
        print(f"    Compressed size: {new_size_mb:.2f} MB")
        print(f"    Compression ratio: {compression_ratio:.1f}%")
        
        # If still too large and not already in aggressive mode, retry with aggressive settings
        if new_size_mb >= target_size_mb and not aggressive:
            print(f"  Compressed PDF ({new_size_mb:.2f} MB) still exceeds threshold. Retrying with aggressive settings...")
            # Clean up and retry
            os.remove(temp_backup)
            return compress_pdf_if_needed(
                pdf_path=pdf_path,
                target_size_mb=target_size_mb,
                max_quality=max_quality,
                max_dimension=max_dimension,
                aggressive=True,
            )
        
        # Success: delete backup
        os.remove(temp_backup)
        print(f"  Cleaned up temporary backup file.")
        return True
        
    except Exception as e:
        print(f"  Error during compression: {e}")
        # Restore original from backup
        if os.path.exists(temp_backup):
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                os.rename(temp_backup, pdf_path)
                print(f"  Restored original PDF from backup.")
            except Exception as restore_error:
                print(f"  Error restoring backup: {restore_error}")
        return False
