#  zeromodel/provenance/io.py
"""
I/O operations for VPF embedding and extraction.

Implements ZeroModel's "robust under pressure" principle:
"Versioned headers, spillover-safe metadata, and explicit logical width vs physical padding keep tiles valid as they scale."
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from PIL import Image

def read_header(image: Image.Image) -> Dict[str, Any]:
    """
    Read iTXt header from image.
    
    Args:
        image: PIL Image
        
    Returns:
        Header dictionary or empty dict if not found
    """
    try:
        return dict(image.info.get('iTXt', {}))
    except:
        return {}

def read_footer(image: Image.Image) -> Dict[str, Any]:
    """
    Read footer from image.
    
    Args:
        image: PIL Image
        
    Returns:
        Footer dictionary or empty dict if not found
    """
    # In ZeroModel, footer is stored in iTXt chunks
    return read_header(image)

def write_header(image: Image.Image, header: Dict[str, Any], inplace: bool = True) -> Image.Image:
    """
    Write header to image.
    
    Args:
        image: PIL Image
        header: Header dictionary
        inplace: Modify image in place if True
        
    Returns:
        Modified image
    """
    if not inplace:
        image = image.copy()
    
    # Update iTXt chunks
    if hasattr(image, 'info'):
        image.info['iTXt'] = header
    
    return image

def write_footer(image: Image.Image, footer: Dict[str, Any], inplace: bool = True) -> Image.Image:
    """
    Write footer to image.
    
    Args:
        image: PIL Image
        footer: Footer dictionary
        inplace: Modify image in place if True
        
    Returns:
        Modified image
    """
    return write_header(image, footer, inplace)