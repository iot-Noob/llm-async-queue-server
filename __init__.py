"""
LLM Async Queue Server - A powerful async LLM service with chart generation capabilities
"""

__version__ = "0.1.0"
__author__ = "Talha"
__description__ = "Async LLM service with chart generation using ECharts"

# Import main classes for easy access
from .main import (
    EcharrParsers,
    AsyncLLM,
    PromptTemplate,
    ChatPromptTemplate,
    StrOutputParser,
    Runnable,
    JsonOutputParser,
    Settings
)

# Define what gets imported with "from package import *"
__all__ = [
    # Core classes
    "EcharrParsers",
    "AsyncLLM",
    "PromptTemplate",
    "ChatPromptTemplate",
    "StrOutputParser",
    "Runnable",
    "JsonOutputParser",
    "Settings",
    
    # Version info
    "__version__",
    "__author__",
    "__description__",
]

# Package metadata
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2024 {__author__}"

# Initialize package-level logger
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def setup_logging(level=logging.INFO):
    """Setup logging for the package"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)
    return logger

# Check dependencies
def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    try:
        import llama_cpp
    except ImportError:
        missing.append("llama-cpp-python")
    
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    
    try:
        import uvloop
    except ImportError:
        missing.append("uvloop")
    
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    
    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )
    
    return True

# Create default instance
_default_parser = None

def get_default_parser():
    """Get or create default parser instance"""
    global _default_parser
    if _default_parser is None:
        _default_parser = EcharrParsers()
    return _default_parser

# Cleanup function
def cleanup():
    """Cleanup resources before exit"""
    global _default_parser
    _default_parser = None
    
    # Force garbage collection
    import gc
    gc.collect()

# Register cleanup on module exit
import atexit
atexit.register(cleanup)