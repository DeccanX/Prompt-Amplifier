"""Document loaders for various file formats."""

from prompt_amplifier.loaders.base import BaseLoader, DirectoryLoader

# Always available
from prompt_amplifier.loaders.txt import TxtLoader, MarkdownLoader
from prompt_amplifier.loaders.csv import CSVLoader
from prompt_amplifier.loaders.json import JSONLoader

__all__ = [
    "BaseLoader",
    "DirectoryLoader",
    "TxtLoader",
    "MarkdownLoader",
    "CSVLoader",
    "JSONLoader",
]

# Optional loaders (require extra dependencies)
try:
    from prompt_amplifier.loaders.docx import DocxLoader
    __all__.append("DocxLoader")
except ImportError:
    pass

try:
    from prompt_amplifier.loaders.excel import ExcelLoader
    __all__.append("ExcelLoader")
except ImportError:
    pass

try:
    from prompt_amplifier.loaders.pdf import PDFLoader
    __all__.append("PDFLoader")
except ImportError:
    pass
