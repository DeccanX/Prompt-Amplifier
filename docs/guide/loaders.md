# Document Loading

Load documents from various formats into Prompt Amplifier.

## Supported Formats

| Format | Extension | Loader | Notes |
|--------|-----------|--------|-------|
| Plain Text | `.txt` | `TxtLoader` | UTF-8 encoding |
| PDF | `.pdf` | `PDFLoader` | Requires `pymupdf` |
| Word | `.docx` | `DocxLoader` | Requires `python-docx` |
| Excel | `.xlsx` | `ExcelLoader` | Requires `openpyxl` |
| CSV | `.csv` | `CSVLoader` | Uses pandas |
| JSON | `.json` | `JSONLoader` | Arrays or objects |

## Quick Start

### Load from File

```python
from prompt_amplifier import PromptForge

forge = PromptForge()

# Single file
forge.load_documents("./data/manual.pdf")

# Directory (all supported formats)
forge.load_documents("./docs/")
```

### Add Text Directly

```python
forge.add_texts([
    "First document content...",
    "Second document content...",
])
```

## Loaders in Detail

### TxtLoader

Load plain text files:

```python
from prompt_amplifier.loaders import TxtLoader

loader = TxtLoader()
documents = loader.load("./data/readme.txt")

# Options
loader = TxtLoader(
    encoding="utf-8",      # File encoding
    split_on="\n\n"        # Split into multiple docs
)
```

### PDFLoader

Load PDF documents:

```python
from prompt_amplifier.loaders import PDFLoader

loader = PDFLoader()
documents = loader.load("./reports/quarterly.pdf")

# Options
loader = PDFLoader(
    extract_images=False,  # Skip image extraction
    page_numbers=[1, 2, 3] # Specific pages only
)
```

!!! note "Installation"
    Requires `pymupdf`: `pip install pymupdf`

### DocxLoader

Load Microsoft Word documents:

```python
from prompt_amplifier.loaders import DocxLoader

loader = DocxLoader()
documents = loader.load("./docs/proposal.docx")

# Options
loader = DocxLoader(
    include_tables=True,   # Include table content
    include_headers=True   # Include headers/footers
)
```

!!! note "Installation"
    Requires `python-docx`: `pip install python-docx`

### ExcelLoader

Load Excel spreadsheets:

```python
from prompt_amplifier.loaders import ExcelLoader

loader = ExcelLoader()
documents = loader.load("./data/sales.xlsx")

# Options
loader = ExcelLoader(
    sheet_name="Sheet1",   # Specific sheet
    header_row=0           # Row for column headers
)
```

!!! note "Installation"
    Requires `openpyxl`: `pip install openpyxl`

### CSVLoader

Load CSV files:

```python
from prompt_amplifier.loaders import CSVLoader

loader = CSVLoader()
documents = loader.load("./data/customers.csv")

# Options
loader = CSVLoader(
    content_columns=["name", "description"],  # Columns to use
    metadata_columns=["id", "category"],      # Columns as metadata
    delimiter=","
)
```

### JSONLoader

Load JSON files:

```python
from prompt_amplifier.loaders import JSONLoader

loader = JSONLoader()
documents = loader.load("./data/products.json")

# For JSON arrays
loader = JSONLoader(
    content_key="description",  # Field for content
    metadata_keys=["id", "name"] # Fields for metadata
)
```

### DirectoryLoader

Load all files in a directory:

```python
from prompt_amplifier.loaders import DirectoryLoader

loader = DirectoryLoader(
    glob_pattern="**/*.pdf",    # Only PDFs
    recursive=True              # Include subdirectories
)
documents = loader.load("./docs/")
```

## Metadata

Documents include metadata about their source:

```python
documents = loader.load("./data/file.pdf")

for doc in documents:
    print(doc.source)     # File path
    print(doc.page)       # Page number (for PDFs)
    print(doc.metadata)   # Additional metadata
```

## Custom Loaders

Create your own loader by extending `BaseLoader`:

```python
from prompt_amplifier.loaders.base import BaseLoader
from prompt_amplifier.models import Document

class MyLoader(BaseLoader):
    """Custom loader for my format."""
    
    def load(self, path: str) -> list[Document]:
        # Read your file format
        with open(path, "r") as f:
            content = f.read()
        
        # Return Document objects
        return [
            Document(
                content=content,
                metadata={"source": path}
            )
        ]
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".myformat"]
```

## Best Practices

### 1. Organize Your Documents

```
docs/
├── policies/
│   ├── hr_policy.pdf
│   └── security_policy.docx
├── products/
│   ├── catalog.xlsx
│   └── descriptions.json
└── faqs/
    └── common_questions.txt
```

### 2. Use Metadata

```python
forge.add_texts(
    texts=["Document content..."],
    metadata=[{"category": "policy", "department": "HR"}]
)
```

### 3. Handle Large Files

For large files, consider:

- Loading in chunks
- Using streaming loaders
- Pre-processing to extract relevant sections

### 4. Validate Before Loading

```python
import os

path = "./data/file.pdf"
if os.path.exists(path):
    forge.load_documents(path)
else:
    print(f"File not found: {path}")
```

## Next Steps

- [Embedders Guide](embedders.md) - Convert documents to vectors
- [Vector Stores Guide](vectorstores.md) - Store for retrieval

