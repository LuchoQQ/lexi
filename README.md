# Knowledge Augmented Legal Assistant

A modular, asynchronous legal assistant framework that combines vector embeddings with knowledge graphs for enhanced legal text analysis and retrieval.

## Features

- **Modular Architecture**: Clean separation of components for easier maintenance and testing
- **Asynchronous Processing**: Uses asyncio for improved performance in I/O-bound operations
- **Batch Processing**: Efficiently processes large datasets in batches
- **Enhanced Knowledge Graph**: Improved entity normalization and relation extraction
- **Hybrid Retrieval**: Combines vector search with graph-based retrieval
- **Query Expansion**: Enhances queries with synonyms and related terms
- **Legal Domain Specialization**: Optimized for legal text analysis
- **Error Handling**: Robust error handling with specific exception types

## Project Structure

```
legal_assistant/
├── __init__.py
├── config.py                  # Configuration handling
├── main.py                    # Main entry point
├── exceptions.py              # Custom exceptions
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Logging configuration
│   └── text_processing.py     # Text processing utilities
├── data/
│   ├── __init__.py
│   ├── loader.py              # Data loading modules
│   └── processor.py           # Data processing utilities
├── embeddings/
│   ├── __init__.py
│   ├── model.py               # Embedding model handler
│   └── vector_store.py        # Vector database interface
├── knowledge_graph/
│   ├── __init__.py
│   ├── builder.py             # Graph construction
│   ├── queries.py             # Graph query operations
│   └── visualizer.py          # Graph visualization
└── assistant/
    ├── __init__.py
    ├── retriever.py           # Information retrieval
    ├── ranker.py              # Result ranking
    └── generator.py           # Response generation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-assistant.git
   cd legal-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a configuration file `config.json` with the following structure:

```json
{
  "embedding_model": "intfloat/multilingual-e5-large",
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "generation_model": "gpt-4-turbo",
  
  "embedding_cache_dir": "./cache/embeddings",
  "chroma_db_dir": "./data/chroma_db",
  "graph_path": "./data/knowledge_graph.pkl",
  "synonyms_path": "./data/legal_synonyms.json",
  
  "collection_name": "codigo_penal_chunks",
  "batch_size": 100,
  
  "retrieval_top_k": 10,
  "use_hybrid_search": true,
  "use_query_expansion": true,
  
  "max_tokens": 1500,
  "temperature": 0.2
}
```

Alternatively, you can set these values through environment variables.

## Usage

### Processing a Legal Code JSON File

```bash
python -m legal_assistant.main --config config.json --json_path data/codigo_penal.json
```

### Querying the Assistant

```bash
python -m legal_assistant.main --config config.json --query "¿Cuál es la diferencia entre dolo y culpa en el Código Penal?"
```

### Interactive Mode

```bash
python -m legal_assistant.main --config config.json
```

## API Usage

```python
import asyncio
from legal_assistant.config import Config
from legal_assistant.assistant.generator import LegalAssistant

async def main():
    # Initialize with configuration
    config = Config("config.json")
    assistant = LegalAssistant(config)
    
    # Load data (only needed once)
    result = await assistant.load_data("data/codigo_penal.json")
    print(f"Loaded {result['chunks_loaded']} chunks")
    
    # Process a query
    response = await assistant.process_query("¿Qué es la legítima defensa?")
    print(response["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

