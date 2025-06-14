# CRAS
CRAS is a text-driven assistant designed to act as a personalized cognitive partner. It ingests, processes, and understands your documents, audio notes, and text files to build a persistent, interconnected knowledge base. Using this knowledge, CRAS can answer complex questions, provide summaries, and help you find connections in your information, all through a conversational interface.

This project is built using a local-first approach, leveraging powerful open-source models optimized for Apple Silicon (MLX) to ensure privacy and performance. The core functionality is built on a Retrieval-Augmented Generation (RAG) pipeline.

## Core Features
**Multimodal Ingestion:** Process PDF documents, plain text files, and audio files (`.mp3`, `.wav`, etc.).

**Intelligent Q&A:** Ask complex questions about the content of your uploaded files. The assistant uses its knowledge base to provide contextually relevant answers.

**Local-First & Private:** All models (LLM, ASR, TTS, Embeddings) run directly on your machine, ensuring your data remains private.

**Persistent Knowledge:** Your processed files create a knowledge base for your session that the assistant uses to answer questions.

## Architecture & Directory Structure
The project is structured to be modular and scalable, separating concerns between data processing, external model services, and the user interface.

```
CRAS/
├── app.py                      # Main Streamlit application file
├── requirements.txt            # All Python package dependencies
├── setup_nltk.py               # Script to download and configure NLTK data
├── fix_signatures.sh           # (macOS only) Script to fix code signature issues
│
├── src/                        # Main source code directory
│   ├── __init__.py
|   ├── config.py               # Config file which has Base Settings for the Models
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_parser.py  # Extracts text from PDF/TXT files and Splits text into manageable chunks
│   │
│   ├── external_services/
│   │   ├── __init__.py
│   │   ├── asr_client.py       # Handles Speech-to-Text with Lightning Whisper MLX
│   │   ├── embedding_client.py # Creates text embeddings with SentenceTransformers
│   │   ├── llm_client.py       # Interacts with the LLM using MLX LM
│   │   └── tts_client.py       # Handles Text-to-Speech with MeloTTS
│   │
│   └── utils/
|       ├── __init__.py
│       └── logger_config.py    # Standardized logging setup
│
├── tests/                      # Test scripts for individual components
│   ├── __init__.py
│   ├── asr_test.py
│   ├── docs_test.py
│   ├── embedding_test.py
│   ├── llm_test.py
│   └── tts_test.py
│
└── data/                       # Directory for temporary files and persistent data
```

## Models Used
Locally-run models
| Task     | Model Library |  Example Model  |  Role in Project  |
| ----------- | ----------- |  ------------  |  ---------------  |
| **LLM**      | `mlx-lm`       |  mlx-community/Phi-3-mini-4k-instruct-4bit-mlx  |  The core "brain" of the assistant. It answers questions and generates responses based on the context provided from the user's documents.|
|  **ASR (Speech-to-Text)**  |  `lightning-whisper-mlx`  |  large-v3  |  Transcribes the user's spoken questions or audio files into text so the LLM can understand them.  |
|  **TTS (Text-to-Speech)**  |  `MeloTTS`  |  EN-US Speaker  |  Converts the LLM's text-based answers into audible speech, allowing the assistant to talk back.  |
|  **Embedding**  |  `sentence-transformers`  |  all-MiniLM-L6-v2  |  Converts text chunks from documents and user queries into numerical vectors (embeddings). This allows the system to find the most relevant information using math.  |

## Setup and Installation Guide
Setting up this project involves navigating several common hurdles in the ML development ecosystem.

### Prerequisites
```
Python 3.12.11
Homebrew
```

### Install System-Level Dependencies (macOS)
Some Python packages rely on underlying system software.
```
# Install the MeCab engine and dictionary, required for multi-lingual text processing by MeloTTS
brew install mecab
brew install mecab-ipadic
```

### Set Up Python Virtual Environment
It is crucial to use a virtual environment to avoid conflicts.
```
# Navigate to the project root directory
cd /path/to/CRAS/

# Create a virtual environment
python3 -m venv lib # or .venv

# Activate it
source lib/bin/activate
```

### Install Python Packages
This project has several tricky dependencies. Install them in this specific order to avoid common errors.
```
# Manually install a modern, compatible version of 'tokenizers'
pip install "tokenizers==0.19.1"

# Install the main ML libraries and their dependencies from requirements.txt
pip install -r requirements.txt

# Install MeloTTS directly from GitHub to bypass its broken PyPI package
pip install git+https://github.com/myshell-ai/MeloTTS.git

# If you encountered a MeCab configuration error, reinstall mecab-python3 with a path hint
# (Replace the path with the output of `which mecab-config`)
env MECAB_CONFIG="/opt/homebrew/bin/mecab-config" pip install mecab-python3==1.0.9 --force-reinstall --no-cache-dir
```

### Download NLTK Data
The text processing pipeline requires data packages from the NLTK library. Run our setup script to download and configure them correctly.
```
python3 setup_nltk.py
```

### Fix macOS Code Signature Issues
On Apple Silicon Macs, the Gatekeeper security feature may block the compiled libraries used by our packages. Run the provided script to approve them all.
```
# Make the script executable
chmod +x fix_signatures.sh

# Run the script
./fix_signatures.sh
```

## How to Run the Application
Once the setup is complete, running the assistant is simple.

Activate your virtual environment:
```
source lib/bin/activate
```

Run the Streamlit app:
```
streamlit run app.py
```

Your web browser will open with the CRAS interface.

## Troubleshooting Common Setup Issues

- **Problem:** `MeloTTS` installation fails with `FileNotFoundError: requirements.txt`

  - **Reason:** The package on PyPI is broken and missing its requirements file.

  - **Solution:** Install directly from GitHub where the file exists: `pip install git+https://github.com/myshell-ai/MeloTTS.git`

- **Problem:** `MeloTTS` installation fails with `RuntimeError: Have you installed MeCab?`

  - **Reason:** A dependency (`fugashi`) needs the MeCab system library but can't find it.

  - **Solution:** Install MeCab using Homebrew: `brew install mecab mecab-ipadic`

- **Problem:** `tokenizers` fails to build with a Rust compiler error

  - **Reason:** `MeloTTS` depends on an old version of `transformers`, which in turn depends on an old version of `tokenizers` that is incompatible with modern Rust compilers on macOS.

  - **Solution:** Manually install a modern, working version of `tokenizers` before installing the other packages: `pip install "tokenizers==0.19.1"`

- **Problem:** `ModuleNotFoundError` when running scripts (e.g., `No module named 'src.utils'`)

  - **Reason:** Python doesn't know your project structure. This happens when running a file directly instead of as a module, or when `__init__.py` files are missing.

  - **Solution:** Ensure every folder in `src/` and `tests/` has an `__init__.py` file. Always run test scripts from the root directory using the `-m` flag (e.g., `python3 -m tests.llm_test`).

- **Problem:** `ImportError: ... not valid for use in process: library load disallowed by system policy`

  - **Reason:** macOS Gatekeeper is blocking compiled library files (`.so`, `.dylib`) because they haven't been notarized by Apple.

  - **Solution:** Manually approve the files using the `xattr` command or the provided `fix_signatures.sh` script.

- **Problem:** `lighting-whisper-mlx` installation failed, `ERROR: Failed to build installable wheels for some pyproject.toml based projects (tiktoken)`
  - **Reason:** The Rust compiler is not installed on your Mac or The Rust compiler is outdated and doesn't support the features used in `tiktoken`'s code.
  - **Solution:** Install the Rust by running this official command
    ```
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```