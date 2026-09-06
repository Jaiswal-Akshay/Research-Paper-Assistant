Setup

Clone the repository and enter the project directory:

git clone https://github.com/Jaiswal-Akshay/Research-Paper-Assistant
cd Research-Paper-Assistant

Create and activate a virtual environment.

Windows PowerShell:

python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

Install the dependencies:

pip install -r requirements.txt

Install Ollama and download Llama 3:

ollama pull llama3

Usage

Index papers from arXiv:

python index_arxiv.py --topic "retrieval augmented generation" --max-results 10

Run the command-line assistant:

python ask_arxiv.py

Run the Streamlit application:

streamlit run app.py