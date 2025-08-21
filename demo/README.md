# ASearcher Visual Demo

This directory contains the visual demo for ASearcher, allowing you to interact with the agent through a web interface.

## Files

- `asearcher_demo.py`: The FastAPI backend service of the demo.
- `asearcher_client.html`: The main HTML file for the client-side interface.
- `client_styles.css`: CSS styles for the client interface.
- `client_script.js`: JavaScript logic for the client interface.
- `start_visual_demo.sh`: A script to easily start the demo service.

## Quickstart

1. **Start the vLLM Server:**
    Before running the demo, you need to have a vLLM server running with the desired model, for example:
    ```bash
    vllm serve path/to/model --host $host --port $port
    ```

2.  **Start the Demo Service:**  
    Replace parameters with the actual path to your service in `demo/start_visual_demo.sh`:
    -   `HOST`: Server host (default: `0.0.0.0`).
    -   `PORT`: Server port (default: `8080`).
    -   `VLLM_URL`: URL of the vLLM server you deploy (default: `http://0.0.0.0:50000`).
    -   `MODEL_NAME`: Model name to use (default: `ASearcher-Web-7B`). You can get this model from [🤗huggingface](https://huggingface.co/inclusionAI/ASearcher-Web-7B)
    Use the provided shell script to start the FastAPI backend.

    Then start the demo service:
    ```bash
    bash demo/start_visual_demo.sh
    ```

3.  **Open the Client:**
    Open the `asearcher_client.html` file in your web browser to access the user interface.

    You can open it directly from your file system, or serve it via a simple HTTP server.

## Backend Service (`asearcher_demo.py`)

The backend is a FastAPI application that provides the following endpoints.

The service uses an `AsyncVLLMClient` to communicate with a vLLM server using an OpenAI-compatible API. It manages the agent's multi-turn reasoning and tool-use process in the background.

### Command-line Arguments

You can customize the service behavior using these command-line arguments:

-   `--host`: Server host (default: `0.0.0.0`).
-   `--port`: Server port (default: `8080`).
-   `--llm-url`: URL of the vLLM server (default: `http://localhost:50000`).
-   `--model-name`: Model name to use (default: `ASearcher-Web-7B`).
-   `--api-key`: API key for the LLM service (default: `EMPTY`).
-   `--reload`: Enable auto-reloading for development.