# LAWA RAG Agent

A modular RAG (Retrieval-Augmented Generation) system for UAE government-related queries with query rewriting, context filtering, and domain-specific knowledge expansion.

## Project Structure

The project is organized into modular components:

```
lawa-rag-agent/
├── app.py                    # Main application entry point and API endpoints
├── modules/                  # Modular components
│   ├── __init__.py           # Makes modules a package
│   ├── config.py             # Configuration, environment variables, system prompt
│   ├── citations.py          # Citation processing utilities
│   ├── query_rewriting.py    # Query rewriting and domain knowledge expansion
│   ├── retrieval.py          # Document retrieval and reranking
│   ├── schemas.py            # Pydantic models for data validation
│   └── utils.py              # Utility functions
├── .env                      # Environment variables (not in version control)
└── combined_vectorstore.json # BM25 sparse vectors for hybrid search
```

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   SECRET_KEY=your_backend_secret_key
   PINECONE_INDEX_NAME=combined-vectorstore
   BACKEND_URL=http://localhost:8000
   SERVICE_NAME=lawa-rag
   SERVICE_ENVIRONMENT=development
   ```

4. Run the application:
   ```
   uvicorn app:app --reload
   ```

## API Endpoints

- WebSocket: `/chat` - For real-time chat interactions
- HTTP POST: `/telegram-chat` - For Telegram bot integration
- HTTP GET: `/health` - Fast liveness and high-level readiness
- HTTP GET: `/health/detailed` - Dependency and release metadata breakdown
- HTTP GET: `/health/generation` - Synthetic end-to-end RAG generation probe

## Features

- **Query Rewriting**: Rewrites user queries for better retrieval performance
- **Message History Filtering**: Keeps only relevant conversation context
- **Domain Knowledge Expansion**: Enhances queries with UAE-specific terminology
- **Out-of-Scope Detection**: Directly responds to queries outside the system's scope
- **Clarification Requests**: Asks for more information when queries are ambiguous
- **Citation Processing**: Extracts and formats citations from responses
- **Fallback Search**: Uses Tavily search when Pinecone retrieval yields no results 

## Testing And CI

This repository uses a layered CI strategy for chatbot reliability:

- fast mocked tests on every push and pull request
- live provider and local live-app validation on `main`
- manual post-deploy smoke checks for the deployed environment

For a reusable guide you can apply to similar chatbot projects, see [`docs/chatbot-ci-pipeline.md`](docs/chatbot-ci-pipeline.md).

# Load Testing with Locust

This repository contains a load testing script using Locust to test the WebSocket chat endpoint.

## Prerequisites

Before running the load test, you need to install the required packages:

```bash
pip install locust websocket-client
```

## Running the Load Test

1. Make sure your application server is running and accessible.

2. Start the Locust web interface:

```bash
locust -f locustfile.py
```

3. Open your browser and navigate to `http://localhost:8089`

4. Configure the test parameters:
   - Host: Enter the URL of your application (e.g., `ws://localhost:8000` or `wss://your-domain.com`)
   - Number of users: Set to 100 (or your desired number of concurrent users)
   - Spawn rate: How quickly to spawn users (e.g., 10 users per second)

5. Start the test by clicking "Start swarming"

6. Monitor the results in real-time through the Locust web interface

## Test Configuration

The test simulates users asking random legal questions through the WebSocket chat endpoint. Each user:

1. Connects to the WebSocket endpoint
2. Sends a randomly selected question
3. Receives streamed responses until completion
4. Waits between 5-15 seconds before asking another question

## Analyzing Results

Locust provides detailed statistics including:
- Response times (min, max, median, average)
- Requests per second
- Failure rates
- Number of users

Use these metrics to determine if your application can handle the target load of 100 concurrent users.

## Customization

You can modify the `sample_questions` list in the `ChatUser` class to use questions more specific to your application domain. 
