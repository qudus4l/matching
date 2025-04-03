# Unified Matching API

This API combines job description generation and AI-based profile matching into a single service.

## Features

- **Job Description Generation**: Create detailed job descriptions based on employer information
- **AI Profile Matching**: Match user profiles to job descriptions with detailed scoring and reasoning
- **Embedding-based Semantic Matching**: Uses advanced AI embeddings for accurate matching
- **Improvement Suggestions**: Provides actionable suggestions for improving profile matches

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the API

Start the API with:

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation: http://localhost:8000/docs

## API Endpoints

### Job Description Generation

- `POST /api/v1/generate-jd`: Generate a job description from employer data

### Profile Matching

- `POST /api/v1/ai-match`: Match a user profile to a job (simplified result)
- `POST /api/v1/ai-match/detailed`: Match a user profile to a job (detailed result)

## Development

Run tests with:

```
pytest
``` 