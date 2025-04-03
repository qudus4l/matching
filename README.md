# AI-Powered Job Matching API

A streamlined API for matching user profiles with job requirements using AI techniques.

## Features

- **AI-Powered Matching**: Uses embeddings to understand the semantic relationship between skills and requirements
- **Multi-Dimensional Analysis**: Evaluates candidates across skills, experience, and field relevance
- **Simple API Response**: Returns just the essential match information:
  - Match percentage
  - Categories that matched
  - Categories that didn't match

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd matching
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file if needed.

## Running the API

Start the API with:
```bash
uvicorn ai_matching.app.main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoint

### POST /api/v1/ai-match

Match a user profile against a job problem description using AI.

**Input:**
```json
{
  "user_data": {
    "status": "OK",
    "user": {
      // User profile data
    }
  },
  "problem": {
    // Job problem description
  }
}
```

**Output:**
```json
{
  "percentage_match": 75,
  "what_matched": ["Skills", "Field"],
  "went_against": ["Experience"]
}
``` 