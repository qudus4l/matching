# AI API

A streamlined API for Problem generation and matching user profiles with job requirements using AI techniques.

## Features

- **Job Description Generation**: AI-powered generation of job descriptions based on employer data
- **AI-Powered Matching**: Uses embeddings to understand the semantic relationship between skills and requirements
- **Multi-Dimensional Analysis**: Evaluates candidates across skills, experience, and field relevance
- **Simple API Response**: Returns essential match information including match percentage and category details

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│                 │     │                   │     │                  │
│  Client Apps    │◄────┤  Unified API      │◄────┤  OpenAI API      │
│  (Frontend)     │     │  (FastAPI)        │     │  (GPT & Ada)     │
│                 │     │                   │     │                  │
└─────────────────┘     └───────────────────┘     └──────────────────┘
                               │
                          ┌────┴───────┐
                          │            │
                          ▼            ▼
               ┌─────────────────┐  ┌──────────────────┐
               │                 │  │                  │
               │ Job Description │  │ Profile Matching │
               │ Generation      │  │ Algorithm        │
               │                 │  │                  │
               └─────────────────┘  └──────────────────┘
                                          │
                                     ┌────┴───────┐
                                     │            │
                                     ▼            ▼
                          ┌─────────────────┐  ┌──────────────────┐
                          │                 │  │                  │
                          │ Embedding       │  │ Local Fallback   │
                          │ Service (OpenAI)│  │ (SentenceTransf.)│
                          │                 │  │                  │
                          └─────────────────┘  └──────────────────┘
```

## System Design

The system uses a combination of NLP techniques to achieve accurate job matching:

1. **Text Embeddings**: Converts user skills and job requirements into vector representations
2. **Semantic Matching**: Determines similarity between skill descriptions using cosine similarity
3. **Multi-factor Analysis**: Evaluates match across different categories (skills, field, experience)
4. **Fallback Mechanisms**: Uses local models when OpenAI is unavailable

## Project Structure

```
.
├── models/                # Data models
│   └── models.py          # Pydantic models for API requests/responses
├── routers/               # API route definitions
│   └── api_router.py      # Endpoint definitions
├── services/              # Business logic
│   ├── embedding_service.py  # Text embedding services
│   └── unified_service.py    # Job matching logic
├── .env                   # Environment variables (OpenAI API key)
├── main.py                # FastAPI application setup
├── run.py                 # Application entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

1.
```bash
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
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

5. Run the application:
```bash
python run.py
```

The API will be available at http://localhost:8000.

## API Documentation

### Base Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page for the API |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Swagger/OpenAPI documentation |

### Problems Generation

#### `POST /api/v1/generate-jd`

Generates a job description based on employer information using OpenAI.

**Request Body:**
```json
{
  "user": {
    "companyName": "TechSolutions Inc.",
    "companySize": "Medium",
    "noOfEmployees": 75,
    "companyType": "Software Development",
    "categories": ["Data Science", "Machine Learning"],
    "skills": ["Python", "TensorFlow", "Data Analysis", "SQL"],
    "problems": ["Inefficient data processing", "Manual data classification"],
    "businessData": ["Customer behavior data", "Transaction logs", "Product usage metrics"],
    "challengesFaced": ["Data silos", "Scaling ML models"],
    "dataAnalysisTools": ["Tableau", "Jupyter Notebooks", "Pandas"],
    "expectedOutcome": ["Automated classification system", "Improved user recommendations"]
  }
}
```

**Response:**
```json
{
  "problem": {
    "payRange": {
      "min": 80000,
      "max": 120000
    },
    "fellowField": "Data Science",
    "type": ["Machine Learning Engineer", "Data Scientist", "AI Developer"],
    "skills": ["Python", "TensorFlow", "SQL", "Machine Learning", "Data Analysis"],
    "description": "TechSolutions Inc. is seeking a talented Machine Learning Engineer to join our growing team. The successful candidate will develop and implement machine learning models to solve complex data processing challenges and automate classification systems...",
    "candidatesQualification": "Bachelor's degree in Computer Science, Data Science, or related field. Minimum 3 years of experience with Python, TensorFlow, and building production-ready ML models...",
    "niceToHaves": "Experience with cloud ML platforms (AWS SageMaker, Azure ML), knowledge of CI/CD pipelines for ML systems, and experience with real-time data processing systems..."
  }
}
```

### Profile Matching

#### `POST /api/v1/ai-match`

Matches a user profile against a job description using AI. Returns a simplified result.

**Request Body:**
```json
{
  "user_data": {
    "status": "OK",
    "user": {
      "_id": "user123",
      "firstName": "John",
      "lastName": "Doe",
      "email": "john.doe@example.com",
      "verified": true,
      "active": true,
      "skills": ["Python", "Data Analysis", "SQL", "Pandas"],
      "categories": ["Data Science"],
      "specialities": ["Data Visualization"],
      "categoriesAndSpecialitiesAdded": true,
      "workExperience": [
        {
          "_id": "exp1",
          "title": "Data Analyst",
          "company": "Analytics Corp",
          "location": "Remote",
          "startDate": "2020-01-15T00:00:00.000Z",
          "endDate": "2022-06-30T00:00:00.000Z",
          "description": "Performed data analysis and created visualizations for business stakeholders"
        }
      ],
      "educationHistory": [
        {
          "_id": "edu1",
          "school": "University of Technology",
          "degree": "Bachelor's",
          "fieldOfStudy": "Computer Science",
          "startDate": "2016-09-01T00:00:00.000Z",
          "endDate": "2020-05-30T00:00:00.000Z",
          "description": "Focused on data science and machine learning"
        }
      ],
      "skillsAdded": true,
      "workExperienceAdded": true,
      "educationHistoryAdded": true,
      "bio": "Data professional with 2+ years of experience in analytics and visualization",
      "bioAdded": true,
      "onboarded": true,
      "profileCompletion": 95,
      "userType": "jobseeker"
    }
  },
  "problem": {
    "payRange": {
      "min": 80000,
      "max": 120000
    },
    "fellowField": "Data Science",
    "type": ["Machine Learning Engineer", "Data Scientist"],
    "skills": ["Python", "TensorFlow", "SQL", "Machine Learning", "Data Analysis"],
    "description": "We are seeking a talented Data Scientist to build machine learning models...",
    "candidatesQualification": "Bachelor's degree in Computer Science or related field, 3+ years of experience...",
    "niceToHaves": "Experience with cloud platforms, knowledge of deep learning frameworks..."
  }
}
```

**Response:**
```json
{
  "percentage_match": 72,
  "what_matched": ["Skills", "Field"],
  "went_against": ["Experience"]
}
```

#### `POST /api/v1/ai-match/detailed`

Returns detailed matching information with reasoning and suggestions.

**Request Body:** Same as `/api/v1/ai-match`

**Response:**
```json
{
  "percentage_match": 72,
  "what_matched": ["Skills", "Field"],
  "went_against": ["Experience"],
  "detailed_reasoning": [
    {
      "feature": "Skills",
      "score": 0.85,
      "explanation": "The candidate has 4 out of 5 required skills (Python, SQL, Data Analysis)",
      "details": {
        "matching_skills": ["Python", "SQL", "Data Analysis"],
        "missing_skills": ["TensorFlow", "Machine Learning"]
      }
    },
    {
      "feature": "Field",
      "score": 0.95,
      "explanation": "The candidate's field matches the job's required field perfectly",
      "details": {
        "user_fields": ["Data Science"],
        "job_field": "Data Science",
        "specialities_bonus": true
      }
    },
    {
      "feature": "Experience",
      "score": 0.45,
      "explanation": "The job requires 3+ years experience but the candidate has only 2 years",
      "details": {
        "required_years": 3,
        "candidate_years": 2,
        "title_match_score": 0.7
      }
    }
  ],
  "suggested_improvements": [
    "Gain experience with TensorFlow and Machine Learning frameworks",
    "Consider pursuing additional experience in a Data Scientist role"
  ],
  "semantic_similarity": 0.78,
  "compatibility_by_area": {
    "Skills": 0.85,
    "Field": 0.95,
    "Experience": 0.45,
    "Education": 0.75,
    "Job Type": 0.8
  }
}
```

## Common Workflows

### 1. Generate Job Description & Find Matching Candidates

```mermaid
sequenceDiagram
    participant Employer
    participant API
    participant OpenAI
    participant DatabaseService
    
    Employer->>API: POST /api/v1/generate-jd (company info)
    API->>OpenAI: Create job description
    OpenAI->>API: Return structured JD
    API->>Employer: Return job description
    
    Employer->>API: Save & search for matches
    API->>DatabaseService: Find matching candidates
    
    loop For each candidate
        API->>API: POST /api/v1/ai-match (candidate profile + JD)
    end
    
    API->>Employer: Return ranked candidate list
```

### 2. Assess Candidate-Job Compatibility

```mermaid
sequenceDiagram
    participant JobSeeker
    participant API
    participant EmbeddingService
    
    JobSeeker->>API: POST /api/v1/ai-match (profile + job)
    API->>EmbeddingService: Generate embeddings
    EmbeddingService->>API: Return embeddings
    
    API->>API: Calculate similarity scores
    API->>API: Evaluate multi-factor match
    API->>JobSeeker: Return match result
    
    JobSeeker->>API: POST /api/v1/ai-match/detailed
    API->>API: Generate detailed reasoning
    API->>API: Create improvement suggestions
    API->>JobSeeker: Return detailed match analysis
```

### 3. Fallback Flow When OpenAI Is Unavailable

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant OpenAI
    participant LocalService
    
    Client->>API: Request matching/generation
    API->>OpenAI: Attempt OpenAI request
    OpenAI--xAPI: Connection error
    
    API->>LocalService: Fall back to local model
    LocalService->>LocalService: Generate embeddings
    LocalService->>API: Return local results
    API->>Client: Return response (with disclaimer)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and completions | Yes |
| `PORT` | Port for the FastAPI server (default: 8000) | No |
| `LOG_LEVEL` | Logging level (default: INFO) | No |

## Acknowledgements

This API uses the following open-source libraries:
- FastAPI: High-performance web framework
- Pydantic: Data validation and settings management
- OpenAI API: Embeddings and completions
- SentenceTransformers: Local embedding model
- Scikit-learn: For similarity calculations and TF-IDF

# AI Matching Algorithm with LLM Verification

## Overview

The AI matching algorithm evaluates the compatibility between a candidate's profile and a job description by analyzing multiple factors and providing a percentage match score along with insights into which areas matched or didn't match. The final score can be verified by an LLM in extreme cases.

## Matching Components

The algorithm analyzes five key categories:

| Category | Weight | Description | Match Threshold |
|----------|--------|-------------|----------------|
| Skills | 25% | Evaluates direct and semantic skill matches | 0.5 (50%) |
| Field | 15% | Assesses compatibility between user's field and job field | 0.6 (60%) |
| Experience | 20% | Analyzes work experience duration and relevance | 0.5 (50%) |
| Semantic | 25% | Measures overall semantic similarity between profile and job | 0.7 (70%) |
| Concepts | 15% | Identifies shared key concepts using TF-IDF analysis | 0.7 (70%) |

## Calculation Process

1. **Individual Category Scoring**:
   - Each category produces a score between 0.0 and 1.0
   - Different thresholds determine if a category is considered "matched"

2. **Weighted Average**:
   - Raw score = (Skills × 0.25) + (Field × 0.15) + (Experience × 0.20) + (Semantic × 0.25) + (Concepts × 0.15)
   - Raw percentage = Raw score × 100

3. **Visible Category Scaling**:
   - Visible categories: Skills, Field, Experience (shown to users)
   - Hidden categories: Semantic, Concepts (used in calculation but not shown)
   - Visible match ratio = Number of matched visible categories ÷ 3
   - Scaled percentage = Raw percentage × (0.3 + 0.7 × Visible match ratio)

4. **Special Case Handling**:
   - If no visible categories match: Maximum score capped at 20%
   - Percentage is rounded to nearest integer and capped at 100%

5. **LLM Verification** (Final Step):
   - An LLM analyzes the full profile, job description, and algorithmic results
   - Can override the match percentage in extreme cases where the algorithm clearly misrepresents the match
   - Only activated when extremely confident in the need for an override
   - Provides reasoning for any override decision

## Category Details

### Skills Match (25%)
- Direct matching: Exact skill name matches
- Semantic matching: Similar skills with similarity > 0.8
- Overall score = (0.7 × Direct match ratio) + (0.3 × Semantic match contribution)
- Threshold: 0.5 (50%)

### Field Match (15%)
- Compares job's required field with user's fields
- Evaluates exact matches and semantic similarity
- Threshold: 0.6 (60%)

### Experience Match (20%)
- Calculates total years of work experience
- Identifies specific year requirements in job qualifications
- Scores based on ratio of actual years to required years
- Threshold: 0.5 (50%)

### Semantic Similarity (25%)
- Creates text representations of user profile and job description
- Calculates vector similarity using embeddings
- Threshold: 0.7 (70%)

### Concept Match (15%)
- Extracts key terms from both profile and job using TF-IDF
- Identifies common concepts between the two
- Score = Number of common concepts ÷ Number of job concepts
- Threshold: 0.7 (70%)

## LLM Verification Details

The LLM verification acts as a safety mechanism to catch edge cases where algorithmic matching might miss important signals. This step:

1. Analyzes the full user profile and job description
2. Reviews all calculated scores and matched/unmatched categories
3. Assesses if the algorithmic score reasonably represents the match quality
4. Only suggests an override if there's a significant mismatch between the calculated score and human judgment
5. Provides a specific reason for any override decision

The LLM is instructed to be conservative with overrides and only make changes in clear cases where:
- A candidate with excellent qualifications is scored too low
- A candidate with clearly insufficient qualifications is scored too high

## API Response

The API returns:
- **percentage_match**: Final calculated match percentage (after potential LLM verification)
- **what_matched**: List of visible categories that met their thresholds
- **went_against**: List of visible categories that didn't meet their thresholds

In the detailed version, it also returns:
- **detailed_reasoning**: Score and explanation for each category
- **suggested_improvements**: Suggestions to improve match percentage
- **semantic_similarity**: Raw semantic similarity score
- **compatibility_by_area**: Scores for all categories

## Examples

1. **Strong Match (>70%)**:
   - Multiple skills match directly
   - Experience meets or exceeds requirements
   - Field aligns well with job requirements
   - High semantic similarity

2. **Medium Match (40-70%)**:
   - Some skills match, others don't
   - Experience may be slightly below requirements
   - Moderate semantic similarity

3. **Poor Match (<40%)**:
   - Few or no matching skills
   - Experience well below requirements or missing
   - Low semantic similarity
   - If no visible categories match, score capped at 20%

4. **LLM Override Examples**:
   - Algorithm: 35%, LLM: 65% - Candidate has unique qualifications not captured by standard categories
   - Algorithm: 60%, LLM: 30% - Candidate has major disqualifying factors not captured by the algorithm

This matching system prioritizes visible factors that users can directly improve while still leveraging semantic and conceptual analysis for nuanced matching, with LLM verification as a final quality check in extreme cases.
