# JD Generator API

An intelligent API service that automatically generates job descriptions based on employer information using OpenAI.

![JD Generator System](https://mermaid.ink/img/pako:eNqFkk9PgzAYxr9Kw0WzLJqZm0c82ZgQD17cmYN7KfAOGigNZYyZ8d3doDo3D8aGQPo-v_58aEYyrhGJkOe0NLRATLxV8GysrdjCDmQN-7vJY-qAqQB_AaqyAhSPmxlVPBUzTLL0jjYd3iuY2_K1poYOZHEfB6H_edG_bNvz5WN_t7yYLx_9sBE0_cPX-K16ANEhZJozdxnhz7wfxp1uS_ZLkD-XuhK2cWaXbQzPa5KOHnbD9vb6OLl_V7bmDHME6F4bbavUVQ3X3S6OHFGdQ7yrG4TelzkhOZeGFPOPK0Kwhpi2Q5UWN7XLqpW8RMP98XhknXL1a7JYm1LQKzmnRDQ3F4rKv95y1RqwBWJpnUiuSrmo0DHVTFPQYQ0v-kNExknKK5KRd8QL8vmXQ9tsvzrMNlI?type=png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The JD Generator API transforms employer information into comprehensive, tailored job descriptions. It leverages OpenAI to generate professional job descriptions that match company needs, business challenges, and required skills.

## Features

- **AI-Powered JD Creation**: Uses OpenAI to generate intelligent, contextually appropriate job descriptions
- **Natural Language Processing**: Creates professional-quality job descriptions tailored to the company's needs
- **Structured Output**: Generates complete job descriptions with consistent formatting
- **Contextual Understanding**: Incorporates business problems and challenges into relevant job requirements
- **Fallback Mechanisms**: Includes rule-based fallbacks if AI generation fails

## Architecture

The system employs a layered architecture to generate comprehensive job descriptions:

![Architecture Diagram](https://mermaid.ink/img/pako:eNqVk09v2zAMxb9KoEsLOHGaNTuEOQX9g2FoB2wreurlYFsZDMSWIUlZsyD57iM7O02LFcMuNvTj43tPpI9e0zcSvNTV2tVkXWqbebbGtoVb0A94vE02QwJcBfQLoL6vgZSlXRhVqfCAEn5pJa0JTCdH1WZ0JVVEb-zlsUbLSQlsxPZt1mTZp9VyNdncbd99vl_fbD_cbdKM_PiHzJB5fdOUOd0nX04vdr9gW6rO8ApeMTALowF3Zey-yPNDkc9uN_dXc-RfwdUxxeW3QNHvN5J139DFdkv2L8V1YpwOZF5P4Mz7Ic0y_xvHZLMh80qDr-p6Y_vNEhyVljrvx36cYnEpNVPW7Ssmcm5Z0kPP6lRkFnXjZ4Yra2qakHWfhZsndv50bq-Nqx95bRpyV9RUxM4ri8o0-DvFJEtjnZuYZaPqUk4pWA1e9R-PiJQ9S1aZxLx-LngV4zYm1tVRiUuXmRjgbWNr-tvCXPCu_ObwC7DI0l0?type=png)

### Components

1. **API Layer**:
   - Receives employer information
   - Validates input data
   - Returns formatted job descriptions

2. **OpenAI Integration**:
   - Crafts detailed prompts based on employer data
   - Processes GPT responses into structured job descriptions
   - Handles error cases and response validation

3. **Fallback System**:
   - Provides rule-based JD generation when OpenAI is unavailable
   - Ensures service reliability even when external services fail

4. **Output Formatting**:
   - Ensures consistent JSON structure
   - Validates all required fields are present

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- An OpenAI API key
- A virtual environment tool like venv or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd jd_generator
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

4. Set up environment variables:

Edit the `.env` file to add your OpenAI API key.

## Configuration

The application can be configured through the `.env` file:

```
# API Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run
```
uvicorn app.main:app --reload
```

## API Documentation

### Endpoints

#### `POST /api/v1/generate-jd`

Generates a job description based on employer information.

**Request**

```json
{
  "status": "OK",
  "user": {
    "_id": "67d760d5994d53ce79626804",
    "email": "employer@company.com",
    "verified": true,
    "active": true,
    "skills": [
      "Structured Thinking",
      "SQL & Databases",
      "Business Acumen",
      "ETL"
    ],
    "categories": [
      "Quantitative Analyst (Quant)"
    ],
    "onboarded": true,
    "profileCompletion": 16.666666666666664,
    "userType": "employer",
    "companyName": "Data Fellows",
    "fullName": "Jane Smith",
    "problems": [
      "Inaccurate financial risk assessments.",
      "Difficulty in modeling complex financial instruments."
    ],
    "companySize": "5-9",
    "companyType": "Sole Proprietorship",
    "noOfEmployees": 75,
    "businessData": [
      "Sales",
      "Customer"
    ],
    "challengesFaced": [
      "Lack of expertise",
      "High cost of analytics tools"
    ],
    "dataAnalysisTools": [
      "Excel/Google Sheets",
      "Business Intelligence Software"
    ],
    "expectedOutcome": [
      "Improve sales performance",
      "Better understand customer behavior"
    ]
  }
}
```

**Response**

```json
{
  "problem": {
    "payRange": {
      "min": 5000,
      "max": 10000
    },
    "fellowField": "Quantitative Analysis",
    "type": [
      "Financial Modeling",
      "Risk Assessment"
    ],
    "skills": [
      "Structured Thinking",
      "SQL & Databases",
      "Business Acumen",
      "ETL",
      "Data Analysis"
    ],
    "description": "We need a solution to address inaccurate financial risk assessments and difficulties in modeling complex financial instruments. Our focus areas include sales and customer data. Our goals are to improve sales performance and better understand customer behavior.",
    "candidatesQualification": "Experience in quantitative analysis with focus on financial modeling and risk assessment. Strong skills in Structured Thinking, SQL & Databases, Business Acumen, and ETL.",
    "niceToHaves": "Experience with Excel/Google Sheets and Business Intelligence Software."
  }
}
```

### Response Structure

| Field | Type | Description |
|-------|------|-------------|
| problem | Object | Container for the job description |
| problem.payRange | Object | Salary range with min and max values |
| problem.fellowField | String | Primary field/category for the job |
| problem.type | Array | List of job types/specialties (2-3 items) |
| problem.skills | Array | Required skills (4-5 items) |
| problem.description | String | Detailed job description text |
| problem.candidatesQualification | String | Required qualifications text |
| problem.niceToHaves | String | Preferred qualifications text |

### Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error, possibly with OpenAI

## Common Workflows

### 1. Basic JD Generation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as JD Generator API
    participant Service as JD Service
    participant OpenAI as OpenAI API
    
    Client->>API: POST /api/v1/generate-jd
    API->>Service: Process employer data
    Service->>Service: Create OpenAI prompt
    Service->>OpenAI: Send prompt
    OpenAI-->>Service: Return generated text
    Service->>Service: Parse and validate response
    Service-->>API: Return job description
    API-->>Client: Return formatted response
```

### 2. Fallback Flow (when OpenAI fails)

```mermaid
sequenceDiagram
    participant Client
    participant API as JD Generator API
    participant Service as JD Service
    participant OpenAI as OpenAI API
    participant Fallback as Fallback System
    
    Client->>API: POST /api/v1/generate-jd
    API->>Service: Process employer data
    Service->>Service: Create OpenAI prompt
    Service->>OpenAI: Send prompt
    OpenAI--xService: API Error
    Service->>Fallback: Generate fallback JD
    Fallback-->>Service: Return basic JD
    Service-->>API: Return job description
    API-->>Client: Return formatted response
```

### 3. End-to-End Example: Creating a Job Post

1. Employer completes their profile with company details, business challenges, and skills needed
2. System sends a request to the JD Generator API with this information
3. API generates a complete job description using OpenAI
4. The generated JD is returned and can be used in job postings
5. Employer can review and post the job description

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Error: `500 Internal Server Error` with OpenAI-related message
   - Solution: Check your API key and OpenAI service status

2. **Missing Required Fields**
   - Error: Invalid response format
   - Solution: Ensure all required employer fields are included

3. **Poor Quality Descriptions**
   - Issue: Generated descriptions lack detail or relevance
   - Solution: Consider upgrading to GPT-4 by changing the model in .env

### Debugging

To enable debug logs, set `DEBUG=True` in your `.env` file.