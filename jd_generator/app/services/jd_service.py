import logging
import os
import json
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

from app.models.models import BusinessData, JobDescription, PayRange

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logger = logging.getLogger(__name__)

class JDGeneratorService:
    """Service for generating job descriptions from employer data using OpenAI."""
    
    def __init__(self):
        """Initialize the JD Generator Service."""
        logger.info("Initializing OpenAI-powered JD Generator Service")
        
        # Default salary range if OpenAI doesn't provide one
        self.default_salary = {"min": 5000, "max": 10000}
        
    def generate_job_description(self, employer: BusinessData) -> JobDescription:
        """
        Generate a job description based on employer information using OpenAI.
        
        Args:
            employer: The employer business data
            
        Returns:
            JobDescription with all fields populated
        """
        logger.info(f"Generating job description for {employer.companyName} using OpenAI")
        
        # Create a prompt for OpenAI
        prompt = self._create_prompt(employer)
        
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Can be upgraded to GPT-4 for better results
                messages=[
                    {"role": "system", "content": "You are a skilled HR professional who creates detailed job descriptions based on employer information. Create a structured job description in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the job description from OpenAI's response
            jd_text = response.choices[0].message.content.strip()
            
            # Parse JSON from OpenAI's response
            try:
                jd_data = json.loads(jd_text)
                
                # Create and return the job description
                job_description = self._convert_to_job_description(jd_data)
                logger.info(f"Successfully generated JD with OpenAI for {employer.companyName}")
                return job_description
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from OpenAI response: {jd_text}")
                return self._create_fallback_job_description(employer)
                
        except Exception as e:
            logger.error(f"Error using OpenAI to generate job description: {str(e)}")
            return self._create_fallback_job_description(employer)
    
    def _create_prompt(self, employer: BusinessData) -> str:
        """Create a prompt for OpenAI based on employer data."""
        
        prompt = f"""
        Please create a detailed job description based on the following employer information:
        
        Company: {employer.companyName}
        Company Size: {employer.companySize}
        Number of Employees: {employer.noOfEmployees}
        Company Type: {employer.companyType}
        
        Job Category/Field: {', '.join(employer.categories) if employer.categories else 'Not specified'}
        
        Required Skills: {', '.join(employer.skills) if employer.skills else 'To be determined based on job needs'}
        
        Business Problems to Solve:
        {', '.join(employer.problems) if employer.problems else 'Not specified'}
        
        Business Data Types:
        {', '.join(employer.businessData) if employer.businessData else 'Not specified'}
        
        Challenges Faced:
        {', '.join(employer.challengesFaced) if employer.challengesFaced else 'Not specified'}
        
        Data Analysis Tools Used:
        {', '.join(employer.dataAnalysisTools) if employer.dataAnalysisTools else 'Not specified'}
        
        Expected Outcomes:
        {', '.join(employer.expectedOutcome) if employer.expectedOutcome else 'Not specified'}
        
        Please generate a complete job description in the following JSON format:
        {{
            "payRange": {{
                "min": [minimum salary as integer],
                "max": [maximum salary as integer]
            }},
            "fellowField": [main field name as string],
            "type": [array of 2-3 job types as strings],
            "skills": [array of 4-5 required skills as strings],
            "description": [job description text as string],
            "candidatesQualification": [qualifications text as string],
            "niceToHaves": [nice-to-have skills/experience text as string]
        }}
        
        Make sure all fields are present and properly formatted.
        """
        
        return prompt
    
    def _convert_to_job_description(self, jd_data: Dict[str, Any]) -> JobDescription:
        """Convert the OpenAI JSON response to a JobDescription object."""
        
        # Extract data with fallbacks
        pay_range_data = jd_data.get("payRange", self.default_salary)
        pay_range = PayRange(
            min=pay_range_data.get("min", 5000),
            max=pay_range_data.get("max", 10000)
        )
        
        fellow_field = jd_data.get("fellowField", "Data Analysis")
        
        # Ensure we have at least 1 job type
        job_types = jd_data.get("type", ["Data Analysis"])
        if not job_types or not isinstance(job_types, list):
            job_types = ["Data Analysis"]
            
        # Ensure we have at least 3 skills
        skills = jd_data.get("skills", ["Analytical Thinking", "Problem Solving", "Communication"])
        if not skills or not isinstance(skills, list) or len(skills) < 3:
            skills.extend(["Analytical Thinking", "Problem Solving", "Communication"])
            skills = list(set(skills))[:5]  # Ensure no duplicates and limit to 5
            
        description = jd_data.get("description", "Job description not provided.")
        qualifications = jd_data.get("candidatesQualification", "Qualifications not provided.")
        nice_to_haves = jd_data.get("niceToHaves", "Experience with relevant tools and technologies.")
        
        return JobDescription(
            payRange=pay_range,
            fellowField=fellow_field,
            type=job_types,
            skills=skills,
            description=description,
            candidatesQualification=qualifications,
            niceToHaves=nice_to_haves
        )
    
    def _create_fallback_job_description(self, employer: BusinessData) -> JobDescription:
        """Create a fallback job description if OpenAI fails."""
        logger.info(f"Using fallback JD generation for {employer.companyName}")
        
        # Use whatever information we have from the employer
        field = employer.categories[0] if employer.categories else "Data Analysis"
        skills = employer.skills if employer.skills and len(employer.skills) >= 3 else ["Analytical Thinking", "Problem Solving", "Communication", "Data Analysis"]
        
        description = f"We are looking for a skilled professional to help with data analysis and business challenges at {employer.companyName}."
        if employer.problems:
            description += f" Key problems to solve include {' and '.join(employer.problems).lower()}."
            
        qualifications = f"Experience in {field} or a related field. Strong technical and analytical skills."
        
        nice_to_haves = "Experience with data analysis tools and business intelligence platforms."
        if employer.dataAnalysisTools:
            nice_to_haves = f"Experience with {', '.join(employer.dataAnalysisTools)}."
        
        return JobDescription(
            payRange=PayRange(min=5000, max=10000),
            fellowField=field,
            type=["Data Analysis", "Business Intelligence"],
            skills=skills[:5],
            description=description,
            candidatesQualification=qualifications,
            niceToHaves=nice_to_haves
        ) 