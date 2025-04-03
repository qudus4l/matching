import logging
from fastapi import APIRouter, Depends, HTTPException

from app.models.models import EmployerResponse, JobDescriptionResponse, JobDescription
from app.services.jd_service import JDGeneratorService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["jd-generator"]
)

@router.post("/generate-jd", response_model=JobDescriptionResponse)
async def generate_job_description(data: EmployerResponse):
    """
    Generate a job description based on employer information.
    
    Takes employer profile data and creates a structured job description with:
    - Appropriate salary range
    - Required field
    - Job types
    - Necessary skills
    - Description of the job
    - Required qualifications
    - Nice-to-have skills/experience
    
    Args:
        data: The employer data containing company and business information
    """
    try:
        logger.info(f"Received JD generation request for employer: {data.user.companyName}")
        
        # Get employer data
        employer = data.user
        
        # Initialize JD generator service
        jd_service = JDGeneratorService()
        
        # Generate job description
        job_description = jd_service.generate_job_description(employer)
        
        # Log and return result
        logger.info(f"Generated JD for field: {job_description.fellowField}, skills: {job_description.skills}")
        return JobDescriptionResponse(problem=job_description)
    except Exception as e:
        logger.error(f"Error generating job description: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating job description: {str(e)}") 