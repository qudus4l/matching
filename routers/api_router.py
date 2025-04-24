import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from models.models import (
    EmployerResponse, JobDescriptionResponse, MatchInput, 
    SimpleMatchResult, AIMatchResult
)
from services.unified_service import UnifiedService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["unified-matching-api"]
)

def convert_to_simple_result(result: AIMatchResult) -> SimpleMatchResult:
    """
    Convert detailed AIMatchResult to SimpleMatchResult with only basic fields.
    Filter out internal categories like 'Semantic' and 'Concepts'.
    """
    # Define valid categories to show to users
    valid_categories = ["Skills", "Field", "Experience", "Education", "Job Type", "Qualifications"]
    
    # Filter what_matched and went_against to only include valid categories
    filtered_matched = [cat for cat in result.what_matched if cat in valid_categories]
    filtered_against = [cat for cat in result.went_against if cat in valid_categories]
    
    return SimpleMatchResult(
        percentage_match=result.percentage_match,
        what_matched=filtered_matched,
        went_against=filtered_against
    )

@router.post("/generate-jd", response_model=JobDescriptionResponse)
async def generate_job_description(data: EmployerResponse):
    """
    Generate multiple job descriptions based on employer information.
    
    Takes employer profile data and creates four structured job descriptions with:
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
        
        # Initialize unified service
        unified_service = UnifiedService()
        
        # Generate job descriptions
        job_descriptions = unified_service.generate_job_description(employer)
        
        # Log and return result
        logger.info(f"Generated {len(job_descriptions)} JDs for employer: {employer.companyName}")
        return JobDescriptionResponse(problems=job_descriptions)
    except Exception as e:
        logger.error(f"Error generating job descriptions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating job descriptions: {str(e)}")

@router.post("/ai-match", response_model=SimpleMatchResult)
async def ai_match_user_to_problem(data: MatchInput, use_openai_completion: bool = False):
    """
    Match a user's profile to a job problem description using AI.
    
    Returns a simplified match result with just the essentials:
    - percentage_match
    - what_matched (filtered to only user-facing categories)
    - went_against (filtered to only user-facing categories)
    
    Args:
        data: The user and problem data
        use_openai_completion: Whether to use OpenAI completions for recommendations
    """
    try:
        # Log the request with the automatically fixed ID
        logger.info(f"Received AI match request for user: {data.user_data.user.firstName} {data.user_data.user.lastName}")
        logger.info(f"User ID (possibly auto-generated): {data.user_data.user._id}")
        logger.info(f"Job field: {data.problem.fellowField}")
        
        # Get user and problem from input data
        user = data.user_data.user
        problem = data.problem
        
        # Initialize unified service and calculate match
        unified_service = UnifiedService(use_openai_completion=use_openai_completion)
        detailed_result = unified_service.calculate_match(user, problem)
        
        # Convert to simple result with filtering
        simple_result = convert_to_simple_result(detailed_result)
        
        logger.info(f"AI Match result: {simple_result.percentage_match}% - Matched: {simple_result.what_matched}, Against: {simple_result.went_against}")
        return simple_result
    except Exception as e:
        logger.error(f"Error processing AI match: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing match: {str(e)}")

@router.post("/ai-match/detailed", response_model=AIMatchResult)
async def ai_match_user_to_problem_detailed(data: MatchInput, use_openai_completion: bool = False):
    """
    Match a user's profile to a job problem description using AI.
    
    Returns a detailed match result with comprehensive information:
    - percentage_match
    - what_matched
    - went_against
    - detailed_reasoning
    - suggested_improvements
    - semantic_similarity
    - compatibility_by_area
    
    Args:
        data: The user and problem data
        use_openai_completion: Whether to use OpenAI completions for recommendations
    """
    try:
        # Log the request with the automatically fixed ID
        logger.info(f"Received detailed AI match request for user: {data.user_data.user.firstName} {data.user_data.user.lastName}")
        logger.info(f"User ID (possibly auto-generated): {data.user_data.user._id}")
        logger.info(f"Job field: {data.problem.fellowField}")
        
        # Get user and problem from input data
        user = data.user_data.user
        problem = data.problem
        
        # Initialize unified service and calculate match
        unified_service = UnifiedService(use_openai_completion=use_openai_completion)
        detailed_result = unified_service.calculate_match(user, problem)
        
        logger.info(f"Detailed AI Match result: {detailed_result.percentage_match}%")
        return detailed_result
    except Exception as e:
        logger.error(f"Error processing detailed AI match: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing detailed match: {str(e)}") 