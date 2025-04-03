import logging
from fastapi import APIRouter, Depends, HTTPException

from app.models.models import MatchInput, AIMatchResult, SimpleMatchResult
from app.services.ai_matching_service import AIMatchingService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["ai-matching"]
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
        logger.info(f"Received AI match request for user: {data.user_data.user.firstName} {data.user_data.user.lastName}")
        logger.info(f"Job field: {data.problem.fellowField}")
        
        # Get user and problem from input data
        user = data.user_data.user
        problem = data.problem
        
        # Initialize AI matching service and calculate match
        matching_service = AIMatchingService(use_openai_completion=use_openai_completion)
        detailed_result = matching_service.calculate_match(user, problem)
        
        # Convert to simple result with filtering
        simple_result = convert_to_simple_result(detailed_result)
        
        logger.info(f"AI Match result: {simple_result.percentage_match}% - Matched: {simple_result.what_matched}, Against: {simple_result.went_against}")
        return simple_result
    except Exception as e:
        logger.error(f"Error processing AI match: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing match: {str(e)}") 