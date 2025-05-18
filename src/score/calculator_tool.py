import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScoreCalculatorInput(BaseModel):
    scores_json: str = Field(description="JSON string containing scores data to calculate averages from")


@tool("calculate_average_scores", args_schema=ScoreCalculatorInput,
      description=("Calculate average scores from provided scores data. "
                   "The scores data must be provided in JSON format. "
                   "The JSON string should contain an array of scores with fields: subject_name, subject_credits and score_over_rall."
                   "Please input data only in this JSON format and has no other words:"
                   "scores: array of: subject_name: ,subject_credits: ,score_over_rall:"))
def calculate_average_scores(scores_json: str) -> str:
    """
    Calculate average scores from provided scores data.

    Args:
        scores_json: JSON string containing scores data

    Returns:
        A JSON string containing the calculated average scores
    """
    try:
        # Parse the scores data
        data = json.loads(scores_json)

        if "scores" not in data or not data["scores"]:
            return json.dumps({"averages": {}, "message": "No scores data provided or scores array is empty"})

        scores = data["scores"]

        # Calculate averages
        total_score = 0
        for score in scores:
            if "score_over_rall" in score and isinstance(score["score_over_rall"], (int, float)):
                total_score += score["score_over_rall"]
            else:
                continue
        total_credits = 0
        for score in scores:
            if "subject_credits" in score and isinstance(score["subject_credits"], (int, float)):
                total_credits += score["subject_credits"]
            else:
                continue

        if total_credits == 0:
            return json.dumps({"averages": {}, "message": "Total credits is zero, cannot calculate average"})
        average_score = total_score / total_credits
        average_score = round(average_score, 2)  # Round to 2 decimal places
        print("__Calculate average score__")
        print(average_score)
        print(total_credits)


        averages = {"average_score": average_score, "total_credits": total_credits}

        return json.dumps({"averages": averages, "message": "Average scores calculated successfully"})

    except Exception as e:
        logger.error(e)
        return json.dumps({"averages": {}, "message": f"Error calculating averages because: {str(e)}"})
