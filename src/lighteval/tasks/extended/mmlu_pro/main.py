# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language

# Configuration for mapping dataset columns to prompt function arguments
MAP_EXAMPLE_TO_PROMPT = {
    "question": "question",  # The column in MMLU-Pro dataset containing the question
    "choices": "options",  # The column with the list of choices
    "gold_idx": "answer_index",  # The column with the index of the correct answer
}

# Create the prompt formatting function for MMLU-Pro
mmlu_pro_prompt_fn = get_mcq_prompt_function(
    Language.ENGLISH,  # MMLU-Pro is in English
    MAP_EXAMPLE_TO_PROMPT,
    MCFFormulation(),  # Standard multiple-choice format (A, B, C, ...)
)

def mmlu_pro_prompt(line, task_name: str = "mmlu_pro"):
    """
    Processes a single line from the MMLU-Pro dataset and formats it
    into a Doc object for evaluation.
    """
    # The get_mcq_prompt_function already creates a function that returns a Doc,
    # but we might want to add specific logic here if needed in the future,
    # or directly use the created function.
    # For now, this wrapper ensures the task_name is correctly passed.
    # It also allows for easy modification or addition of specific fields later.
    
    # Correctly call the generated prompt function
    doc_from_template = mmlu_pro_prompt_fn(line, task_name=task_name)

    # Example of adding or modifying specific fields if necessary:
    # doc_from_template.specific_data = line.get("category") # If you want to store category
    
    return Doc(
        task_name=task_name,
        query=doc_from_template.query,
        choices=doc_from_template.choices,
        gold_index=doc_from_template.gold_index,
        instruction=doc_from_template.instruction,
        target_for_metric=doc_from_template.target_for_metric,
        specific={
            "category": line.get("category"),
            "src": line.get("src"),
            "question_id": line.get("question_id"),
            # Add other MMLU-Pro specific fields if needed for analysis
        }
    )

# Define the LightevalTaskConfig for MMLU-Pro
mmlu_pro_task = LightevalTaskConfig(
    name="mmlu_pro",
    prompt_function=mmlu_pro_prompt,
    suite=["extended"],
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],  # "test" for main data, "validation" for the small val set
    evaluation_splits=["test"], # Evaluate on the main test data
    few_shots_split="validation", # Use validation for few-shot examples
    few_shots_select="random",
    metric=[Metrics.loglikelihood_acc], # Loglikelihood accuracy for multiple choice
    generation_size=-1, # Not strictly generative, but CoT might produce long outputs
    stop_sequence=None, # Default stop sequences are usually fine for MCQ
)

# TASKS_TABLE must be defined in this file for lighteval to discover the task
TASKS_TABLE = [mmlu_pro_task]
