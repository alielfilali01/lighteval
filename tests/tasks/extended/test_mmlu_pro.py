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

from lighteval.tasks.extended.mmlu_pro.main import mmlu_pro_prompt 
# We import the specific prompt function directly for testing

def test_mmlu_pro_prompt_formatting():
    """Test MMLU-Pro prompt formatting to ensure it handles 10 choices correctly."""
    sample_input = {
        "question_id": "test_id_123",
        "question": "What is the primary ingredient in traditional Japanese miso soup?",
        "options": [
            "Tofu", "Seaweed", "Miso paste", "Rice", "Mushrooms",
            "Soy sauce", "Bonito flakes", "Green onions", "Dashi stock", "Ginger"
        ],
        "answer": "C", # Corresponds to "Miso paste"
        "answer_index": 2, # 0-indexed
        "cot_content": "Some explanation here...",
        "category": "food_and_drink",
        "src": "synthetic_example"
    }

    # Get the prompt function (already imported)
    # For MMLU-Pro, we defined mmlu_pro_prompt directly in main.py
    prompt_fn = mmlu_pro_prompt

    doc = prompt_fn(sample_input, task_name="mmlu_pro_test")

    expected_query = """\
Question: What is the primary ingredient in traditional Japanese miso soup?
 A. Tofu
 B. Seaweed
 C. Miso paste
 D. Rice
 E. Mushrooms
 F. Soy sauce
 G. Bonito flakes
 H. Green onions
 I. Dashi stock
 J. Ginger
Answer:"""  # Note: No trailing space after "Answer:" based on standard MCFFormulation

    assert doc.query.strip() == expected_query.strip() # Use strip to avoid issues with trailing newlines/spaces
    
    expected_choices = [
        " A", " B", " C", " D", " E", 
        " F", " G", " H", " I", " J"
    ]
    assert doc.choices == expected_choices
    assert doc.gold_index == 2
    assert doc.task_name == "mmlu_pro_test"
    assert doc.specific["category"] == "food_and_drink"
    assert doc.specific["question_id"] == "test_id_123"
