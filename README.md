# Similar Questions Evaluation Framework

This repository contains APIs, evaluation scripts and evaluation results for the following 
- Evaluation of the relevance of similar questions 
- Solution Builder for solving academic problems with param to substantiate answer with it's similar questions
- Evaluation scripts to test the Similar questions Evaluation API and Solution Builder API with the test dataset
- Solution comparision and evaluation


Repository Index

├── configs
│   ├── comparitive_analysis_prompts_mas.yaml - Prompts for comparitive analysis of solutions via a multi-agent system
│   ├── comparitive_analysis_prompts.yaml - Prompts for comparitive analysis of solutions via a single agent system
│   ├── evaluation_prompts_mas.yaml - Prompts for evaluation of similar questions via a multi-agent system
│   ├── evaluation_prompts_s.yaml - Prompts for evaluation of similar questions via a single agent system
│   ├── solution_builder_prompts_sim.yaml - Prompts for solution builder with similar questions via a single agent system
│   └── solution_builder_prompts.yaml - Prompts for solution builder without similar questions via a single agent system
├── eval_data_exploration.ipynb
├── evals
│   ├── evaluation.log
│   ├── evaluation_results.csv - Solution comparision results for the test dataset (without MAS)
│   ├── evaluation_results_sample.csv 
│   ├── similar_questions_evaluation.log
│   └── similar_questions_evaluation_results.csv - Solution comparision results for the test dataset (without MAS)
├── main_logic.ipynb - Notebook where most of the logic was workshopped before implementing as APIs
├── README.md
├── requirements.txt
├── scripts
│   ├── evaluate_similar_questions.py - Script for executing similar questions evaluation API on the test dataset
│   └── evaluate_solution_builders.py - Script for executing solution builder API on the test dataset
├── servers
│   ├── compare_answers_mas.py - API for comparing solutions generated with and without similar questions via a multi-agent system
│   ├── compare_answers.py - API for comparing solutions generated with and without similar questions via a single agent system
│   ├── evaluate_similar_question_mas.py - API for evaluating similar questions via a multi-agent system
│   ├── evaluate_similar_question.py - API for evaluating similar questions via a single agent system
│   ├── solution_builder_no_sim.py - API for solution builder without similar questions 
│   ├── solution_builder.py - an integrated API for solution builder with and without similar questions based on param
│   └── solution_builder_with_sim.py - API for solution builder with similar questions
└── similar_question_data.json


## Installation and usage

Installation:

```
git clone https://github.com/bharathraj-v/similar-q-eval
cd similar-q-eval
pip install -r requirements.txt
```

Usage:

1. Solution Builder API (integrated)

Run the server via
```
cd similar-q-eval/
PORT=8000 python servers/solution_builder.py
```

Server Usage

python```
import requests

BASE_URL = "http://localhost:8000"

response1 = requests.post(
    f"{BASE_URL}/solve",
    json={
        "question": "ENTER_QUESTION_HERE",
        "use_similar_questions": False
    }
)

print("=== STANDARD AGENT ===")
print(f"Status: {response1.status_code}")
print(f"Agent used: {response1.json()['agent_used']}")
print(f"Answer: {response1.json()['data']['final_answer']}")
print()

# 2. Similar questions agent
response2 = requests.post(
    f"{BASE_URL}/solve", 
    json={
        "question": "ENTER_QUESTION_HERE",
        "use_similar_questions": True
    }
)

print("=== SIMILAR QUESTIONS AGENT ===")
print(f"Status: {response2.status_code}")
print(f"Agent used: {response2.json()['agent_used']}")
print(f"Answer: {response2.json()['data']['final_answer']}")
```

2. Similar Questions Evaluation API

Run the server via
```
cd similar-q-eval/
PORT=8001 python servers/evaluate_similar_question.py
```

Server Usage

python```
import requests

BASE_URL = "http://localhost:8001"

payload = {
    "question_text": "ENTER_QUESTION_HERE",
    "similar_question": "ENTER_SIMILAR_QUESTION_HERE",
    "summarized_solution_approach": "ENTER_SUMMARIZED_SOLUTION_APPROACH_HERE"
}

response = requests.post(
    f"{BASE_URL}/evaluate",
    json=payload
)

print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))
```

3. Evaluation Scripts

Run the scripts via the following commands while the servers are up and running
```
cd similar-q-eval/
python scripts/evaluate_similar_questions.py
python scripts/evaluate_solution_builders.py
```


## Evaluation Results and Findings


| Metric | Score |
|--------|-------|
| Avg. Similar Questions Evaluation | **59 / 100** |
| Avg. Solution Builder (with similar questions) | **96 / 100** |
| Avg. Solution Builder (without similar questions) | **95 / 100** |

**Correlations:**

| Correlation Pair | Value |
|------------------|-------|
| Similarity vs. Solution Builder (no sim) | **0.03** |
| Similarity vs. Solution Builder (with sim) | **0.05** |

Findings:

- The similarity evaluation scores are not correlated with the solution builder scores meaning the LLM with reasoning is able to ignore the irrelevant similar questions


## To-do

- [ ] Evaluate with MAS servers and check if findings correlate