# Similar Questions Relevance Evaluation Framework

This repository contains APIs, evaluation scripts and evaluation results for the following: 
- Evaluation of the relevance of similar questions 
- Solution Builder for solving academic problems with param to substantiate answer with it's similar questions
- Evaluation scripts to test the Similar questions Evaluation API and Solution Builder API with the test dataset
- Solution Builders comparision and evaluation - **LLM-as-a-Judge**

# Repository Index
### Tree
```
├── configs
│   ├── comparitive_analysis_prompts_mas.yaml
│   ├── comparitive_analysis_prompts.yaml
│   ├── evaluation_prompts_mas.yaml
│   ├── evaluation_prompts_s.yaml
│   ├── solution_builder_prompts_sim.yaml
│   └── solution_builder_prompts.yaml
├── eval_data_exploration.ipynb
├── evals
│   ├── evaluation.log
│   ├── evaluation_results.csv
│   ├── evaluation_results_sample.csv
│   ├── similar_questions_evaluation.log
│   └── similar_questions_evaluation_results.csv
├── main_logic.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── evaluate_similar_questions.py
│   └── evaluate_solution_builders.py
├── servers
│   ├── compare_answers_mas.py
│   ├── compare_answers.py
│   ├── evaluate_similar_question_mas.py
│   ├── evaluate_similar_question.py
│   ├── solution_builder_no_sim.py
│   ├── solution_builder.py
│   └── solution_builder_with_sim.py
└── similar_question_data.json
```

### Details

- **configs/**
  - `comparitive_analysis_prompts_mas.yaml` – Prompts for comparative analysis via a multi-agent system  
  - `comparitive_analysis_prompts.yaml` – Prompts for comparative analysis via a single agent system  
  - `evaluation_prompts_mas.yaml` – Prompts for evaluating similar questions via a multi-agent system  
  - `evaluation_prompts_s.yaml` – Prompts for evaluating similar questions via a single agent system  
  - `solution_builder_prompts_sim.yaml` – Prompts for solution builder with similar questions (single agent)  
  - `solution_builder_prompts.yaml` – Prompts for solution builder without similar questions (single agent)  

- **eval_data_exploration.ipynb** – Notebook for dataset exploration  

- **evals/**
  - `evaluation.log` – Logs of evaluation runs  
  - `evaluation_results.csv` – Solution comparison results (without MAS)  
  - `evaluation_results_sample.csv` – Sample results  
  - `similar_questions_evaluation.log` – Logs for similar questions evaluation  
  - `similar_questions_evaluation_results.csv` – Similar questions evaluation results (without MAS)  

- **main_logic.ipynb** – Notebook where core logic was first prototyped  

- **scripts/**
  - `evaluate_similar_questions.py` – Runs similar questions evaluation API on dataset  
  - `evaluate_solution_builders.py` – Runs solution builder evaluation on dataset  

- **servers/**
  - `compare_answers_mas.py` – Compare solutions with/without similar questions (MAS)  
  - `compare_answers.py` – Compare solutions with/without similar questions (single agent)  
  - `evaluate_similar_question_mas.py` – Evaluate similar questions via MAS  
  - `evaluate_similar_question.py` – Evaluate similar questions (single agent)  
  - `solution_builder_no_sim.py` – Solution builder without similar questions  
  - `solution_builder.py` – Integrated solution builder (param-based)  
  - `solution_builder_with_sim.py` – Solution builder with similar questions  

- **similar_question_data.json** – Dataset of similar questions  


## Installation and usage

**Installation:**

```
git clone https://github.com/bharathraj-v/similar-q-eval
cd similar-q-eval
pip install -r requirements.txt
```

**Usage:**

### 1. Solution Builder API (integrated)

Run the server via
```
cd similar-q-eval/
PORT=8000 python servers/solution_builder.py
```

Server Usage


```
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

### 2. Similar Questions Evaluation API

Run the server via
```
cd similar-q-eval/
PORT=8001 python servers/evaluate_similar_question.py
```
(Similarly, evaluate_similar_question_mas.py can be used for the multi-agent system)

Server Usage

```
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

### 3. Comparitive Analysis Scripts

Run the server via

```
cd similar-q-eval/
PORT=8002 python servers/compare_answers.py
```
(Similarly, compare_answers_mas.py can be used for the multi-agent system)

Server Usage

```
import requests
import json

URL = "http://127.0.0.1:8003/analyse"

# Example question
payload = {
    "question": "A car accelerates from rest to a speed of 20 m/s in 5 seconds. What is its acceleration?",
    "sim_answer_explanation": "Using the formula v = u + at, where u=0, v=20, t=5. So, 20 = 0 + a*5. This gives a = 4 m/s^2.",
    "sim_answer_final_answer": "4 m/s^2",
    "no_sim_answer_explanation": "Acceleration is the change in velocity over time. a = (final velocity - initial velocity) / time. a = (20 - 0) / 5. Therefore, a = 4 m/s^2.",
    "no_sim_answer_final_answer": "The acceleration is 4 meters per second squared."
}

response = requests.post(URL, json=payload)

print(json.dumps(response.json(), indent=2))
```

(Similarly, compare_answers_mas.py can be used for the multi-agent system but with just the question as answers are auto-generated by agent)

### 4. Evaluation Scripts

Run the scripts via the following commands while the servers are up and running
```
cd similar-q-eval/
python scripts/evaluate_similar_questions.py
python scripts/evaluate_solution_builders.py
```


## Evaluation Results and Findings


| Metric | Avg. Score |
|--------|-------|
| Similar Questions Evaluation | **59 / 100** |
| Solution Builder (with similar questions) | **96 / 100** |
| Solution Builder (without similar questions) | **95 / 100** |

**Correlations:**

| Correlation Pair | Value |
|------------------|-------|
| Similarity vs. Solution Builder (no sim) | **0.03** |
| Similarity vs. Solution Builder (with sim) | **0.05** |

**Findings:**

- The similarity evaluation scores are not correlated with the solution builder scores, meaning the solution builder with similar questions is able to ignore the irrelevant similar questions



**To Note**
- The similar questions are currently only being fetched from the dataset. Current API has modularity in regards to changing the similar question fetching logic with vector search.
- Both solution builders have been evaluated with thinking budget set to -1 (unlimited), evaluration results may vary with no thinking.


## To-do

- [ ] Evaluate with MAS servers and check if findings correlate
