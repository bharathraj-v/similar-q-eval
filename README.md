# Similar Questions Relevance Evaluation Framework

This repository contains APIs, evaluation scripts and evaluation results for the following 
- Evaluation of the relevance of similar questions 
- Solution Builder for solving academic problems with param to substantiate answer with it's similar questions
- Evaluation scripts to test the Similar questions Evaluation API and Solution Builder API with the test dataset
- Solution Builders comparision and evaluation - LLM-as-a-Judge


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

1. Solution Builder API (integrated)

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

**Findings:**

- The similarity evaluation scores are not correlated with the solution builder scores, meaning the solution builder with similar questions is able to ignore the irrelevant similar questions


## To-do

- [ ] Evaluate with MAS servers and check if findings correlate