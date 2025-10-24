## 🔧Evaluate Paper2Web

### Evaluation Pipeline

**Generate QA pairs:**
```bash
python create_paper_questions.py --paper_folder "path/to/paper" --model_name o3
```

**Run comprehensive evaluation:**
```bash
python run_all_evaluations.py --paper_name "Paper Title" --base_dir "path/to/papers" --judge_version "v2" --auto_generate_qa
```

**Individual evaluation metrics:**
```bash
python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "path/to/papers" --judge_version "path/to/dictionary/of/code" --metric informative_judge
```
```bash
python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "path/to/papers" --judge_version "path/to/dictionary/of/code" --metric qa
```
```bash
python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "path/to/papers" --judge_version "path/to/dictionary/of/code" --metric aesthetic_judge
```

### Supported Evaluation Metrics

- `informative_judge`: Information quality assessment
- `aesthetic_judge`: Visual design evaluation
- `qa`: Question-answering accuracy
- `completeness_llm`: Content completeness
- `connectivity_llm`: Navigation structure
- `interactivity_judge`: Interactive features

### Automatic Evaluation
```bash
# Generate QA pairs for evaluation
python create_paper_questions.py --paper_folder "path/to/paper"
```

```bash
# Run all evaluations
python run_all_evaluations.py --paper_name "Paper Title" --base_dir "path/to/papers" --judge_version "path/to/dictionary/of/code" --auto_generate_qa
```

