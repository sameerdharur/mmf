The code for this project is in three parts :

# 1. VQA experiments in Pythia.

- The setup instructions on this page are to be followed to setup Pythia, download the VQAv2 dataset and the Pythia model : https://github.com/facebookresearch/mmf/tree/master/projects/pythia.

- This now goes by the name of the MMF framework. More documentation relevant to the setup process of the code can be found here : https://mmf.readthedocs.io/en/latest/

- The VQA Introspect dataset is accessible here : https://www.microsoft.com/en-us/research/project/vqa-introspect/

- As a sample, we have attached a subset of the train and validation splits of the data which can be used for these experiments.

- The locations to the data can be specified in ```pythia/common/defaults/configs/tasks/vqa/vqa_introspect.yml```

- The hyperparameters for the model can be changed in ```configs/vqa/vqa2/pythia_introspect.yml```

- To train a model (with the default hyperparameters), run the following command :

```bash 
python tools/run.py --tasks vqa --datasets vqa_introspect --model pythia_introspect --config configs/vqa/vqa2/pythia_introspect.yml --resume_file data/models/pythia.pth
```

- To evaluate a trained model, run the following command replacing ```pythia.pth``` with the name of the model to be evaluated :

```bash
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia.yml --resume_file data/models/pythia.pth --run_type val
```
# 2. Ranking and Similarity analysis.

- The validation script could be executed with the trained model to generate the Grad-CAM vectors for each of the three types of questions - reasoning, sub-questions and irrelevant questions. These are then written to a CSV file. 

- Once these CSV files are generated, run the Jupyter notebook ```similarity_ranking.ipynb``` which contains a step-by-step walkthrough of the code to compute similarity vectors and the ranking metrics.

# 3. Visual grounding analysis.
