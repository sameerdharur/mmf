Code for the paper ```SOrTing VQA Models : Contrastive Gradient Learning for Improved Consistency``` - https://arxiv.org/abs/2010.10038

If you use this code, please consider citing the paper as:

```
@article{dharur2020sort,
  title={SOrT-ing VQA Models: Contrastive Gradient Learning for Improved Consistency},
  author={Dharur, Sameer and Tendulkar, Purva and Batra, Dhruv and Parikh, Devi and Selvaraju, Ramprasaath R},
  journal={arXiv preprint arXiv:2010.10038},
  year={2020}
}
```

Replicating the experiments of the paper broadly involves two parts :

# 1. VQA experiments in Pythia.

- The setup instructions on this page are to be followed to setup Pythia, download the VQAv2 dataset and the Pythia model : https://github.com/facebookresearch/mmf/tree/master/projects/pythia.

- This now goes by the name of the MMF framework. More documentation relevant to the setup process of the code can be found here : https://mmf.readthedocs.io/en/latest/

- The VQA Introspect dataset is accessible here : https://www.microsoft.com/en-us/research/project/vqa-introspect/

- For our experiments, we used the dataset in the ```.npy``` format whose converted files can be found [here](https://drive.google.com/file/d/1PItvD8FkQoLAgEwItSwQ0Fzo0YbxvXs5/view?usp=sharing). Copy the train and validation files from here into the same folder containing the VQA dataset files (i.e, ```imdb/vqa```) installed from the above setup. 

- Alternatively, the locations to the data can be specified in ```pythia/common/defaults/configs/tasks/vqa/vqa_introspect.yml```

- The hyperparameters for the model can be changed in ```configs/vqa/vqa2/pythia_introspect.yml```

- To train a model (with the default hyperparameters), run the following command :

```bash 
python tools/run.py --tasks vqa --datasets vqa_introspect --model pythia_introspect --config configs/vqa/vqa2/pythia_introspect.yml --resume_file data/models/pythia.pth
```

- To evaluate a trained model, run the following command replacing ```pythia.pth``` with the name of the model to be evaluated :

```bash
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia.yml --resume_file data/models/pythia.pth --run_type val
```

- The final model reported in the paper can be downloaded [here](https://drive.google.com/file/d/18SCI6CvOVlvLlevxvka_vlQAeTKqGqao/view?usp=sharing).

# 2. Ranking and Similarity analysis.

- The validation script could be executed with the trained model to generate the Grad-CAM vectors for each of the three types of questions - reasoning, sub-questions and irrelevant questions. These are then written to a CSV file. 

- Once these CSV files are generated, run the Jupyter notebook ```similarity_ranking.ipynb``` which contains a step-by-step walkthrough of the code to compute similarity vectors and the ranking metrics.
