The code for this project is in three parts :

# 1. VQA experiments in Pythia.

- The setup instructions on this page are to be followed to setup Pythia and download the VQAv2 dataset : https://github.com/facebookresearch/mmf/tree/master/projects/pythia.

- This now goes by the name of the MMF framework.

With or without the changes to these files, the script can be executed by running the following command :

```bash
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia.yml --resume_file data/models/pythia.pth --run_type val
```
