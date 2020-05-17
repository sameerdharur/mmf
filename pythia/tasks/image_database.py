# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import json
import pdb

class ImageDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in Pythia
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """

    def __init__(self, imdb_path):
        super().__init__()
        self._load_imdb(imdb_path)

    def _load_imdb(self, imdb_path):
        if imdb_path.endswith(".npy"):
            self._load_npy(imdb_path)
        elif imdb_path.endswith(".jsonl"):
            self._load_jsonl(imdb_path)
        else:
            raise ValueError("Unknown file format for imdb")

    def _load_jsonl(self, imdb_path):
        with open(imdb_path, "r") as f:
            db = f.readlines()
            for idx, line in enumerate(db):
                db[idx] = json.loads(line.strip("\n"))
            self.data = db
            self.start_idx = 0

    def _load_npy(self, imdb_path):
        self.db = np.load(imdb_path, allow_pickle=True)
        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get("metadata", {})
            self.data = self.db.get("data", [])
            print("Yes dict")
        else:
            # TODO: Deprecate support for this
            self.metadata = {"version": 1}
            self.data = self.db
            # Handle old imdb support
            if "image_id" not in self.data[0]:
                self.start_idx = 1

        if len(self.data) == 0:
            self.data = self.db

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]

        # Hacks for older IMDBs
        #pdb.set_trace()
        if "answers" not in data:
            #print("Answers in data")
            if "all_answers" in data and "valid_answers" not in data:
                data["answers"] = data["all_answers"]
            if "valid_answers" in data:
                data["answers"] = data["valid_answers"]
            if "main_answer_str" in data:
                data["answers"] = data["main_answer_str"]
            if "sub_answer_str" in data:
                data["answers_sq"] = data["sub_answer_str"]
            if "other_answer_str" in data:
                data["answers_oq"] = data["other_answer_str"]

        # TODO: Later clean up VizWIz IMDB from copy tokens
        if "answers" in data and data["answers"][-1] == "<copy>":
            #print("Answers not in data")
            data["answers"] = data["answers"][:-1]
        
        #pdb.set_trace()
        #print("Data : {}".format(data))

        return data

    def get_version(self):
        return self.metadata.get("version", None)
