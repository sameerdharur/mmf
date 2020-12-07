# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import tqdm
import pdb

from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.tasks.image_database import ImageDatabase
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.general import get_pythia_root


class VQAIntrospectDataset(BaseDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("vqa_introspect", dataset_type, config)
        imdb_files = self.config.imdb_files
        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.dataset = dataset_type

        if dataset_type == 'train' or dataset_type == 'val':
            if 'imdb_train2014.npy' in self.imdb_file or 'imdb_val2014.npy' in self.imdb_file or 'imdb_vqa2014_val_reasoning_questions' in self.imdb_file:
                self.dataset = 'train_vqa'
            elif 'train_introspect.npy' in self.imdb_file or 'train_binary.npy' in self.imdb_file:
                self.dataset = 'train_introspect'

        self.imdb = ImageDatabase(self.imdb_file)

        self.kwargs = kwargs
        self.image_depth_first = self.config.image_depth_first
        self._should_fast_read = self.config.fast_read

        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

        self._use_features = False
        if hasattr(self.config, "image_features"):
            self._use_features = True
            self.features_max_len = self.config.features_max_len
            self._return_info = self.config.get("return_info", True)

            all_image_feature_dirs = self.config.image_features[dataset_type]
            curr_image_features_dir = all_image_feature_dirs[imdb_file_index]
            curr_image_features_dir = curr_image_features_dir.split(",")
            curr_image_features_dir = self._get_absolute_path(curr_image_features_dir)

            self.features_db = FeaturesDataset(
                "coco",
                directories=curr_image_features_dir,
                depth_first=self.image_depth_first,
                max_features=self.features_max_len,
                fast_read=self._should_fast_read,
                imdb=self.imdb,
                return_info=self._return_info,
            )

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def __len__(self):
        return len(self.imdb)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self._name, self._dataset_type
                )
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.imdb)), miniters=100, disable=not is_main_process()
            ):
                self.cache[idx] = self.load_item(idx)

    def get_item(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()
        current_sample.dataset_name = self.dataset

        if self.dataset == 'train_vqa':
            
            text_processor_argument = {"tokens": sample_info["question_tokens"]}
            processed_question = self.text_processor(text_processor_argument)
            current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
            )
            current_sample.text = processed_question["text"]
            current_sample.question_text = sample_info["question_str"]
            current_sample.text_sq = current_sample.text
            current_sample.text_oq = current_sample.text
            current_sample.reasoning_question = sample_info["question_str"]
            current_sample.reasoning_answer = sample_info["answers"][0]
            current_sample.sub_question = sample_info["question_str"]
            current_sample.other_question = sample_info["question_str"]

        elif self.dataset == 'train_introspect' or self.dataset == 'test':
           
            text_processor_argument = {"text": sample_info["main_question_str"]}
            processed_question = self.text_processor(text_processor_argument)
            current_sample.text = processed_question["text"]
            if "sub_question_str" in sample_info:
                text_processor_argument_sq = {"text": sample_info["sub_question_str"]}
                processed_question_sq = self.text_processor(text_processor_argument_sq)
                current_sample.text_sq = processed_question_sq["text"]
                
            if "other_question_str" in sample_info:
                text_processor_argument_oq = {"text": sample_info["other_question_str"]}
                processed_question_oq = self.text_processor(text_processor_argument_oq)
                current_sample.text_oq = processed_question_oq["text"]
                
            current_sample.question_text = sample_info["main_question_str"]
            current_sample.reasoning_question = sample_info["main_question_str"]
            current_sample.reasoning_answer = sample_info["main_answer_str"][0]
            current_sample.sub_question = sample_info["sub_question_str"]
            current_sample.other_question = sample_info["other_question_str"]
            current_sample.text_len = torch.tensor(
            len(sample_info["main_question_tokens"]), dtype=torch.int
            )

        else:
            
            text_processor_argument = {"text": sample_info["question_str"]}
            processed_question = self.text_processor(text_processor_argument)
            current_sample.text = processed_question["text"]
            if "sub_question_str" in sample_info:
                text_processor_argument_sq = {"text": sample_info["sub_question_str"]}
                processed_question_sq = self.text_processor(text_processor_argument_sq)
                current_sample.text_sq = processed_question_sq["text"]
                
            if "other_question_str" in sample_info:
                text_processor_argument_oq = {"text": sample_info["other_question_str"]}
                processed_question_oq = self.text_processor(text_processor_argument_oq)
                current_sample.text_oq = processed_question_oq["text"]
            else:
                current_sample.text_oq = current_sample.text_sq
                
            current_sample.question_text = sample_info["question_str"]
            current_sample.reasoning_question = sample_info["question_str"]
            current_sample.reasoning_answer = sample_info["answers"][0]
            current_sample.sub_question = sample_info["sub_question_str"]
            current_sample.other_question = sample_info["sub_question_str"]
            current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int)

    
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def add_ocr_details(self, sample_info, sample):
        if self.use_ocr:
            # Preprocess OCR tokens
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
            # Get embeddings for tokens
            context = self.context_processor({"tokens": ocr_tokens})
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({"info": sample_info["ocr_info"]})[
                "bbox"
            ]

        return sample

    def add_answer_info(self, sample_info, sample):

        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]
            sample.gt_answer_index = processed_soft_copy_answers["answers_indices"][0]

        if "answers_sq" in sample_info:
            answers = sample_info["answers_sq"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.answers_sq = processed_soft_copy_answers["answers"]
            sample.targets_sq = processed_soft_copy_answers["answers_scores"]
            sample.gt_answer_index_sq = processed_soft_copy_answers["answers_indices"][0]
        elif "sub_answers" in sample_info:
            answers = sample_info["sub_answers"]
            answer_processor_arg = {"answers": answers}
            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.answers_sq = processed_soft_copy_answers["answers"]
            sample.targets_sq = processed_soft_copy_answers["answers_scores"]
            sample.gt_answer_index_sq = processed_soft_copy_answers["answers_indices"][0]
        else:
            sample.answers_sq = sample.answers
            sample.targets_sq = sample.targets
            sample.gt_answer_index_sq = sample.gt_answer_index

        if "answers_oq" in sample_info:
            answers = sample_info["answers_oq"]
            answer_processor_arg = {"answers": answers}
            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.answers_oq = processed_soft_copy_answers["answers"]
            sample.targets_oq = processed_soft_copy_answers["answers_scores"]
            sample.gt_answer_index_oq = processed_soft_copy_answers["answers_indices"][0]
        else:
            sample.answers_oq = sample.answers_sq
            sample.targets_oq = sample.targets_sq
            sample.gt_answer_index_oq = sample.gt_answer_index_sq

        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == self.context_processor.PAD_TOKEN:
                answer = "unanswerable"

            predictions.append({"question_id": question_id.item(), "answer": answer})

        return predictions
