# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.tasks.vqa.vizwiz import VizWizBuilder
from pythia.tasks.vqa.vqa_introspect.ocr_dataset import VQAIntrospectOCRDataset


@Registry.register_builder("vqa_introspect_ocr")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "VQAIntrospect_OCR"
        self.set_dataset_class(VQAIntrospectOCRDataset)
