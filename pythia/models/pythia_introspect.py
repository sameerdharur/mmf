# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F
import pdb
from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.embeddings import (ImageEmbedding, PreExtractedEmbedding,
                                       TextEmbedding)
from pythia.modules.encoders import ImageEncoder
from pythia.modules.layers import (ClassifierLayer, ModalCombineLayer,
                                   ReLUWithWeightNormFC)
from pythia.utils.configuration import ConfigNode
from scipy.spatial import distance
import numpy as np
import json


@registry.register_model("pythia_introspect")
class PythiaIntrospect(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    def build(self):
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embeddings(self, attr="text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        text_embeddings = []
        text_embeddings_list_config = self.config[attr]

        embeddings_out_dim = 0

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding.type
            embedding_kwargs = ConfigNode(text_embedding.params)

            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(embedding_type, **embedding_kwargs)

            text_embeddings.append(embedding)
            embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, nn.ModuleList(text_embeddings))

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args["model_data_dir"] = self.config["model_data_dir"]

    def _init_feature_encoders(self, attr):
        feat_encoders = []
        feat_encoders_list_config = self.config[attr + "_feature_encodings"]
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder["type"]
            encoder_kwargs = feat_encoder["params"]
            encoder_kwargs["model_data_dir"] = self.config["model_data_dir"]

            feat_model = ImageEncoder(encoder_type, feature_dim, **encoder_kwargs)

            feat_encoders.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_encoders", nn.ModuleList(feat_encoders))

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = len(
            getattr(self.config, "{}_feature_encodings".format(attr))
        )

        self.feature_embeddings_out_dim = 0

        for _ in range(num_feature_feat):
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = ImageEmbedding(
                    getattr(self, attr + "_feature_dim"),
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params
                )
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_embedding.out_dim

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        self.feature_embeddings_out_dim *= getattr(self, attr + "_feature_dim")

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1, attr2):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr]["type"],
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            **self.config[config_attr]["params"]
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def _init_classifier(self, combined_embedding_dim):
        # TODO: Later support multihead
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")

        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config):
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": combine_layer.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config["optimizer_attributes"]["params"]["lr"] * 0.1),
            },
        ]

        return params

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim

    def process_text_embedding(
        self, sample_list, embedding_attr="text_embeddings", info=None
    ):
        text_embeddings = []
        #pdb.set_trace()

        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        if not info:
            texts = getattr(sample_list, embedding_attr.split("_")[0])
        elif info == "sub_question":
            texts = getattr(sample_list, embedding_attr.split("_")[0] + '_sq')
        elif info == "other_question":
            texts = getattr(sample_list, embedding_attr.split("_")[0] + '_oq')

        # Get embedding models
        text_embedding_models = getattr(self, embedding_attr)

        for text_embedding_model in text_embedding_models:
            # TODO: Move this logic inside
            if isinstance(text_embedding_model, PreExtractedEmbedding):
                embedding = text_embedding_model(sample_list.question_id)
            else:
                embedding = text_embedding_model(texts)
            text_embeddings.append(embedding)

        text_embeddding_total = torch.cat(text_embeddings, dim=1)

        return text_embeddding_total

    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, extra=[], batch_size_t=None
    ):
        feature_embeddings = []
        feature_attentions = []
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(
                sample_list, "{}_feature_{:d}".format(attr, feature_idx), None
            )
            if feature is None:
                break
            feature_idx += 1
            feature = feature[:batch_size_t]
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final attended image features
        for i, feature in enumerate(features):
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, "{}_info_{:d}".format(attr, i), {})
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)
            #pdb.set_trace()

            # Get all of the feature embeddings
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]

            # Forward through these embeddings one by one
            for feature_embedding_model in feature_embedding_models:
                inp = (encoded_feature, text_embedding_total, feature_dim, extra)

                embedding, attention = feature_embedding_model(*inp)
                feature_embeddings.append(embedding)
                feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        layer_model = getattr(self, layer)
        joint_embeddings = layer_model(*feature_embeddings)
        if args[2] == "main":
            self.question_embedding = layer_model.question_embedding
        elif args[2] == "sub_question":
            self.question_embedding_sq = layer_model.question_embedding
        elif args[2] == "other_question":
            self.question_embedding_oq = layer_model.question_embedding
        #pdb.set_trace()
        #self.combine_layer = self.layer
        #joint_embedding = self.combine_layer(feature_embeddings)
        #pdb.set_trace()
        return joint_embeddings
        #return getattr(self, layer)(*feature_embeddings)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.classifier(joint_embedding)

    def compute_grad_cam(self, sample_list, model_output, question=None):
            #pdb.set_trace()
        #pdb.set_trace()
        if question == "main":
            #self.importance_vectors_reas = []
            scores = model_output['scores']
            classes = sample_list['gt_answer_index']
            classes_one_hot = torch.zeros_like(scores)
            classes_one_hot[range(classes_one_hot.shape[0]), classes] = 1
            #grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding, grad_outputs = classes_one_hot, create_graph=True)[0].to(self.device)
            grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding, grad_outputs = classes_one_hot, create_graph=True)[0]
            importance_vectors_cam = grads * self.joint_embedding
            #self.importance_vectors_reas.append(self.question_embedding)
            #pdb.set_trace()
            self.importance_vectors_reas = importance_vectors_cam
            model_output["importance_vectors_reas"] = importance_vectors_cam
            #self.importance_vectors_reas.append(torch.cat((importance_vectors_cam, self.question_embedding), 1))
        elif question == "sq":
            #self.importance_vectors_sq = []
            scores = model_output['scores_sq']
            classes = sample_list['gt_answer_index_sq']
            classes_one_hot = torch.zeros_like(scores)
            classes_one_hot[range(classes_one_hot.shape[0]), classes] = 1
            #grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding_sq, grad_outputs = classes_one_hot, create_graph=True)[0].to(self.device)
            grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding_sq, grad_outputs = classes_one_hot, create_graph=True)[0]
            importance_vectors_cam = grads * self.joint_embedding_sq
            #self.importance_vectors_sq.append(self.question_embedding_sq)
            self.importance_vectors_sq = importance_vectors_cam
            model_output["importance_vectors_sq"] = importance_vectors_cam
            #self.importance_vectors_sq.append(torch.cat((importance_vectors_cam, self.question_embedding_sq), 1))
        elif question == "oq":
            #self.importance_vectors_oq = []
            scores = model_output['scores_oq']
            classes = sample_list['gt_answer_index_oq']
            classes_one_hot = torch.zeros_like(scores)
            classes_one_hot[range(classes_one_hot.shape[0]), classes] = 1
            #grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding_oq, grad_outputs = classes_one_hot, create_graph=True)[0].to(self.device)
            grads = torch.autograd.grad(outputs = scores, inputs = self.joint_embedding_oq, grad_outputs = classes_one_hot, create_graph=True)[0]
            importance_vectors_cam = grads * self.joint_embedding_oq
            #self.importance_vectors_oq.append(self.question_embedding_oq)
            self.importance_vectors_oq = importance_vectors_cam
            model_output["importance_vectors_oq"] = importance_vectors_cam
            #self.importance_vectors_oq.append(torch.cat((importance_vectors_cam, self.question_embedding_oq), 1))

    def cosine_distance(self, vec_1, vec_2):
        batched_distance_vector = []
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(vec_1.shape[0]):
            norm_vec_1 = vec_1[i]/torch.max(vec_1[i])
            norm_vec_2 = vec_2[i]/torch.max(vec_2[i])
            distance = 1 - cos_similarity(norm_vec_1.unsqueeze(0), norm_vec_2.unsqueeze(0))
            batched_distance_vector.append(distance)
        return torch.cat(batched_distance_vector)

        
    def compute_distances(self, sample_list, model_output):
        model_output['distance_reas_sub'] = self.cosine_distance(self.importance_vectors_reas, self.importance_vectors_sq)
        model_output['distance_reas_other'] = self.cosine_distance(self.importance_vectors_reas, self.importance_vectors_oq)


    def forward(self, sample_list):
        #pdb.set_trace()
        if sample_list.dataset_name == 'train_vqa':
            #pdb.set_trace()
            sample_list.text = self.word_embedding(sample_list.text)
            text_embedding_total = self.process_text_embedding(sample_list)

            image_embedding_total, _ = self.process_feature_embedding(
                "image", sample_list, text_embedding_total
            )

            if self.inter_model is not None:
                image_embedding_total = self.inter_model(image_embedding_total)

            joint_embedding = self.combine_embeddings(
                ["image", "text"], [image_embedding_total, text_embedding_total], "main"
            )
        #pdb.set_trace()

            self.joint_embedding = joint_embedding

            model_output = {"scores": self.calculate_logits(joint_embedding)}

        # General stuff for distance computations

            model_output["scores_sq"] = model_output["scores"]
            model_output["scores_oq"] = model_output["scores"]
            model_output["distance_reas_sub"] = torch.zeros_like(model_output["scores"])
            model_output["distance_reas_other"] = torch.zeros_like(model_output["scores"])

        elif sample_list.dataset_name == 'train_introspect' or sample_list.dataset_name == 'test':

        # Compute the scores for the reasoning question
            sample_list.text = self.word_embedding(sample_list.text)
            text_embedding_total = self.process_text_embedding(sample_list)

            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )

            if self.inter_model is not None:
                image_embedding_total = self.inter_model(image_embedding_total)

            joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "main"
            )
        #pdb.set_trace()

            self.joint_embedding = joint_embedding

            model_output = {"scores": self.calculate_logits(joint_embedding)}

        # Compute the scores for the sub-question

            sample_list.text_sq = self.word_embedding(sample_list.text_sq)
            text_embedding_total = self.process_text_embedding(sample_list, info="sub_question")
            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )
            joint_embedding_sq = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "sub_question"
            )
            self.joint_embedding_sq = joint_embedding_sq
            model_output["scores_sq"] = self.calculate_logits(joint_embedding_sq)
        
        # Compute the scores for the other-question

            sample_list.text_oq = self.word_embedding(sample_list.text_oq)
            text_embedding_total = self.process_text_embedding(sample_list, info="other_question")
            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )
            joint_embedding_oq = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "other_question"
            )
            self.joint_embedding_oq = joint_embedding_oq
            model_output["scores_oq"] = self.calculate_logits(joint_embedding_oq)

        # General stuff for distance computations
            self.compute_grad_cam(sample_list, model_output, question="main")
            self.compute_grad_cam(sample_list, model_output, question="sq")
            self.compute_grad_cam(sample_list, model_output, question="oq")

            self.compute_distances(sample_list, model_output)

        else:

            sample_list.text = self.word_embedding(sample_list.text)
            text_embedding_total = self.process_text_embedding(sample_list)

            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )

            if self.inter_model is not None:
                image_embedding_total = self.inter_model(image_embedding_total)

            joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "main"
            )
        #pdb.set_trace()

            self.joint_embedding = joint_embedding

            model_output = {"scores": self.calculate_logits(joint_embedding)}

        # Compute the scores for the sub-question

            sample_list.text_sq = self.word_embedding(sample_list.text_sq)
            text_embedding_total = self.process_text_embedding(sample_list, info="sub_question")
            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )
            joint_embedding_sq = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "sub_question"
            )
            self.joint_embedding_sq = joint_embedding_sq
            model_output["scores_sq"] = self.calculate_logits(joint_embedding_sq)
        
        # Compute the scores for the other-question

            sample_list.text_oq = self.word_embedding(sample_list.text_oq)
            text_embedding_total = self.process_text_embedding(sample_list, info="other_question")
            image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
            )
            joint_embedding_oq = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total], "other_question"
            )
            self.joint_embedding_oq = joint_embedding_oq
            model_output["scores_oq"] = self.calculate_logits(joint_embedding_oq)

        # General stuff for distance computations
            self.compute_grad_cam(sample_list, model_output, question="main")
            self.compute_grad_cam(sample_list, model_output, question="sq")
            self.compute_grad_cam(sample_list, model_output, question="oq")

            self.compute_distances(sample_list, model_output)


        #self.compute_grad_cam()
        #pdb.set_trace()

        #image_embedding_total, _ = self.process_feature_embedding(
        #    "image", sample_list, text_embedding_total
        #)

        #if self.inter_model is not None:
        #    image_embedding_total = self.inter_model(image_embedding_total)

        #joint_embedding = self.combine_embeddings(
        #    ["image", "text"], [image_embedding_total, text_embedding_total]
        #)

        #self.joint_embedding = joint_embedding

        #model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output


# TODO: Update
@registry.register_model("pythia_question_only")
class PythiaQuestionOnly(PythiaIntrospect):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        fa_txt = self.image_text_multi_modal_combine_layer.module.fa_txt
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_txt(text_embedding_total))

        linear_text = self.classifier.module.linear_text
        f_o_text = self.classifier.module.f_o_text
        scores = linear_text(f_o_text(joint_embedding))

        model_output = {"scores": scores}

        return model_output


# TODO: Update
@registry.register_model("pythia_image_only")
class PythiaImageOnly(PythiaIntrospect):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        fa_image = self.image_text_multi_modal_combine_layer.module.fa_image
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_image(image_embedding_total))

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output
