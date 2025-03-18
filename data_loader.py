from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import collections
import pickle
import os
from tqdm import tqdm
from copy import deepcopy
import json


class DataCenter():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.preprocess_data()
        self.sample_evid_and_ref()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.evidence_setting = args.evidence_setting
        self.language_model = args.language_model
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.max_text_length = args.max_text_length
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references

    def load_data(self):

        with open('../data/' + self.dataset_name + '/claims.json', 'r') as file:
            claims_list = json.load(file)
            self.claims = {}
            self.claims = {claim['claim_id']: claim for claim in claims_list}
        with open('../data/' + self.dataset_name + '/evidence.json', 'r') as file:
            evidences_list = json.load(file)
            self.evidences = {}
            self.evidences = {evidence['evid_id']: evidence for evidence in evidences_list}

        self.has_contexts = False
        if os.path.exists('../data/' + self.dataset_name + '/contexts.json'):
            self.has_contexts = True
            with open('../data/' + self.dataset_name + '/contexts.json', 'r') as file:
                contexts_list = json.load(file)
                self.contexts = {}
                self.contexts = {context['ctx_id']: context for context in contexts_list}

        self.has_references = False
        if os.path.exists('../data/' + self.dataset_name + '/references.json'):
            self.has_references = True
            with open('../data/' + self.dataset_name + '/references.json', 'r') as file:
                references_list = json.load(file)
                self.references = {}
                self.references = {references['ref_id']: references for references in references_list}

        self.num_claims, self.num_evidences = len(self.claims), len(self.evidences)
        if self.has_contexts:
            self.num_contexts = len(self.contexts)
        if self.has_references:
            self.num_references = len(self.references)

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        if self.dataset_name == 'check_covid' and self.evidence_setting == 'retrieved':
            claims_new = {}
            for claim_id in self.claims.keys():
                if self.claims[claim_id]['label'] != 'nei':
                    claims_new[len(claims_new)] = deepcopy(self.claims[claim_id])
            self.claims = claims_new

    def split_data(self):

        self.training_claim_ids, self.dev_claim_ids, self.test_claim_ids = [], [], []
        if 'train_dev_test' in self.claims[0]:
            for claim_id in self.claims.keys():
                if self.claims[claim_id]['train_dev_test'] == 'train':
                    self.training_claim_ids.append(claim_id)
                elif self.claims[claim_id]['train_dev_test'] == 'dev':
                    self.dev_claim_ids.append(claim_id)
                else:
                    self.test_claim_ids.append(claim_id)
        else:
            self.training_claim_ids = np.arange(int(self.num_claims * 0.72))
            self.dev_claim_ids = np.arange(len(self.training_claim_ids), int(self.num_claims * 0.8))
            self.test_claim_ids = np.arange(len(self.training_claim_ids) + len(self.dev_claim_ids), self.num_claims)
        self.training_claim_ids, self.dev_claim_ids, self.test_claim_ids = np.array(self.training_claim_ids), np.array(self.dev_claim_ids), np.array(self.test_claim_ids)

        if len(self.test_claim_ids) == 0:  # this means the dataset doesn't have test set but only dev set
            self.test_claim_ids = self.dev_claim_ids
        self.num_training_claims, self.num_test_claims = len(self.training_claim_ids), len(self.test_claim_ids)

        self.label_names = np.unique([self.claims[claim_id]['label'] for claim_id in self.claims.keys()])
        self.label_name2label_id = {label_name: label_id for label_id, label_name in enumerate(self.label_names)}
        self.label_id2label_name = {label_id: label_name for label_name, label_id in self.label_name2label_id.items()}
        self.training_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.training_claim_ids])
        self.dev_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.dev_claim_ids])
        self.test_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.test_claim_ids])
        self.num_labels = len(self.label_names)

        self.label_name2training_claim_ids = {label_name: [] for label_name in self.label_names}
        for claim_id in self.training_claim_ids:
            label = self.claims[claim_id]['label']
            self.label_name2training_claim_ids[label].append(claim_id)
        self.label_id2training_claim_ids = {label_id: self.label_name2training_claim_ids[self.label_id2label_name[label_id]] for label_id in range(self.num_labels)}

        self.test_evid_ids = []
        for claim_id in self.test_claim_ids:
            self.test_evid_ids.extend(self.claims[claim_id]['gold_evid_ids'])
        self.test_evid_ids = np.unique(self.test_evid_ids)
        self.num_test_evid = len(self.test_evid_ids)

    def preprocess_data(self):

        claim_texts = {}
        for claim_id in self.claims.keys():
            claim_texts[claim_id] = self.claims[claim_id]['claim_text']

        evid_texts = {}
        for evid_id in range(self.num_evidences):
            evid_texts[evid_id] = self.evidences[evid_id]['evid_text']

        if self.has_contexts:
            ctx_texts = {}
            for ctx_id in range(self.num_contexts):
                ctx_texts[ctx_id] = self.contexts[ctx_id]['ctx_text']

        if self.has_references:
            ref_texts = {}
            for ref_id in range(self.num_references):
                ref_texts[ref_id] = self.references[ref_id]['ref_text']

        self.claim_input_ids, self.claim_attention_mask = self.generate_input_ids_and_attention_mask(claim_texts)
        self.evid_input_ids, self.evid_attention_mask = self.generate_input_ids_and_attention_mask(evid_texts)
        if self.has_contexts:
            self.ctx_input_ids, self.ctx_attention_mask = self.generate_input_ids_and_attention_mask(ctx_texts)
        if self.has_references:
            self.ref_input_ids, self.ref_attention_mask = self.generate_input_ids_and_attention_mask(ref_texts)

    def generate_input_ids_and_attention_mask(self, texts):

        input_ids, attention_mask = {}, {}
        for text_id in texts.keys():
            text = texts[text_id]
            text = text.strip()
            tokenized_text = self.tokenizer.batch_encode_plus([text], max_length=self.max_text_length, padding='max_length', truncation=True)
            input_ids[text_id] = np.squeeze(tokenized_text['input_ids'])
            attention_mask[text_id] = np.squeeze(tokenized_text['attention_mask'])

        return input_ids, attention_mask

    def sample_evid_and_ref(self):

        self.sampled_evid_ids = {}
        for claim_id in self.claims.keys():
            claim = self.claims[claim_id]
            evidence_ids = claim['gold_evid_ids'] if self.evidence_setting == 'gold' else claim['bm25_retrieved_evid_ids']
            replace = len(evidence_ids) < self.num_sampled_evidence
            sampled_evid_indices = np.random.choice(len(evidence_ids), size=self.num_sampled_evidence, replace=replace)
            self.sampled_evid_ids[claim_id] = [evidence_ids[sampled_evid_idx] for sampled_evid_idx in sampled_evid_indices]

        if self.has_references:
            self.sampled_ref_ids = {}
            for evid_id in self.evidences.keys():
                evidence = self.evidences[evid_id]
                if len(evidence['ref_ids']) == 0:
                    self.sampled_ref_ids[evid_id] = [-1] * self.num_sampled_references
                    continue
                replace = len(evidence['ref_ids']) < self.num_sampled_references
                sampled_ref_indices = np.random.choice(len(evidence['ref_ids']), size=self.num_sampled_references, replace=replace)
                self.sampled_ref_ids[evid_id] = [evidence['ref_ids'][sampled_ref_idx] for sampled_ref_idx in sampled_ref_indices]


class Data(Dataset):

    def __init__(self, data, mode, num_shots):

        super(Data, self).__init__()
        self.data = data
        self.mode = mode
        self.num_shots = num_shots

        if self.mode == 'supervised':
            self.claim_ids = self.data.training_claim_ids
            self.labels = self.data.training_labels

        elif self.mode == 'few_shot':
            self.claim_ids = []
            for label_name in self.data.label_names:
                claim_ids_one_label = self.data.label_name2training_claim_ids[label_name]
                claim_indices_one_label = np.random.choice(len(claim_ids_one_label), size=self.num_shots, replace=False)
                self.claim_ids.extend([claim_ids_one_label[claim_idx] for claim_idx in claim_indices_one_label])
            self.claim_ids = np.array(self.claim_ids)
            np.random.shuffle(self.claim_ids)

            self.labels = []
            for claim_id in self.claim_ids:
                self.labels.append(self.data.label_name2label_id[self.data.claims[claim_id]['label']])
            self.labels = np.array(self.labels)

        elif self.mode == 'test':
            self.claim_ids = self.data.test_claim_ids
            self.labels = self.data.test_labels

    def __len__(self):

        return len(self.claim_ids)

    def __getitem__(self, idx):

        claim_id = self.claim_ids[idx]
        label = self.labels[idx]

        return claim_id, label