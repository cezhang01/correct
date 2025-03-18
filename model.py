import torch
import torch.nn as nn
import numpy as np
import sklearn
import transformers
from transformers import BertConfig, BertModel
from encoder import Encoder
from decoder import Classifier


class Model(nn.Module):

    def __init__(self, args, data):

        super(Model, self).__init__()
        self.data = data
        self.parse_args(args)
        if args.local_rank in [-1, 0]:
            self.show_config()
        self.generate_modules(args, data)

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.current_device = args.device
        self.mode = args.mode
        self.num_shots = args.num_shots
        self.ddp = args.distributed_training
        if self.ddp:
            self.world_size = args.world_size
        self.evidence_setting = args.evidence_setting
        self.num_prompt_embs = args.num_prompt_embs
        self.random_prompt_init = args.random_prompt_init
        self.language_model = args.language_model
        self.num_claims = self.data.num_claims
        self.num_training_claims = len(self.data.training_claim_ids)
        self.num_evidences = self.data.num_evidences
        self.has_contexts = self.data.has_contexts
        args.has_contexts = self.data.has_contexts
        if self.has_contexts:
            self.num_contexts = self.data.num_contexts
        self.has_references = self.data.has_references
        args.has_references = self.data.has_references
        if self.has_references:
            self.num_references = self.data.num_references
        self.num_labels = self.data.num_labels
        args.num_labels = self.data.num_labels
        self.max_text_length = args.max_text_length
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references
        self.num_epochs = args.num_epochs
        self.learning_rate_for_lm = args.learning_rate_for_lm
        self.learning_rate_for_prompt_embs = args.learning_rate_for_prompt_embs
        self.minibatch_size = args.minibatch_size

    def show_config(self):

        print('******************************************************')
        print('dataset name:', self.dataset_name)
        print('torch version:', torch.__version__)
        print('np version:', np.__version__)
        print('sklearn version:', sklearn.__version__)
        print('transformers version:', transformers.__version__)
        print('device:', self.current_device)
        print('distributed training:', self.ddp)
        if self.ddp:
            print('world size:', self.world_size)
        print('mode:', self.mode)
        if self.mode == 'few_shot':
            print('#shots:', self.num_shots)
        print('evidence setting:', self.evidence_setting)
        print('#prompt embeddings:', self.num_prompt_embs)
        print('random prompt init:', self.random_prompt_init)
        print('language model:', self.language_model)
        print('max text length:', self.max_text_length)
        print('#claims:', self.num_claims)
        print('#training claims:', self.num_training_claims)
        print('#evidence:', self.num_evidences)
        if self.has_contexts:
            print('#contextual documents:', self.num_contexts)
        if self.has_references:
            print('#referential documents:', self.num_references)
        print('#labels:', self.num_labels)
        print('#sampled evidence:', self.num_sampled_evidence)
        print('#sampled references:', self.num_sampled_references)
        print('#epochs:', self.num_epochs)
        print('learning rate for lm:', self.learning_rate_for_lm)
        print('learning rate for prompt embeddings:', self.learning_rate_for_prompt_embs)
        print('minibatch size:', self.minibatch_size)
        print('******************************************************')

    def generate_modules(self, args, data):

        config = BertConfig.from_pretrained(args.pretrained_model_name_or_path)
        self.lm = BertModel.from_pretrained(args.pretrained_model_name_or_path, config=config)
        args.num_hidden_layers = config.num_hidden_layers
        args.hidden_size = config.hidden_size

        self.encoder = Encoder(self.lm, config, args, data)
        self.classifier = Classifier(config, args)

    def preprocess_data(self, claim_ids, data):

        claim_ids = claim_ids.detach().cpu().numpy()
        evid_ids = np.array([data.sampled_evid_ids[claim_id] for claim_id in claim_ids])
        evid_ids = np.reshape(evid_ids, [-1])

        claim_input_ids = np.array([data.claim_input_ids[claim_id] for claim_id in claim_ids])
        claim_attention_mask = np.array([data.claim_attention_mask[claim_id] for claim_id in claim_ids])
        claim_input_ids = np.reshape(claim_input_ids, [-1, self.max_text_length])
        claim_input_ids = torch.LongTensor(claim_input_ids).to(self.current_device)
        claim_attention_mask = np.reshape(claim_attention_mask, [-1, self.max_text_length])
        claim_attention_mask = torch.LongTensor(claim_attention_mask).to(self.current_device)

        evid_input_ids = np.array([data.evid_input_ids[evid_id] for evid_id in evid_ids])
        evid_attention_mask = np.array([data.evid_attention_mask[evid_id] for evid_id in evid_ids])
        evid_input_ids = np.reshape(evid_input_ids, [-1, self.max_text_length])
        evid_input_ids = torch.LongTensor(evid_input_ids).to(self.current_device)
        evid_attention_mask = np.reshape(evid_attention_mask, [-1, self.max_text_length])
        evid_attention_mask = torch.LongTensor(evid_attention_mask).to(self.current_device)

        ctx_input_ids, ctx_attention_mask = None, None
        if data.has_contexts:
            ctx_ids = [data.evidences[evid_id]['ctx_id'][0] for evid_id in evid_ids]
            ctx_input_ids = np.array([data.ctx_input_ids[ctx_id] for ctx_id in ctx_ids])
            ctx_attention_mask = np.array([data.ctx_attention_mask[ctx_id] for ctx_id in ctx_ids])
            ctx_input_ids = np.reshape(ctx_input_ids, [-1, self.max_text_length])
            ctx_input_ids = torch.LongTensor(ctx_input_ids).to(self.current_device)
            ctx_attention_mask = np.reshape(ctx_attention_mask, [-1, self.max_text_length])
            ctx_attention_mask = torch.LongTensor(ctx_attention_mask).to(self.current_device)

        ref_input_ids, ref_attention_mask = None, None
        if data.has_references:
            ref_input_ids, ref_attention_mask = [], []
            for evid_id in evid_ids:
                ref_ids_one_evid = data.sampled_ref_ids[evid_id]
                if ref_ids_one_evid[0] == -1:
                    ref_input_ids_one_evid = np.tile(np.reshape(np.array(data.evid_input_ids[evid_id]), [1, -1]), [self.num_sampled_references, 1])
                    ref_attention_mask_one_evid = np.tile(np.reshape(np.array(data.evid_attention_mask[evid_id]), [1, -1]), [self.num_sampled_references, 1])
                else:
                    ref_input_ids_one_evid = np.array([data.ref_input_ids[ref_id] for ref_id in ref_ids_one_evid])
                    ref_attention_mask_one_evid = np.array([data.ref_attention_mask[ref_id] for ref_id in ref_ids_one_evid])
                ref_input_ids.append(ref_input_ids_one_evid)
                ref_attention_mask.append(ref_attention_mask_one_evid)
            ref_input_ids = np.concatenate(ref_input_ids, axis=0)
            ref_attention_mask = np.concatenate(ref_attention_mask, axis=0)
            ref_input_ids = np.reshape(ref_input_ids, [-1, self.max_text_length])
            ref_input_ids = torch.LongTensor(ref_input_ids).to(self.current_device)
            ref_attention_mask = np.reshape(ref_attention_mask, [-1, self.max_text_length])
            ref_attention_mask = torch.LongTensor(ref_attention_mask).to(self.current_device)

        return (claim_input_ids,
                claim_attention_mask,
                evid_input_ids,
                evid_attention_mask,
                ctx_input_ids,
                ctx_attention_mask,
                ref_input_ids,
                ref_attention_mask)

    def forward(self, claim_ids, labels, data, mode):

        # preprocess data
        (claim_input_ids,
         claim_attention_mask,
         evid_input_ids,
         evid_attention_mask,
         ctx_input_ids,
         ctx_attention_mask,
         ref_input_ids,
         ref_attention_mask) = self.preprocess_data(claim_ids, data)

        # evidence encoder
        evid_emb = self.encoder(lm=self.lm,
                                input_ids=evid_input_ids,
                                attention_mask=evid_attention_mask,
                                ctx_input_ids=ctx_input_ids,
                                ctx_attention_mask=ctx_attention_mask,
                                ref_input_ids=ref_input_ids,
                                ref_attention_mask=ref_attention_mask,
                                claim_or_evid='evid',
                                mode=mode)

        # claim encoder
        claim_emb_list = self.encoder(lm=self.lm,
                                      input_ids=claim_input_ids,
                                      attention_mask=claim_attention_mask,
                                      evid_emb=evid_emb,
                                      claim_or_evid='claim',
                                      mode=mode)

        # classifier
        loss, y_pred = self.classifier(claim_emb_list, evid_emb, labels)

        return [loss, y_pred]