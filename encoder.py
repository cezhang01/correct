import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertSelfAttention
import numpy as np


class PromptLearner(nn.Module):

    def __init__(self, config, args, data, token_embs):

        super(PromptLearner, self).__init__()
        self.current_device = args.device
        self.language_model = args.language_model
        self.num_prompt_embs = args.num_prompt_embs
        self.random_prompt_init = args.random_prompt_init
        self.hidden_size = args.hidden_size
        self.num_labels = args.num_labels
        self.training_claim_ids = data.training_claim_ids
        self.training_labels = data.training_labels
        self.token_embs = token_embs
        self.num_sampled_references = args.num_sampled_references

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.Linear(self.hidden_size, self.hidden_size)])
        self.tanh = nn.Tanh()
        self.init_prompts(data)

    def init_prompts(self, data):

        if self.random_prompt_init:
            with torch.no_grad():
                token_ids = np.random.randint(len(self.token_embs), size=[self.num_labels, self.num_prompt_embs])
                self.prompt_embs = self.token_embs[token_ids]
        else:
            self.prompt_embs = []
            for label_id in range(self.num_labels):
                claim_ids_one_label = data.label_id2training_claim_ids[label_id]

                claim_input_ids_one_label = np.array([data.claim_input_ids[claim_id][1:self.num_prompt_embs + 1] for claim_id in claim_ids_one_label])
                claim_input_ids_one_label = torch.IntTensor(claim_input_ids_one_label)

                evid_ids_one_label = np.array([data.sampled_evid_ids[claim_id] for claim_id in claim_ids_one_label])
                evid_ids_one_label = np.reshape(evid_ids_one_label, [-1])
                evid_input_ids_one_label = np.array([data.evid_input_ids[evid_id][1:self.num_prompt_embs + 1] for evid_id in evid_ids_one_label])
                evid_input_ids_one_label = torch.IntTensor(evid_input_ids_one_label)

                with torch.no_grad():
                    prompt_embs_claim = self.token_embs(claim_input_ids_one_label)
                    prompt_embs_evid = self.token_embs(evid_input_ids_one_label)
                    prompt_embs = torch.concat([prompt_embs_claim, prompt_embs_evid], dim=0)
                    prompt_embs = torch.mean(prompt_embs, dim=0, keepdim=True)
                    self.prompt_embs.append(prompt_embs)

            self.prompt_embs = torch.concat(self.prompt_embs, dim=0)

        self.prompt_embs = nn.Parameter(self.prompt_embs)

    def forward(self, input_ids, token_embs, evid_emb):

        inputs_embeds = token_embs(input_ids)

        temperature = 100
        scaling = self.tanh(self.linear_layers[0](evid_emb) / temperature)
        shifting = self.tanh(self.linear_layers[1](evid_emb) / temperature)
        scaling = scaling.unsqueeze(dim=1).expand(-1, self.num_prompt_embs, -1)
        shifting = shifting.unsqueeze(dim=1).expand(-1, self.num_prompt_embs, -1)

        prompts_list = []
        for label_id in range(self.num_labels):
            prompt_embs = self.prompt_embs[label_id, :, :]
            prompt_embs = prompt_embs.unsqueeze(0).expand(inputs_embeds.size(0), -1, -1)
            prompt_embs = torch.multiply(prompt_embs, scaling + 1) + shifting
            prompts = torch.concat([inputs_embeds[:, :1, :], prompt_embs, inputs_embeds[:, 1:, :]], dim=1)  # may need to add contect prompts to the end of the sequence
            prompts_list.append(prompts)

        return prompts_list


class EncoderLayer(nn.Module):

    def __init__(self, config, args):

        super(EncoderLayer, self).__init__()
        self.language_model = args.language_model
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers
        self.has_contexts = args.has_contexts
        self.has_references = args.has_references
        self.num_prompt_embs = args.num_prompt_embs

        config_graph_conv = BertConfig.from_pretrained('bert-base-uncased')
        config_graph_conv.num_attention_heads = 1
        config_graph_conv.hidden_size = self.hidden_size
        self.graph_conv_layers = nn.ModuleList([BertSelfAttention(config=config_graph_conv),
                                                BertSelfAttention(config=config_graph_conv),
                                                BertSelfAttention(config=config_graph_conv)])

    def forward(self, lm, hidden_states, attention_mask, ctx_hidden_states, ctx_attention_mask, ref_hidden_states, ref_attention_mask, claim_or_evid, mode):

        all_hidden_states = ()

        for layer_id in range(self.num_hidden_layers):
            all_hidden_states = all_hidden_states + (hidden_states,)
            if layer_id > 0:
                cls_emb = hidden_states[:, 3, :].clone()

                # multi-evidence reasoning
                attention_mask_tmp = attention_mask.clone()
                if claim_or_evid == 'evid':
                    cls_emb_reshape = torch.reshape(cls_emb, [-1, self.num_sampled_evidence, self.hidden_size])
                    cls_emb_agg = torch.reshape(self.graph_conv_layers[0](cls_emb_reshape)[0], [-1, self.hidden_size])
                    hidden_states[:, 0, :] = cls_emb_agg
                elif claim_or_evid == 'claim':
                    attention_mask_tmp[:, :, :, :3] = -10000.0

                # contextual document reasoning
                if ctx_hidden_states is not None:
                    ctx_cls_emb = ctx_hidden_states[:, 3, :].clone()
                    self_and_ctx_emb = torch.concat([torch.unsqueeze(cls_emb, dim=1), torch.unsqueeze(ctx_cls_emb, dim=1)], dim=1)
                    ctx_cls_emb_agg = self.graph_conv_layers[1](self_and_ctx_emb)[0][:, 0, :]
                    hidden_states[:, 1, :] = ctx_cls_emb_agg
                    ctx_lm_layer_outputs = lm.encoder.layer[layer_id](ctx_hidden_states, attention_mask=ctx_attention_mask)

                # referential document reasoning
                if ref_hidden_states is not None:
                    ref_cls_emb = ref_hidden_states[:, 3, :].clone()
                    ref_cls_emb = torch.reshape(ref_cls_emb, [-1, self.num_sampled_references, self.hidden_size])
                    self_and_ref_emb = torch.concat([torch.unsqueeze(cls_emb, dim=1), ref_cls_emb], dim=1)
                    ref_cls_emb_agg = self.graph_conv_layers[2](self_and_ref_emb)[0][:, 0, :]
                    hidden_states[:, 2, :] = ref_cls_emb_agg
                    ref_lm_layer_outputs = lm.encoder.layer[layer_id](ref_hidden_states, attention_mask=ref_attention_mask)

                lm_layer_outputs = lm.encoder.layer[layer_id](hidden_states, attention_mask=attention_mask_tmp)

            else:
                attention_mask_tmp = attention_mask.clone()
                attention_mask_tmp[:, :, :, :3] = -10000.0
                lm_layer_outputs = lm.encoder.layer[0](hidden_states, attention_mask=attention_mask_tmp)
                if ctx_hidden_states is not None:
                    ctx_lm_layer_outputs = lm.encoder.layer[0](ctx_hidden_states, attention_mask=ctx_attention_mask)
                if ref_hidden_states is not None:
                    ref_lm_layer_outputs = lm.encoder.layer[0](ref_hidden_states, attention_mask=ref_attention_mask)

            hidden_states = lm_layer_outputs[0]
            if ctx_hidden_states is not None:
                ctx_hidden_states = ctx_lm_layer_outputs[0]
            if ref_hidden_states is not None:
                ref_hidden_states = ref_lm_layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states


class Encoder(nn.Module):

    def __init__(self, lm, config, args, data):

        super(Encoder, self).__init__()
        self.language_model = args.language_model
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references
        self.hidden_size = args.hidden_size
        self.num_prompt_embs = args.num_prompt_embs
        self.num_labels = args.num_labels
        self.has_contexts = args.has_contexts
        self.has_references = args.has_references

        self.encoder_layer = EncoderLayer(config, args)
        token_embs = lm.embeddings.word_embeddings
        self.prompt_learner = PromptLearner(config, args, data, token_embs)

    def prepend_hidden_states_and_attention_mask(self, hidden_states, attention_mask):

        num_texts = hidden_states.size(0)

        # prepend hidden states
        station_placeholder = torch.zeros([num_texts, 3, hidden_states.size(-1)]).type(hidden_states.dtype).to(hidden_states.device)
        hidden_states = torch.cat([station_placeholder, hidden_states], dim=1)

        # prepend attention mask
        station_mask = torch.zeros([num_texts, 3], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)

        return hidden_states, attention_mask

    def encoder(self, lm, hidden_states, attention_mask, ctx_input_ids=None, ctx_attention_mask=None, ref_input_ids=None, ref_attention_mask=None, claim_or_evid='evid', mode='train'):

        # prepend hidden states and attention mask
        hidden_states, attention_mask = self.prepend_hidden_states_and_attention_mask(hidden_states, attention_mask)

        attention_mask[:, 0] = 1.0  # multi-evidence reasoning

        # initialize hidden states and attention mask of contexts
        ctx_hidden_states, ctx_extended_attention_mask = None, None
        if ctx_input_ids is not None:
            attention_mask[:, 1] = 1.0  # contextual document reasoning
            ctx_hidden_states = lm.embeddings(input_ids=ctx_input_ids)
            ctx_hidden_states, ctx_attention_mask = self.prepend_hidden_states_and_attention_mask(ctx_hidden_states, ctx_attention_mask)
            ctx_extended_attention_mask = (1.0 - ctx_attention_mask[:, None, None, :]) * -10000.0

        # initialize hidden states and attention mask of references
        ref_hidden_states, ref_extended_attention_mask = None, None
        if ref_input_ids is not None:
            attention_mask[:, 2] = 1.0  # referential document reasoning
            ref_hidden_states = lm.embeddings(input_ids=ref_input_ids)
            ref_hidden_states, ref_attention_mask = self.prepend_hidden_states_and_attention_mask(ref_hidden_states, ref_attention_mask)
            ref_extended_attention_mask = (1.0 - ref_attention_mask[:, None, None, :]) * -10000.0

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # encoder
        encoder_outputs = self.encoder_layer(lm=lm,
                                             hidden_states=hidden_states,
                                             attention_mask=extended_attention_mask,
                                             ctx_hidden_states=ctx_hidden_states,
                                             ctx_attention_mask=ctx_extended_attention_mask,
                                             ref_hidden_states=ref_hidden_states,
                                             ref_attention_mask=ref_extended_attention_mask,
                                             claim_or_evid=claim_or_evid,
                                             mode=mode)

        hidden_states = encoder_outputs[-1]
        cls_emb = hidden_states[:, 3, :]
        if claim_or_evid == 'evid':
            cls_emb = torch.reshape(cls_emb, [-1, self.num_sampled_evidence, self.hidden_size])
            cls_emb = torch.mean(cls_emb, dim=1)

        return cls_emb

    def forward(self, lm, input_ids, attention_mask, ctx_input_ids=None, ctx_attention_mask=None, ref_input_ids=None, ref_attention_mask=None, evid_emb=None, claim_or_evid='evid', mode='train'):

        num_texts = input_ids.size(0)

        if claim_or_evid == 'claim':
            # obtain prompt embeddings
            token_embs = lm.embeddings.word_embeddings
            prompts_list = self.prompt_learner(input_ids, token_embs, evid_emb)
            station_mask = torch.ones([num_texts, self.num_prompt_embs], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask[:, :1], station_mask, attention_mask[:, 1:]], dim=-1)

            # claim encoder
            text_emb_list = []
            for label_id in range(self.num_labels):
                hidden_states = lm.embeddings(inputs_embeds=prompts_list[label_id])
                text_emb = self.encoder(lm=lm,
                                        hidden_states=hidden_states,
                                        attention_mask=attention_mask,
                                        claim_or_evid=claim_or_evid,
                                        mode=mode)
                text_emb_list.append(text_emb)
            return text_emb_list

        elif claim_or_evid == 'evid':
            # evidence encoder
            hidden_states = lm.embeddings(input_ids=input_ids)
            text_emb = self.encoder(lm=lm,
                                    hidden_states=hidden_states,
                                    attention_mask=attention_mask,
                                    ctx_input_ids=ctx_input_ids,
                                    ctx_attention_mask=ctx_attention_mask,
                                    ref_input_ids=ref_input_ids,
                                    ref_attention_mask=ref_attention_mask,
                                    claim_or_evid=claim_or_evid,
                                    mode=mode)
            return text_emb