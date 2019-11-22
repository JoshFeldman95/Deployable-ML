from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch
import torchtext
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers.modeling_distilbert import DistilBertForSequenceClassification, DistilBertModel, Embeddings, Transformer

import spacy
spacy.load('en')
from spacy.lang.en import STOP_WORDS


# Set params
device = 'cpu'
pretrained_weights = 'distilbert-base-uncased' # specify type of BERT we want


class EmbeddingsWrapper(Embeddings):
    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)               # (bs, max_seq_length, dim)
        return embeddings, word_embeddings


class DistilBertModelWrapper(DistilBertModel):
    def __init__(self, config):
        super(DistilBertModel, self).__init__(config)

        self.embeddings = EmbeddingsWrapper(config)   # Embeddings
        self.transformer = Transformer(config) # Encoder

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device) # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds, word_embeds = self.embeddings(input_ids)   # (bs, seq_length, dim)
        tfmr_output = self.transformer(x=inputs_embeds,
                                       attn_mask=attention_mask,
                                       head_mask=head_mask)
        hidden_state = tfmr_output[0]
        output = (hidden_state, ) + tfmr_output[1:]

        return output, word_embeds # last-layer hidden-state, (all hidden_states), (all attentions)


class DistilBertForSequenceClassificationWrapper(DistilBertForSequenceClassification):
    def __init__(self, config):
        super(DistilBertForSequenceClassificationWrapper, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModelWrapper(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output, word_embeds = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask,
                                            inputs_embeds=inputs_embeds)
        hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                    # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs, word_embeds  # (loss), logits, (hidden_states), (attentions)

def normalize(x, dim):
    norm = (x**2).sum(dim)
    magnitude = norm.sqrt()
    return x/magnitude.unsqueeze(len(magnitude.shape))

def attack(BERT_with_inputs, batch, TEXT, eps):
    # get embedding grad scores
    out, embeddings = BERT_with_inputs(batch.text)
    label = batch.label - 1
    loss = criterion(out[0], label)
    embeddings.retain_grad()
    loss.backward()

    # mask stop words and special tokens
    stop_ids = [0, 101, 102]
    
    for sw in STOP_WORDS:
        try:
            stop_ids.append(TEXT.vocab.stoi[sw])
        except KeyError:
            pass

    mask = batch.text == 0
    for  sw_id in stop_ids:
        mask = mask | (batch.text == sw_id)
    mask = ~mask

    # mask embedding grads
    embeddings.grad = embeddings.grad * mask.unsqueeze(2)

    # Get adversarial embeddings
    adv_embeddings = embeddings + eps*embeddings.grad

    # get all embeddings
    all_embeddings = BERT_with_inputs.distilbert.embeddings.word_embeddings.weight

    # compute cosine distance between adversarial embeddings and
    # all original embeddings to find closest words
    adv_embeddings_normed = normalize(adv_embeddings, 2)
    all_embeddings_normed = normalize(all_embeddings, 1)
    cos_sim = torch.matmul(adv_embeddings_normed, all_embeddings_normed.T)

    # get adversarial text
    _, adv_tokens = cos_sim.topk(1, dim = 2)
    adv_tokens = adv_tokens.squeeze(2)
    return adv_tokens
