import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util
from common_utils import get_logger
import logging

logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name='')


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]
    # token_h.shape: batch-size*seq-length, emb-size
    token_h = h.view(-1, emb_size)
    # flat.shape: batch-size*seq-length
    # the view() function cannot be applied to a discontiguous tensor. This is probably because view() requires that
    # the tensor to be contiguously stored so that it can do fast reshape in memory.
    # To solve this, simply add contiguous() to a discontiguous tensor, to create contiguous copy and then apply view()
    # I think it may be a old writing way, as x is from torch.stack, already contiguous, not need to call
    # flat = x.contiguous().view(-1)
    flat = x.view(-1)

    # get contextualized embedding of given token
    # tmp = (flat == token)  # shape: the same as flat, batch-size*seq-length
    # logger.debug(f'tmp {tmp.shape}')
    # tmp = token_h[tmp, :]  # shape: keeps only the true value in tmp, the return shape is: batch-size, emd-size
    token_h = token_h[flat == token, :]

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations
    1. assume max entity size, that's the max word number within the entity is l< 100. I think it had better read
    max length from entity size, sometimes may larger than 100 words. Especially in biomedical.
    2. entity_types_count and relation_types_count are from input_reader which read trian and eva corpus and then the
    values may be different. So, I should assign max_pairs and mmax_entity_count which are based on whole analysis on
    the whole dataset.
    """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int,
                relation_type_count: int, entity_type_count: int,
                size_embedding: int, prop_drop: float, freeze_transformer: bool,
                max_pairs: int = 100, max_entity_count = 100,
                ):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_type_count)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_type_count)
        self.size_embeddings = nn.Embedding(max_entity_count, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_type_count = relation_type_count
        self._entity_type_count = entity_type_count
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        """  
        entity_clf shape: batch-size, entities-num, entity_type_count
        rel_clf shape: batch_size, relation_num, _relation_type_count
        """
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify relations
        # max(min(relations.shape[1], self._max_pairs), 1), nice design, some doc may have no relation and
        # so use max(a, 1). relations.shape[1] is the current batch docs max relations numbers.
        # as there is zero corner case process, it is not necessary 
        h_large = h.unsqueeze(1).repeat(1, min(relations.shape[1], self._max_pairs), 1, 1)
        # h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_type_count]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_type_count]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        """
        entity_masks.shape: batch-size, entities-num, seq-length
        h.shape: batch-size, seq-length, bert-dim-size
        size_embeddings.shape: batch-size, entities-num, size-emb
        """
        # max pool entity candidate spans

        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)

        # entity_spans_pool.shape: batch-size, entities-num, seq-length, bert-dim-size
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # entity_spans_pool.shape: batch-size, entities-num, bert-dim-size
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        # shape: batch-size, emb-size
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        # shape: batch-size, entities-num, entity_type_count
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start:int):
        """
        entity_spans.shape: batch-size, entities-num, bert-dim-size
        size_embeddings.shape: batch-size, entities-num, size-emb
        relations.shape: batch-size, relation-num, 2
        rel_masks.shape: batch-size, relation-num, seq-len

        h is NOT the orig bert last hidden state
        h.shape: batch-size, relation-num, seq-len, bert-dim-size
        """
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        # shape: batch-size, relation-num, 2, bert-dim-size
        entity_pairs = util.batch_index(entity_spans, relations)
        # batch-size, relation-num, 2 * bert-dim-size
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        # shape: batch-size, relation-num, 2 * size-emb
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        # shape: batch-size, relation-num, bert-dim-size
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        # shape: batch-size, relation-num, relation_type_count
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
