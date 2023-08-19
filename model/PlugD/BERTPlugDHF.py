import transformers
from transformers import BertModel,BertForMaskedLM
from torch import nn
from typing import Optional, Tuple, Union
import torch
import types
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions,BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import apply_chunking_to_forward
import math
import random

def change_bert_forward(bert):
    def _self_attention_forward(
        self_self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        ctx_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self_self.query(hidden_states)

        if ctx_hidden_states is None:
            concat_hidden = hidden_states
        else:
            concat_hidden = torch.cat([hidden_states, ctx_hidden_states], dim=1)
        key_layer = self_self.transpose_for_scores(self_self.key(concat_hidden))
        value_layer = self_self.transpose_for_scores(self_self.value(concat_hidden))

        query_layer = self_self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self_self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self_self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self_self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        return outputs

    def _attention_forward(
        att_self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        ctx_hidden_states=None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = att_self.self(
            hidden_states,
            attention_mask,
            ctx_hidden_states=ctx_hidden_states,
        )
        attention_output = att_self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def _layer_forward(
        layer_self,
        hidden_states,
        attention_mask=None,
        ctx_hidden_states=None,
        use_cache=True,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = layer_self.attention(
            hidden_states,
            attention_mask,
            ctx_hidden_states=ctx_hidden_states,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            layer_self.feed_forward_chunk, layer_self.chunk_size_feed_forward, layer_self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs
    
    def _enc_forward(
        enc_self,
        hidden_states=None,
        attention_mask=None,
        ctx_hidden_states=None,
        output_hidden_states=None,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(enc_self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                ctx_hidden_states=ctx_hidden_states,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
    

    def bert_forward(
        bert_self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_input_ids: Optional[torch.Tensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        '''Encode CTX'''
        if ctx_input_ids is not None:
            batch_size, ctx_seq_length = ctx_input_shape = ctx_input_ids.size()
            device = ctx_input_ids.device 

            if ctx_attention_mask is None:
                ctx_attention_mask = torch.ones(((batch_size, ctx_seq_length)), device=device)
            ctx_extended_attention_mask: torch.Tensor = bert_self.get_extended_attention_mask(ctx_attention_mask, ctx_input_shape)

            ctx_embedding_output = bert_self.embeddings(
                input_ids=ctx_input_ids,
                token_type_ids=torch.ones(ctx_input_shape, dtype=torch.long, device=device),
                past_key_values_length=0,
            )
            ctx_encoder_outputs = bert_self.encoder(
                ctx_embedding_output,
                attention_mask=ctx_extended_attention_mask,
                output_hidden_states=False,
            )

            ctx_sequence_output = ctx_encoder_outputs[0]
            # print(ctx_extended_attention_mask.shape)
            # print(ctx_sequence_output.shape)
        else:
            ctx_sequence_output = None

        '''Encode query'''
        batch_size, seq_length = input_shape = input_ids.size()
        device = input_ids.device 

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if ctx_input_ids is not None:
            concat_mask = attention_mask.unsqueeze(2) * torch.cat([attention_mask, ctx_attention_mask], dim=1).unsqueeze(1)
        else:
            concat_mask = attention_mask
        extended_attention_mask: torch.Tensor = bert_self.get_extended_attention_mask(concat_mask, None)


        embedding_output = bert_self.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros(input_shape, dtype=torch.long, device=device),
            past_key_values_length=0,
        )
        encoder_outputs = bert_self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            ctx_hidden_states=ctx_sequence_output,
            output_hidden_states=True,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = bert_self.pooler(sequence_output) if bert_self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    bert.forward = types.MethodType(bert_forward, bert)
    bert.encoder.forward = types.MethodType(_enc_forward, bert.encoder)
    for layer in bert.encoder.layer:
        layer.forward = types.MethodType(_layer_forward, layer)
        layer.attention.forward = types.MethodType(_attention_forward, layer.attention)
        layer.attention.self.forward = types.MethodType(_self_attention_forward, layer.attention.self)

class BERTPlugD(nn.Module):
    def __init__(self, t5path="bert-base-uncased"):
        super().__init__()
        # self.model = BertModel.from_pretrained(t5path)
        self.model = BertForMaskedLM.from_pretrained(t5path)
        change_bert_forward(self.model.bert)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, query_input_ids, query_attention_mask, ctx_input_ids=None, ctx_attention_mask=None):
        out = self.model.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            ctx_input_ids=ctx_input_ids,
            ctx_attention_mask=ctx_attention_mask,
        )
        sequence_output = out[0]
        prediction_scores = self.model.cls(sequence_output)
        return prediction_scores
        hidden = out["pooler_output"]
        logits = self.output(hidden)

        return logits

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

if __name__ =="__main__":
    set_seed(2333)
    # model = T5ForConditionalGeneration.from_pretrained("/liuzyai04/thunlp/xcj/PLMs/t5-base")
    # # print(model.config)
    # change_t5_forward(model)
    inp = torch.randint(100, 10000, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.int)
    ctx_inp = torch.randint(100, 10000, (2, 20))
    ctx_attention_mask = torch.ones(2, 20, dtype=torch.int)
    # model(
    #     input_ids=inp, attention_mask=attention_mask,
    #     ctx_input_ids=ctx_inp, ctx_attention_mask=ctx_attention_mask
    # )
    model = BERTPlugD("bert-base-uncased")
    print(model(inp, attention_mask, ctx_inp, ctx_attention_mask))
