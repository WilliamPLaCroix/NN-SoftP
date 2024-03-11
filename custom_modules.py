from typing import List, Optional, Tuple, Union

from transformers import Trainer, LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from torch import nn
from torch.nn import CrossEntropyLoss
import torch


class WeightedCELossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_weights()

    def compute_weights(self):
        """
        Calculates class weights based on label distribution in training data.
        """
        train_df = self.train_dataset.to_pandas()
        target_counts = train_df["label"].value_counts()
        self.pos_weights = len(train_df) / (2 * target_counts[0])  # Assuming positive label is 0
        self.neg_weights = len(train_df) / (2 * target_counts[1])
        print(f"Label weights are:\n\ttrue: {self.pos_weights}\n\tfake: {self.neg_weights}")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.neg_weights, self.pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class LlamaClfCnn(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = LlamaModel(config)
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=5)

        self.reduce = nn.Linear(640, 64) # 640 = 128 * 5
        self.score = nn.Linear(64, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        x = self.conv1(hidden_states)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.reduce(x)
        logits = self.score(x)

#        if input_ids is not None:
#            batch_size = input_ids.shape[0]
#        else:
#            batch_size = inputs_embeds.shape[0]
#
#        if self.config.pad_token_id is None and batch_size != 1:
#            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#        if self.config.pad_token_id is None:
#            sequence_lengths = -1
#        else:
#            if input_ids is not None:
#                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
#                sequence_lengths = sequence_lengths % input_ids.shape[-1]
#                sequence_lengths = sequence_lengths.to(logits.device)
#            else:
#                sequence_lengths = -1
#
#        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
#            if self.config.problem_type is None:
#                if self.num_labels == 1:
#                    self.config.problem_type = "regression"
#                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                    self.config.problem_type = "single_label_classification"
#                else:
#                    self.config.problem_type = "multi_label_classification"
#
#            if self.config.problem_type == "regression":
#                loss_fct = MSELoss()
#                if self.num_labels == 1:
#                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                else:
#                    loss = loss_fct(pooled_logits, labels)
#            elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#            elif self.config.problem_type == "multi_label_classification":
#                loss_fct = BCEWithLogitsLoss()
#                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

