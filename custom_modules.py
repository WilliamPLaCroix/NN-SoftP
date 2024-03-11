from transformers import Trainer
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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.neg_weights, self.pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
