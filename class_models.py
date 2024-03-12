import torch
from transformers import AutoModelForCausalLM
from trying_BERT import bnb_config, number_of_labels

class MLP_with_sentiment_etc(torch.nn.Module):
    def __init__(self, language_model):
        super(MLP_with_sentiment_etc, self).__init__()
        """
        # TODO : move lm_out and self.lm outside of class declaration
        """
        self.lm = AutoModelForCausalLM.from_pretrained(language_model, quantization_config=bnb_config)#, device_map='auto')
        self.requires_grad_(False)
        self.lm_out_size = self.lm.config.hidden_size

        """
        Param: self.idden size is a hyperparameter that can be tuned to increase the model's capacity
        Param: self.reducer is a linear layer that reduces the size of the LM output to hidden_size to
        prevent it from outweighing the additional features
        Param: self.classifier is a linear layer that reduces the size of the concatenated features to the number of labels
        """
        self.hidden_size = 100
        self.reducer = torch.nn.Linear(self.lm_out_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.classifier = torch.nn.Linear(self.hidden_size+5, number_of_labels, dtype=bnb_config.bnb_4bit_compute_dtype)


    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        # lm_out is the foundation of the model output, while hidden_states[-1] gives us word embeddings
        # we do mean pooling to get a single consistently sized vector for the entire sequence
        """
        # TODO : move lm_out and self.lm outside of class declaration
        """
        lm_out = self.lm(input_ids, attention_mask, output_hidden_states=True, labels=input_ids)
        outputs = lm_out.hidden_states[-1]
        outputs = torch.mean(outputs, dim=1, dtype=bnb_config.bnb_4bit_compute_dtype)

        # calculates perplexity as mean subword suprisal from LM output logits
        logits = torch.nn.functional.softmax(lm_out.logits, dim=-1).detach()
        probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(dim=2)).squeeze(-1)
        subword_surp = -1 * torch.log2(probs) * attention_mask
        mean_surprisal = subword_surp.sum(dim=1) / attention_mask.sum(dim=1)

        # bring LM output size down so that it doesn't outweigh the additional features
        outputs = self.reducer(outputs)
        outputs = torch.nn.LeakyReLU(outputs)
        
        # concatenate mean-pooled LM output with the additional features
        outputs = torch.cat((outputs.to(bnb_config.bnb_4bit_compute_dtype), 
                            sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                            perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1),
                            mean_surprisal.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                        dim=1)
        
        # final prediction is reduced to len(class_labels)
        return self.classifier(outputs)
