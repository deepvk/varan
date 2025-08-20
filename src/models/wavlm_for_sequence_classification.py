from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import WavLMForSequenceClassification

from src.models.varan import AttentionDistributionPredictor
from src.utils.classification import VaranClassifivationOutput


class VaranWavLMForSequenceClassification(WavLMForSequenceClassification):
    def __init__(
        self,
        config,
        varan_attention_num_heads=16,
        varan_separate_heads=True,
        kl_beta=0.1,
    ):
        self.varan_attention_num_heads = varan_attention_num_heads
        self.varan_separate_heads = varan_separate_heads
        self.kl_beta = kl_beta
        super().__init__(config)
        self.posterior_predictor = AttentionDistributionPredictor(
            config.hidden_size, num_heads=16
        )
        self.post_init()
        self._prior_distribution = None

    def set_prior_distribution(self, prior_distribution):
        self._prior_distribution = prior_distribution

    def use_separate_classifiers(self):
        output_hidden_size = (
            self.config.output_hidden_size
            if hasattr(self.config, "add_adapter") and self.config.add_adapter
            else self.config.hidden_size
        )

        self.projector = nn.ModuleList(
            [
                nn.Linear(output_hidden_size, self.config.proj_codevector_dim)
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.config.proj_codevector_dim, self.config.num_labels)
                for _ in range(self.config.num_hidden_layers)
            ]
        )

    def predict_posterior_distribution(self, attention_mask, hidden_states):
        """
        This function averages hiddens across sequence length and predicts weights to compute feature vector.
        :param attention_mask: [batch_size, seq_len, hid_dim] 1 non-masked, 0 masked
        :param hidden_states: tuple of size n_layers with [batch_size, seq_len, hid_dim] logits
        :return: [batch_size, n_layers] predicted logits to compute weights for VARAN
        """
        input_lengths = self._get_input_length(attention_mask)  # batch_size
        concatenated_hidden_states = torch.stack(
            [hidden_states[i] for i in range(1, len(hidden_states))], dim=-1
        )  # [batch_size, seq_len, hid_dim, n_layers]
        # apply mask along sequence length
        attention_mask = self.forward_padding_mask(hidden_states[0], attention_mask)
        masked_hidden_states = concatenated_hidden_states * attention_mask.unsqueeze(
            -1
        ).unsqueeze(-1)
        # average hiddens across sequence length
        averaged_masked_hidden_states = masked_hidden_states.sum(
            1
        ) / input_lengths.unsqueeze(-1).unsqueeze(-1)
        # predict weights per layer per sample
        predicted_weight_logits = self.varan_weights_predictor(
            averaged_masked_hidden_states.permute(
                0, 2, 1
            )  # [batch_size, n_layers, hidden_dim]
        )  # [batch_size, n_layers]
        return predicted_weight_logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> VaranClassifivationOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = True

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        def pool_hidden_states(hidden_states: Tuple[torch.FloatTensor]):
            # remove first output
            hidden_states = torch.stack(
                hidden_states[1:], dim=1
            )  # bs, n_layers, seq_len, hid_dim
            bs, n_layers, seq_len, hid_dim = hidden_states.shape
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[2], attention_mask
            )  # bs, seq_len
            padding_mask_exp = (
                padding_mask.unsqueeze(-1).unsqueeze(1).expand_as(hidden_states)
            )
            hidden_states[~padding_mask_exp] = 0.0
            return hidden_states.sum(2) / padding_mask.sum(-1).view(-1, 1, 1)

        pooled = pool_hidden_states(outputs[2])
        predicted_weight_logits = self.posterior_predictor(
            pooled  # [batch_size, n_layers, hidden_dim]
        )  # [batch_size, n_layers, hidden_dim]
        bs, num_layers, h_dim = pooled.shape
        if self.varan_separate_heads:
            logits = torch.zeros(
                bs,
                num_layers,
                self.config.num_labels,
                dtype=pooled.dtype,
                device=pooled.device,
            )
            for i, (cl, pr) in enumerate(zip(self.classifier, self.projector)):
                logits[:, i] = cl(pr(pooled[:, i]))
        else:
            logits = self.classifier(self.projector(pooled))

        bs, num_layers, num_classes = logits.shape
        predictions = (
            F.softmax(predicted_weight_logits, 1)
            .unsqueeze(-1)
            .repeat(1, 1, num_classes)
            * F.log_softmax(logits, -1)
        ).sum(1)
        assert predictions.shape == (bs, num_classes)
        loss = None
        kl_loss = None
        final_likelihood_loss = None
        if labels is not None:
            losses_likelihood = torch.zeros(bs, num_layers, device=pooled.device)
            for i in range(logits.size(1)):
                losses_likelihood[:, i] = F.cross_entropy(logits[:, i], labels.view(-1))
            final_likelihood_loss = (
                (F.softmax(predicted_weight_logits, 1) * losses_likelihood)
                .sum(1)
                .mean(0)
            )
            target_distribution = self._prior_distribution[:bs]
            kl_loss = F.kl_div(
                torch.log_softmax(predicted_weight_logits, dim=-1),
                target_distribution,
                reduction="batchmean",
            )
            loss = final_likelihood_loss + self.kl_beta * kl_loss
        return VaranClassifivationOutput(
            loss=loss,
            logits=predictions,
            ce_loss=final_likelihood_loss,
            kl_loss=kl_loss,
            layer_distribution_logits=predicted_weight_logits,
            hidden_states=outputs[1:],
        )
