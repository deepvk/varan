import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMPreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from src.models.varan import AttentionDistributionPredictor
from src.utils.asr import ASRVaranOutput


class WavLMForCTC(WavLMPreTrainedModel):
    def __init__(
        self,
        config,
        prior_distribution,
        layer_aggregation_method="last_layer",
        varan_attention_num_heads=16,
        varan_separate_heads=True,
        kl_beta=0.1,
        blank_token_id=0,
    ):
        self.layer_aggregation_method = layer_aggregation_method
        self.varan_attention_num_heads = varan_attention_num_heads
        self.varan_separate_heads = varan_separate_heads
        self.prior_distribution = prior_distribution
        self.kl_beta = kl_beta
        self.blank_token_id = blank_token_id
        super().__init__(config)
        self.wavlm = WavLMModel(config)
        self.weights = None
        self.varan_weights_predictior = None
        if self.layer_aggregation_method == "weighted_sum":
            self.weights = nn.Parameter(torch.ones(config.num_hidden_layers))
        elif self.layer_aggregation_method == "varan":
            self.varan_weights_predictor = AttentionDistributionPredictor(
                hid_dim=config.hidden_size, num_heads=self.varan_attention_num_heads
            )
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WavLMForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        self.vocab_size = config.vocab_size
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.projector = (
            nn.Linear(output_hidden_size, 256)
            if self.layer_aggregation_method != "varan" or not self.varan_separate_heads
            else nn.ModuleList(
                [
                    nn.Linear(output_hidden_size, 256)
                    for _ in range(config.num_hidden_layers)
                ]
            )
        )
        self.lm_head = (
            nn.Linear(256, config.vocab_size)
            if self.layer_aggregation_method != "varan" or not self.varan_separate_heads
            else nn.ModuleList(
                [
                    nn.Linear(output_hidden_size, config.vocab_size)
                    for _ in range(config.num_hidden_layers)
                ]
            )
        )
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def _get_input_length(self, attention_mask):
        """
        Takes attention mask passed for TransformerEncoder and computes non-padded sequence length
        :param attention_mask: [batch_size, seq_len, hid_dim] 1 non-masked, 0 masked
        :return: [batch_size]
        """
        assert attention_mask is not None, "Expected attention mask to be not None."
        input_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        return input_lengths

    @staticmethod
    def forward_padding_mask(
        features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        return padding_mask.all(-1)

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

    def varan_loss(
        self,
        predicted_weights_logits,
        predicted_log_probs,
        flattened_targets,
        input_lengths,
        target_lengths,
    ):
        batch_size, n_layers = predicted_weights_logits.shape
        ctc_loss = torch.zeros(
            (n_layers, batch_size), device=predicted_weights_logits.device
        )
        predicted_weight_probas = F.softmax(predicted_weights_logits, dim=-1)
        # Initialize prior distribution of shape [per_device_train_batch_size, n_layers]
        # if per_device_eval_batch_size smaller -> cut.
        target_distribution = self.prior_distribution[:batch_size, :]
        for i in range(predicted_log_probs.shape[0]):
            log_probs = predicted_log_probs[i, :, :]
            loss = (
                nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_token_id,
                    reduction="none",
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                / input_lengths
            )
            ctc_loss[i, :] = loss * predicted_weight_probas[:, i]
        assert predicted_log_probs.shape[0] == n_layers, (
            f"Must compute CTC per layer, "
            f"got len(predicted_log_probs) = {predicted_log_probs.shape[0]}"
        )
        # Sum per layer, average per sample
        ctc_loss = ctc_loss.sum(dim=0)
        assert (
            len(ctc_loss) == batch_size
        ), f"Expected to sum ce loss per layer before averaging, got: {len(ctc_loss)}"
        ctc_loss = torch.mean(ctc_loss)
        kl_loss = F.kl_div(
            torch.log_softmax(predicted_weights_logits, dim=-1),
            target_distribution,
            reduction="batchmean",
        )
        total_loss = self.kl_beta * kl_loss + ctc_loss
        return total_loss, kl_loss, ctc_loss, None, None

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[ASRVaranOutput, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state, cnn_extract_features, hidden_states = (
            outputs[0],
            outputs[1],
            outputs[2],
        )
        if self.layer_aggregation_method == "last_layer":
            asr_head_input = last_hidden_state
        elif self.layer_aggregation_method == "weighted_sum":
            weight_probas = F.softmax(self.weights, dim=0)
            concatenated_hidden_states = torch.stack(
                [hidden_states[i] for i in range(1, len(hidden_states))], dim=0
            )
            _, *origin_shape = concatenated_hidden_states.shape
            weighted_hiddens_sum = (
                weight_probas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * concatenated_hidden_states
            ).sum(dim=0)
            asr_head_input = weighted_hiddens_sum.view(*origin_shape)
        elif self.layer_aggregation_method == "varan":
            layer_distribution_logits = self.predict_posterior_distribution(
                attention_mask, hidden_states
            )
            batch_size, seq_len, hidden_dim = hidden_states[0].shape
            n_layers = len(hidden_states) - 1
            asr_head_input = torch.zeros(
                (n_layers, batch_size, seq_len, hidden_dim),
                device=hidden_states[0].device,
            )
            for i in range(n_layers):
                asr_head_input[i, :, :, :] = hidden_states[i + 1]

        else:
            assert (
                False
            ), "Expected 'layer_aggregation_method' be one of: [last_layer, weighted_sum, varan]."
        if self.layer_aggregation_method == "varan":
            n_layers = asr_head_input.size(0)
            batch_size, seq_len, hidden_dim = asr_head_input[0].shape
            logits = torch.zeros(
                (n_layers, batch_size, seq_len, self.vocab_size),
                device=asr_head_input.device,
            )
            for i in range(n_layers):
                if self.varan_separate_heads:
                    logits[i, :, :, :] = self.lm_head[i](
                        self.projector[i](asr_head_input[i, :, :, :])
                    )
                else:
                    logits[i, :, :, :] = self.lm_head(
                        self.projector(asr_head_input[i, :, :, :])
                    )
        else:
            logits = self.lm_head(self.projector(asr_head_input))

        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            input_lengths = self._get_input_length(attention_mask)
            # assuming that padded tokens are -100
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            if self.layer_aggregation_method == "varan":
                # ctc_loss doesn't support fp16
                log_probs = torch.empty(
                    (n_layers, seq_len, batch_size, self.vocab_size),
                    device=logits.device,
                )
                for i in range(n_layers):
                    log_probs[i] = F.log_softmax(
                        logits[i], dim=-1, dtype=torch.float32
                    ).transpose(0, 1)
                # compute varan loss
                (
                    total_loss,
                    kl_loss,
                    ctc_loss,
                    layerwise_ctc_loss,
                    layerwise_probability_weight,
                ) = self.varan_loss(
                    predicted_weights_logits=layer_distribution_logits,
                    predicted_log_probs=log_probs,
                    flattened_targets=labels,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                )
                return ASRVaranOutput(
                    loss=total_loss,
                    ctc_loss=ctc_loss.detach().cpu(),
                    kl_loss=kl_loss.detach().cpu(),
                    logits=logits,
                    layer_distribution_logits=layer_distribution_logits.detach().cpu(),
                    hidden_states=outputs.hidden_states,
                    input_lengths=input_lengths,
                )

            else:
                # ctc_loss doesn't support fp16
                log_probs = F.log_softmax(
                    logits, dim=-1, dtype=torch.float32
                ).transpose(0, 1)
                total_loss = F.ctc_loss(
                    log_probs,
                    labels,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                return ASRVaranOutput(
                    loss=total_loss,
                    ctc_loss=None,
                    kl_loss=None,
                    logits=logits,
                    layer_distribution_logits=None,
                    hidden_states=None,
                    input_lengths=input_lengths,
                )
