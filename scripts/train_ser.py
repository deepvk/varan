import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Optional

import evaluate
import numpy as np
import wandb
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification, HfArgumentParser,
                          Trainer, TrainingArguments, set_seed)

from src.data.ser_dataset import (collate_function, get_classification_dataset,
                                  get_iemocap_dataset, get_labels2idx)
from src.models.data2vec_for_sequence_classification import \
    VaranData2VecForSequenceClassification
from src.models.varan import PriorDistributionConstructor
from src.models.wavlm_for_sequence_classification import \
    VaranWavLMForSequenceClassification


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    upstream_model: str = field(
        default="wavlm", metadata={"help": "Upstream model name"}
    )
    num_labels: int = field(
        default=7, metadata={"help": "Number of classes for the classification."}
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout applied to Upstream model output before ASR head."},
    )
    layer_aggregation_method: str = field(
        default="last_layer",
        metadata={
            "help": "How to aggregate output and path it to the LM head."
            " Possible ways: last_layer, weighted_sum, varan."
        },
    )
    varan_attention_num_heads: int = field(
        default=16,
        metadata={
            "help": "Number of Attention heads of VARAN weights prediction model"
        },
    )
    varan_separate_heads: bool = field(
        default=True,
        metadata={"help": "Same or different classifiers for differen Varan Layers"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    feat_proj_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the projected features."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    mask_time_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    use_separate_classifiers: bool = field(default=False)
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    wandb_api_key: str = field(metadata={"help": "API key for wandb login"})

    dataset_name: str = field(
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )

    wandb_project_name: str = field(
        default="ser",
        metadata={"help": "wandb project name"},
    )
    wandb_mode: str = field(
        default="online", metadata={"help": "log to wandb online offline or disabled"}
    )
    test_split_iemocap: str = field(
        default="05M",
        metadata={
            "help": "which split will be used for validation for iemocap dataset"
        },
    )

    train_split_name: Optional[List[str]] = list_field(
        default=["train"],
        metadata={
            "help": (
                "The name of sets used for training. Separate sets mentioned in the field will be concatenated "
                "Available splits for LibriSpeech: train.clean.100, train.clean.360, "
                "train.clean.500. Default is ['train.clean.100']"
            )
        },
    )
    eval_split_name: Optional[List[str]] = list_field(
        default=["validation"],
        metadata={
            "help": (
                "The name of sets used for validation. Separate sets mentioned in the field will be concatenated "
                "Available splits for LibriSpeech: validation.clean, validation.other, "
                "Default is ['validation.clean', 'validation.other']"
            )
        },
    )
    test_split_name: Optional[List[str]] = list_field(
        default=["test"],
        metadata={
            "help": (
                "The name of sets used for test. Separate test sets won't be concatenated "
                "Available splits for LibriSpeech: test.clean, test.other, "
                "Default is ['test.clean', 'test.other']"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    label_column_name: str = field(
        default="emotion",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    streaming: bool = field(
        default=False, metadata={"help": "If True load Iterable dataset else Map."}
    )
    timeout_limit: int = field(
        default=3600,
        metadata={"help": "Number of seconds to wait on the data loading side"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )

    eval_metrics: List[str] = list_field(
        default=["accuracy"],
        metadata={
            "help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )

    use_lora: bool = field(
        default=True,
        metadata={"help": ("Use Lora decomposition for fine-tuning")},
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": ("Only usef if use Lora is True, use modification Dora then")
        },
    )

    r: int = field(
        default=16,
        metadata={"help": ("Rank: Lora dimension")},
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": ("Rank: The alpha parameter for Lora scaling.")},
    )

    lora_alpha_equals_r: bool = field(
        default=True,
        metadata={
            "help": ("If True will set lora_alpha parameter equals to lora rank.")
        },
    )

    target_modules: Optional[List[str]] = list_field(
        default=["intermediate_dense", "output_dense"],
        metadata={"help": ("The names of the modules to apply the adapter to. ")},
    )

    modules_to_save: Optional[List[str]] = list_field(
        default=[],
        metadata={
            "help": (
                "List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint. "
            )
        },
    )


@dataclass
class DistributionArguments:
    """
    Arguments for initializing prior distribution for VARAN layer aggregation method.
    """

    prior_distribution: str = field(
        default="chi2_reversed",
        metadata={"help": "uniform, geometric, geometric_reversed or chi2"},
    )
    p_geometric_pmf: float = field(
        default=0.45,
        metadata={"help": "higher values -> heavier head or tail (if reversed)"},
    )
    chi2_df: float = field(default=2, metadata={"help": "Degrees of freedom"})
    chi2_nc: float = field(
        default=10, metadata={"help": "Higher values make tail heavier."}
    )
    kl_beta: float = field(
        default=0.05,
        metadata={"help": "A coefficient in front of KL loss between distributions."},
    )


def count_trainable_parameters(model):
    model_parameters_trainable = filter(lambda p: p.requires_grad, model.parameters())
    params_trainable = sum([np.prod(p.size()) for p in model_parameters_trainable])
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    return f"Trainable parameters: {params_trainable} / total parameters {total_params}"


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            DistributionArguments,
        )
    )
    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    run_name_specified = False
    if config_file:
        config_args = parser.parse_json_file(json_file=os.path.abspath(config_file))
        raw_config_json = json.loads(Path(config_file).read_text(encoding="utf-8"))

        config_arg_idx = sys.argv.index(config_file)
        other_args = sys.argv[config_arg_idx + 1 :]
        print(other_args)
        eq_filtered_args = []
        for arg in other_args:
            if "=" in arg:
                eq_filtered_args += arg.split("=")
            else:
                eq_filtered_args += [arg]
        other_args = eq_filtered_args
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}
        if "run_name" in arg_names or "run_name" in raw_config_json:
            run_name_specified = True

        required_args = [
            (act.option_strings[0], "dummy")
            for act in parser._actions
            if act.required
            and not any(act_s[2:] in arg_names for act_s in act.option_strings)
        ]
        required_args = [
            arg for req_dummy_args in required_args for arg in req_dummy_args
        ]  # Flatten
        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(
            args=cli_args, look_for_args_file=False
        )
        all_args = []
        print(f"Args to be replaced: {arg_names}")
        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Have to check explicitly for no_ for the automatically added negated boolean arguments
            # E.g. find_unused... vs no_find_unused...
            cli_d = {
                k: v
                for k, v in dataclasses.asdict(cli_dc).items()
                if k in arg_names or f"no_{k}" in arg_names
            }
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))
        model_args, data_args, training_args, distribution_args = all_args
    else:
        model_args, data_args, training_args, distribution_args = (
            parser.parse_args_into_dataclasses()
        )
    data_args: DataTrainingArguments
    model_args: ModelArguments
    training_args: TrainingArguments
    distribution_args: DistributionArguments

    with training_args.main_process_first(
        desc="dataset map special characters removal"
    ):
        if "iemocap" in data_args.dataset_name:
            dataset = get_iemocap_dataset(
                data_args.dataset_name,
                label2idx=get_labels2idx(data_args.dataset_name),
                audio_column=data_args.audio_column_name,
                label_column=data_args.label_column_name,
                test_split=data_args.test_split_iemocap,
            )
        else:
            dataset = get_classification_dataset(
                data_args.dataset_name,
                label2idx=get_labels2idx(data_args.dataset_name),
                audio_column=data_args.audio_column_name,
                label_column=data_args.label_column_name,
            )
    if training_args.local_rank == 0:
        print(dataset)
    # Set seed before initializing model.
    set_seed(training_args.seed + training_args.local_rank)

    def get_model():
        if model_args.layer_aggregation_method == "varan":
            if training_args.local_rank == 0:
                print(
                    f"Varunning with {'separate classifiers' if model_args.use_separate_classifiers else 'a single classifier'}... 🦎🦎🦎"
                )

            if "wavlm" in model_args.model_name_or_path:
                upstream_model = VaranWavLMForSequenceClassification
            elif "data2vec" in model_args.model_name_or_path:
                upstream_model = VaranData2VecForSequenceClassification
            else:
                raise AttributeError(
                    f"Expected upstream_model be either 'wavlm' or 'data2vec'"
                )

            model = upstream_model.from_pretrained(
                model_args.model_name_or_path,
                varan_attention_num_heads=model_args.varan_attention_num_heads,
                kl_beta=distribution_args.kl_beta,
                num_labels=model_args.num_labels,
            )
            if (
                model_args.layer_aggregation_method == "varan"
                and model_args.use_separate_classifiers
            ):
                print(f"Reinitialized separate classifiers")
                model.use_separate_classifiers()
        else:
            model = AutoModelForAudioClassification.from_pretrained(
                model_args.model_name_or_path,
                num_labels=model_args.num_labels,
                use_weighted_layer_sum=model_args.layer_aggregation_method
                == "weighted_sum",
            )
        model.freeze_feature_encoder()
        return model

    model = get_model()

    # Apply LoRa or DoRa if specified

    if data_args.use_lora:
        # if use LoRa do not load best model at end
        training_args.load_best_model_at_end = False

        if (
            model_args.layer_aggregation_method == "varan"
            and model_args.use_separate_classifiers
        ):
            classifier_names = [
                f"classifier.{name}" for name, _ in model.classifier.named_modules()
            ]
            projector_names = [
                f"projector.{name}" for name, _ in model.projector.named_modules()
            ]
            posterior_predictor_names = ["posterior_predictor"]

            data_args.modules_to_save.extend(classifier_names)
            data_args.modules_to_save.extend(projector_names)
            data_args.modules_to_save.extend(posterior_predictor_names)
        elif model_args.layer_aggregation_method == "varan":
            classifier_names = ["classifier"]
            projector_names = ["projector"]
            data_args.modules_to_save.extend(
                classifier_names + projector_names + ["posterior_predictor"]
            )
        elif model_args.layer_aggregation_method == "last_layer":
            classifier_names = ["classifier"]
            projector_names = ["projector"]
            data_args.modules_to_save.extend(classifier_names + projector_names)
        elif model_args.layer_aggregation_method == "weighted_sum":
            classifier_names = ["classifier"]
            projector_names = ["projector"]
            layer_weights = ["layer_weights"]
            data_args.modules_to_save.extend(
                classifier_names + projector_names + layer_weights
            )

        print("Modules except LoRa that will be trained:", data_args.modules_to_save)

        if data_args.lora_alpha_equals_r:
            data_args.lora_alpha = data_args.r

        lora_config = LoraConfig(
            r=data_args.r,
            target_modules=data_args.target_modules,
            modules_to_save=data_args.modules_to_save,
            lora_alpha=data_args.lora_alpha,
        )
        # model = get_peft_model(model, lora_config)
        model.add_adapter(lora_config)
        model.enable_adapters()
        print(count_trainable_parameters(model))
        # model.print_trainable_parameters()

    # Define prior distribution (used if trained with VARAN layer aggregation method)
    prior_distribution_constructor = PriorDistributionConstructor(
        distribution_args,
        batch_size=training_args.per_device_train_batch_size,
        encoder_layers=model.config.num_hidden_layers,
    )
    try:
        prior_distribution = prior_distribution_constructor().to(
            "cpu" if training_args.use_cpu else "cuda"
        )
    except:
        prior_distribution = prior_distribution_constructor().to("mps")
    if model_args.layer_aggregation_method == "varan":
        model.set_prior_distribution(prior_distribution)

    # Load the accuracy metric from the datasets package
    metrics = evaluate.combine(["accuracy", "hyperml/balanced_accuracy"])
    metrics_to_average = evaluate.combine(["f1", "precision", "recall"])

    # print(model(torch.from_numpy(dataset['train'][0]['input_values']['array']).unsqueeze(0)))

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        try:
            predictions = np.argmax(eval_pred.predictions, axis=1)
        except ValueError:
            predictions = np.argmax(eval_pred.predictions[0], axis=1)

        return {
            **{
                f"macro_{k}": v
                for k, v in metrics_to_average.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                    average="macro",
                ).items()
            },
            **{
                f"weighted_{k}": v
                for k, v in metrics_to_average.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                    average="weighted",
                ).items()
            },
            **metrics.compute(predictions=predictions, references=eval_pred.label_ids),
        }

    data_collator = partial(
        collate_function,
        feature_extractor=AutoFeatureExtractor.from_pretrained(
            model_args.model_name_or_path
        ),
    )

    wandb.login(key=data_args.wandb_api_key)
    wandb.init(
        project=data_args.wandb_project_name,
        name=training_args.run_name,
        mode=data_args.wandb_mode,
    )
    training_args.output_dir = training_args.output_dir + f"_{wandb.run.id}"

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["eval"] if training_args.do_eval else None,
    )

    trainer.train()

    # Custom load best model at end for LoRa
    if data_args.use_lora:
        training_args.do_train = False
        lora_trained_logs_dir = glob(f"{training_args.output_dir}/*", recursive=False)[
            0
        ]
        print(f"Print directory to load the model from: {lora_trained_logs_dir}")
        model = get_model()
        if model_args.layer_aggregation_method == "varan":
            model.set_prior_distribution(prior_distribution)
        model.load_adapter(lora_trained_logs_dir)
        if (
            model_args.layer_aggregation_method == "varan"
            and model_args.use_separate_classifiers
        ):
            with open(lora_trained_logs_dir + "/adapter_model.safetensors", "rb") as f:
                b = f.read()
            from safetensors.torch import load

            state_dict = load(b)
            projector_state_dict = {
                k: v for k, v in state_dict.items() if "projector" in k
            }
            classifier_state_dict = {
                k: v for k, v in state_dict.items() if "classifier" in k
            }
            for i, (cl_layer, pr_layer) in enumerate(
                zip(model.classifier, model.projector)
            ):
                cl_layer.modules_to_save["default"].weight.data.copy_(
                    classifier_state_dict[f"base_model.model.classifier.{i}.weight"]
                )
                cl_layer.modules_to_save["default"].bias.data.copy_(
                    classifier_state_dict[f"base_model.model.classifier.{i}.bias"]
                )

                pr_layer.modules_to_save["default"].weight.data.copy_(
                    projector_state_dict[f"base_model.model.projector.{i}.weight"]
                )
                pr_layer.modules_to_save["default"].bias.data.copy_(
                    projector_state_dict[f"base_model.model.projector.{i}.bias"]
                )

        print(f"Successfully loaded LoRa model!")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=None,
        eval_dataset=dataset["eval"] if training_args.do_eval else None,
    )

    metrics = trainer.evaluate()

    wandb.log(metrics)


if __name__ == "__main__":
    main()
