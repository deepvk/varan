from typing import Dict

import torch
from datasets import DatasetDict, load_dataset

LABELS = {
    "xbgoose/ravdess": {
        "angry": 0,
        "happy": 1,
        "calm": 2,
        "sad": 3,
        "fearful": 4,
        "disgust": 5,
        "neutral": 2,  # merging neutral and calm
        "surprised": 6,
    },
    "Ar4ikov/iemocap_audio_text": {
        "ang": 0,  # 0
        "hap": 1,  # 1
        "exc": 2,  # 1
        "sad": 3,  # 2
        "fru": -1,  # -
        "fea": -1,  # -
        "sur": -1,  # -
        "neu": 3,  # 3
        "xxx": -1,  # -
        "oth": -1,  # -
        "dis": -1,  # -
    },
}


def get_labels2idx(dataset_name: str):
    return LABELS[dataset_name]


def collate_function(examples, feature_extractor, pad_to_miltiple_of: int = 16):
    audio_arrays = [x["input_values"]["array"] for x in examples]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
        padding="longest",
        # max_length=feature_extractor.sampling_rate * 10,
        # truncation=True,
        pad_to_multiple_of=pad_to_miltiple_of,
    )
    inputs["labels"] = torch.tensor([x["label"] for x in examples])
    return inputs


def get_iemocap_dataset(
    name_or_path: str,
    label2idx: Dict[str, int],
    audio_column: str = "audio",
    label_column: str = "emotion",
    test_split: str = "05M",
):
    dataset = load_dataset(name_or_path)["train"]
    dataset = dataset.remove_columns(
        list(set(dataset.column_names) - set([audio_column, "titre", label_column]))
    )
    dataset = dataset.map(
        lambda x: {"label": label2idx[x[label_column]]}, remove_columns=label_column
    ).filter(lambda x: x["label"] > 0)

    def get_split_from_titre(titre: str):
        return titre[3:6]

    dataset = dataset.map(
        lambda x: {"split": get_split_from_titre(titre=x["titre"])},
        remove_columns=["titre"],
    )
    dataset = dataset.rename_column(audio_column, "input_values")
    dataset_splitted = DatasetDict(
        {
            "train": dataset.filter(lambda x: x["split"] != test_split),
            "eval": dataset.filter(lambda x: x["split"] == test_split),
        }
    )
    return dataset_splitted


def get_classification_dataset(
    name_or_path: str,
    label2idx: Dict[str, int],
    audio_column: str = "audio",
    label_column: str = "emotion",
):
    dataset = load_dataset(name_or_path)
    dataset_splitted = DatasetDict(
        {
            "train": dataset.filter(lambda x: x["actor"] in [i for i in range(20)])[
                "train"
            ],
            "validation": dataset.filter(
                lambda x: x["actor"] in [i for i in range(20, 22)]
            )["train"],
            "test": dataset.filter(lambda x: x["actor"] in [i for i in range(22, 24)])[
                "train"
            ],
        }
    )
    columns_to_remove = list(
        set(dataset_splitted["train"].column_names) - set((audio_column, "label"))
    )
    dataset = dataset_splitted.map(
        lambda x: {"label": label2idx[x[label_column]]},
        remove_columns=columns_to_remove,
    )
    dataset = dataset.rename_column(audio_column, "input_values")
    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "eval": DatasetDict(
                {"validation": dataset["validation"], "test": dataset["test"]}
            ),
        }
    )
    return dataset
