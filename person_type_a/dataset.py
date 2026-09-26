"""CIFAR-10 → 字母槽样本适配器（含顺序增广与四路切分）。"""
from __future__ import annotations

import random

from .encoding import assign_letters
from .schema import ABSTAIN_LABEL, QuestionSpec, ScenarioConfig

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

CIFAR10_CRITERIA = [
    "aircraft in sky or on runway", "car, truck-like vehicle",
    "bird", "small pet animal", "wild animal with antlers",
    "domestic pet animal", "amphibian", "large riding animal",
    "water vessel", "wheeled vehicle", "cannot determine",
]


def make_cifar10_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="cifar10",
        system="You are an image classifier. Output only the option letter.",
        scene="CIFAR-10 image classification",
        evidence=("object shape", "color", "texture", "background context"),
        questions=(QuestionSpec(
            qid="object", kind="choice",
            instructions="What object is shown in this image?",
            options=tuple(CIFAR10_CLASSES),
            criteria=tuple(CIFAR10_CRITERIA),
        ),),
    )


class CIFAR10LetterSlot:
    """CIFAR-10 数据集的字母槽包装（惰性加载 torchvision）。"""

    SPLITS = {"train": (0, 40000), "calibration": (40000, 45000),
              "val": (45000, 50000), "test": (50000, 60000)}

    def __init__(self, root: str = "./data", download: bool = True,
                 split: str = "train", augment_order: bool = True, seed: int = 42):
        from torchvision.datasets import CIFAR10
        self.ds = CIFAR10(root=root, train=True, download=download)
        lo, hi = self.SPLITS[split]
        self._indices = list(range(lo, hi))
        self.augment_order = augment_order
        self.seed = seed
        self.classes = CIFAR10_CLASSES

    def __len__(self):
        return len(self._indices)

    def _make_meta(self, label: int, epoch_seed: int = 0) -> dict:
        rng = random.Random((label, epoch_seed, self.seed) if self.augment_order
                            else (self.seed,))
        perm = list(range(len(self.classes)))
        rng.shuffle(perm)
        ordered = [self.classes[i] for i in perm] + [ABSTAIN_LABEL]
        letters = assign_letters(tuple(ordered))
        gold_idx = ordered.index(self.classes[label])
        return {"ordered": ordered, "letters": letters, "gold_idx": gold_idx}

    def __getitem__(self, idx: int):
        img, label = self.ds[self._indices[idx]]
        meta = self._make_meta(label, epoch_seed=idx)
        return img, label, meta
