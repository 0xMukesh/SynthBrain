from torch.utils.data import Dataset
from PIL import Image
import os

from src.constants import (
    BRICS_CLASSES,
    BRICS_SHORT_CODE_MAPPING,
    BRICS_CLASS_IDX_MAPPING,
)


class BRISCDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()

        self.dataset_path = f"{root}/{split}"
        self.classes = BRICS_CLASSES
        self.transform = transform

        self.files = []

        for c in self.classes:
            class_files = os.listdir(os.path.join(self.dataset_path, c))

            for cf in class_files:
                if "ax_t1" in cf:
                    self.files.append(os.path.join(self.dataset_path, c, cf))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        img = Image.open(file_path)
        label = BRICS_SHORT_CODE_MAPPING[os.path.basename(file_path).split("_")[-3]]

        if self.transform:
            img = self.transform(img)

        return img, BRICS_CLASS_IDX_MAPPING[label]
