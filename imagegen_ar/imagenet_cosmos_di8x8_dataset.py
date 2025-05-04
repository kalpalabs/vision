

from typing import Any, Dict, List, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.datasets._packed import PackedDataset


class ImagenetCosmosDataset(Dataset):
    def __init__(
        self,
        column: str = "text",
    ) -> None:
        self._column = column # 'cosmos_di8x8_tokens'
        self._data = load_dataset("pshishodia/imagenet-1k-256", split='train', columns=[self._column], num_proc=64)
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        # expecting a 2d list. 
        twod_tokens = sample[self._column]
        
        # torchtune.datasets.TextCompletionDataset: No need to offset labels by 1 - happens in the recipe
        tokens = [token for row in twod_tokens for token in row]
        return {"tokens": tokens, "labels" : tokens.copy()}


def imagenet_cosmos_dataset(
    column: str = "text",
    packed: bool = False,
    split_across_pack: bool = True,
    max_seq_len: Optional[int] = None
):
    ds = ImagenetCosmosDataset(
        column=column,
    )
    if packed:
        if max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=max_seq_len, split_across_pack=split_across_pack
        )
    return ds
