from typing import Any
from torch.utils.data import Dataset

class NeRFSynthesicDataset(Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()
        
    def __len__(self) -> Any:
        pass
    
    def __getitem__(self, index: Any) -> Any:
        return super().__getitem__(index)
    
def get_dataset(cfg, data_path):
    pass