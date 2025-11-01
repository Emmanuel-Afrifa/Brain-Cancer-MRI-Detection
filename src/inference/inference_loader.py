from torch.utils.data import DataLoader, Dataset

class InferenceDataLoader:
    """
    This class abstracts the creation of data loaders for the inference dataset
    
    Attributes:
        dataset(Dataset):
            dataset whose data loaders are to be created.
        config (dict):
            Global configurations
            
    Methods:
        get_inference_loaders(self):
            Returns the data loaders for the `dataset`.
    """
    def __init__(self, dataset: Dataset, config: dict) -> None:
        self.dataset = dataset
        self.config = config
        self.batch_size = self.config["train"]["batch_size"]
        
    def get_inference_loaders(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader