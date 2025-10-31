from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import shutil

logging = logging.getLogger(__name__)

class DataSplitter:
    """
    This class abstracts the splitting of the dataset into train, val and test sets
    
    Attributes:
        config (dict):
            Global configuration
            
    Methods:
        get_splits(self, output_path: str) -> None:
            This method takes the source destination as an argument, splits the data and creates
            three subdirectories (train, val, test) and copies the respective files into their
            corresponding folders.
    
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_dir = self.config["data"]["data_dir"]
        self.root = Path(self.data_dir)
        self.classes = [dir.name for dir in self.root.iterdir() if dir.is_dir()]
        self.seed = self.config["seed"]
        
    def get_splits(self, output_path: str) -> None:
        """
        Splits the dataset into train, val and test sets

        Args:
            output_path (str): 
                Specifies the output dir
        """
        
        logging.info(f"Splitting dataset {self.data_dir} into training, validation and test sets.")
        output_dir = Path(output_path)
        for cls in self.classes:
            images = list((self.root / cls).glob("*"))
            train, tmp = train_test_split(images, test_size=0.25, random_state=self.seed)
            val, test = train_test_split(tmp, test_size=0.5, random_state=self.seed)
            
            for subset, subset_files in zip(["train", "val", "test"], [train, val, test]):
                out_dir = output_dir / subset / cls
                output_dir.mkdir(parents=True, exist_ok=True)
                for file in subset_files:
                    shutil.copy(file, out_dir / file.name)
        logging.info("Splitting operation completed.")