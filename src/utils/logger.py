from logging.handlers import RotatingFileHandler
from pathlib import Path
import logging

def setup_logging(dest_path: str, filename: str,  level = logging.INFO) -> None:
    """
    This function sets up the global logging configurations.

    Args:
        dest_path (str): 
            Destination path (including parent directories and file name)
        level (logging._Level, optional): 
            Logging level. Defaults to logging.INFO.
    """
    dest_dir = Path(dest_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    full_dest_path = dest_dir / filename
    
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
        
    # Setting up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s %(message)s")
    )
    
    file_handler = RotatingFileHandler(full_dest_path, mode="a", maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s")
    )
    
    logging.basicConfig(
        handlers=[file_handler, console_handler],
        style="%",
        datefmt="%Y-%d-%d %H:%M",
        level=level  
    )
