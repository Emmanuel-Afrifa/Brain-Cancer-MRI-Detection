from src.data.data_splitter import DataSplitter
import os
import shutil

config = {
    "data": {
        "data_dir": ""
    },
    "seed":
        27
}

def test_data_splitter(tmp_path):
    base_dir = tmp_path / "fake_data"
    class_dir = [base_dir / f"class_{i}" for i in range(3)]
    config["data"]["data_dir"] = base_dir
    
    for d in class_dir:
        d.mkdir(parents=True)
        for j in range(9):
            (d / f"img_{j}.png").write_text("fake_image")
            
    output_dir = tmp_path / "splits"
    DataSplitter(config=config).get_splits(output_path=output_dir)
    
    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()
    assert (output_dir / "test").exists()
    
    for split in ["train", "val", "test"]:
        for i in range(3):
            assert (output_dir / split / f"class_{i}").exists()
            
    total_imgs = sum(len(os.listdir(d)) for d in class_dir)
    total_splits = sum(len(os.listdir(output_dir / split / f"class_{i}")) for split in ["train", "val", "test"] for i in range(3))
    
    assert total_imgs == total_splits