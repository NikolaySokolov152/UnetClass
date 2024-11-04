import sys
sys.path.append("src/")

from src.dataGenerator import DataGeneratorReaderAll
from trainer import generator_parcer
import json

is_silence_mode=False
save_path = "numpy_dataset"
path_config = "segmentation/config_test.json"

print("parse")

with open(path_config) as config_buffer:
    print("open config")
    config = json.loads(config_buffer.read())
config["train"]["num_class"] = 6

generator = generator_parcer(config, is_silence_mode)
generator.saveNpyData(save_path)