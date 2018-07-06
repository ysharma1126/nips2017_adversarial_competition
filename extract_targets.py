
from run_attacks_and_defenses import DatasetMetadata

data = DatasetMetadata('dataset/dev_dataset.csv')
data.save_target_classes('dataset/images/target_class.csv')
