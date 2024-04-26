import importlib

DATASET = {
        'image_datasets': 'ImageDataset',
        }

def create_dataset(dataset_type, dataset_path, **kwargs):

    module = importlib.import_module(f'src.datasets.{dataset_type}')
    return getattr(module, DATASET[dataset_type])(dataset_path, **kwargs), module.collate_fn
