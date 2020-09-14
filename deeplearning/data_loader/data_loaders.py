from torchvision import datasets, transforms
from deeplearning.base import BaseDataLoader
from deeplearning.data.dataset import PreConvertedChestXpertDataSet, PetastormDataSet, PetastormDataSet_hdf5

# Dataloader for various purpose.


# this base loader loads the label and images of preconverted data
class PreConvertedChestXpertDataLoader(BaseDataLoader):
    def __init__(self, data_set_config, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = PreConvertedChestXpertDataSet(data_set_config, 'train' if training else 'test')
        super(PreConvertedChestXpertDataLoader, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split, num_workers=num_workers)

# this base loader loads the label and images from petastorm parquet files
class PetastormDataLoader_hdf5(BaseDataLoader):
    def __init__(self, data_set_config, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = PetastormDataSet_hdf5(data_set_config, 'train' if training else 'test')
        super(PetastormDataLoader_hdf5, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split, num_workers=num_workers)

# this base loader loads the label and images from petastorm parquet files, converts it ot h5 and load it.
class PetastormDataLoader(BaseDataLoader):
    def __init__(self, data_set_config, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = PetastormDataSet(data_set_config, 'train' if training else 'test')
        super(PetastormDataLoader, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split, num_workers=num_workers)