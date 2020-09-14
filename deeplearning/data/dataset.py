from torch.utils.data import Dataset
import numpy as np
import cv2
import tables
import os
import pathlib
import PIL.Image as Image
import pandas as pd
import torch

# loads preconverted label and images h5 files pointed to in config.json. This will not load all the data into
# memeory and instead only loads requested data into memory
from torchvision import datasets, transforms
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import tables


# dataset loading the etl extracted h5 files to be consumable
class PreConvertedChestXpertDataSet(Dataset):
    def __init__(self, data_set_config, mode='train'):
        self.cfg = data_set_config
        self.label_data = tables.open_file(self.cfg["label_npy"], mode='r')
        self.image_data = tables.open_file(self.cfg["image_npy"], mode='r')
        self.labels = ['No Finding',
                       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                       'Support Devices']


        self.UOnes_mapping = {1: 1, "nan": 0, 0: 0, -1: 1}  # uncertain label is treated as true label, no record is treated as 0
        self.UZeros_mapping = {1: 1, "nan": 0, 0: 0, -1: 0}  # uncertain label is treated as true label, no record is treated as 0

        self.vectorized_label_transform = np.vectorize(
            lambda label: self.UOnes_mapping[label] if not np.isnan(label) else self.UOnes_mapping["nan"])

    def __getitem__(self, idx):

        return torch.from_numpy(self.image_data.root.image[idx]).float(), torch.from_numpy(
            self.vectorized_label_transform(self.label_data.root.labels[idx]))

    def __len__(self):
        return self.image_data.root.image.shape[0]


class PetastormDataSet(Dataset):
    def __init__(self, data_set_config, mode='train'):
        self.cfg = data_set_config
        data_dir = self.cfg['data_url']
        transform = TransformSpec(self._transform_row,
                                  removed_fields=['Path', 'origin', 'pid', 'Sex', 'Age', 'Frontal_Lateral', 'AP_PA',
                                                  'No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly',
                                                  'Lung_Opacity',
                                                  'Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                                  'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other', 'Fracture',
                                                  'Support_Devices'])
        print("About to call make_reader")
        with make_reader("hdfs://bootcamp.local:9000/user/local/output/", transform_spec=transform,
                         hdfs_driver='libhdfs') as reader:
            rows = list(reader)
        print("About to call reader.stop")
        reader.stop()
        reader.join()

        self.label_data = rows
        self.image_data = rows

    def __getitem__(self, idx):
        return self.image_data[idx][1].float(), torch.from_numpy(self.label_data[idx][0])

    def __len__(self):
        return len(self.image_data)

    def _transform_row(self, input_row):
        transform = transforms.Compose([
            transforms.ToTensor()  # Read Numpy as as W*H*C, converted to C*H*W via ToTensor function
            #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # In addition, the petastorm pytorch DataLoader does not distinguish the notion of
        # data or target transform, but that actually gives the user more flexibility.
        result_row = {
            'Path': input_row['Path'],
            'origin': input_row['origin'],
            'pid': input_row['pid'],
            'Sex': input_row['Sex'],
            'Age': input_row['Age'],
            'Frontal_Lateral': input_row['Frontal_Lateral'],
            'AP_PA': input_row['AP_PA'],
            'No_Finding': input_row['No_Finding'],
            'Enlarged_Cardiomediastinum': input_row['Enlarged_Cardiomediastinum'],
            'Cardiomegaly': input_row['Cardiomegaly'],
            'Lung_Opacity': input_row['Lung_Opacity'],
            'Lung_Lesion': input_row['Lung_Lesion'],
            'Edema': input_row['Edema'],
            'Consolidation': input_row['Consolidation'],
            'Pneumonia': input_row['Pneumonia'],
            'Atelectasis': input_row['Atelectasis'],
            'Pneumothorax': input_row['Pneumothorax'],
            'Pleural_Effusion': input_row['Pleural_Effusion'],
            'Pleural_Other': input_row['Pleural_Other'],
            'Fracture': input_row['Fracture'],
            'Support_Devices': input_row['Support_Devices'],
            'Labels': input_row['Labels'],
            'Resized_np': transform(input_row['Resized_np'])
        }
        return result_row


class PetastormDataSet_hdf5(Dataset):
    def __init__(self, data_set_config, mode='train'):
        self.cfg = data_set_config
        data_dir = self.cfg['data_url']
        transform = TransformSpec(self._transform_row,
                                  removed_fields=['Path', 'origin', 'pid', 'Sex', 'Age', 'Frontal_Lateral', 'AP_PA',
                                                  'No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly',
                                                  'Lung_Opacity',
                                                  'Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                                  'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other', 'Fracture',
                                                  'Support_Devices'])

        labels_file_path = os.path.join("../sample_outputs/dl_data/labels.h5")
        image_file_path = os.path.join("../sample_outputs/dl_data/image.h5")
        pathlib.Path("../sample_outputs/dl_data/").mkdir(parents=True, exist_ok=True)

        # If the hdf5 files already exist, skip the loading of the ETL results from the data_url
        # as the ETL results are essentially used to create the hdf5 files which are then actually used
        # for data indexing and access and etc. during training
        if os.path.exists(labels_file_path) and os.path.exists(image_file_path):
            print("labels and image hdf5 files already exists", flush=True)
            print("Skipping loading of ETL results to create these hdf5 files as they already exist.", flush=True)
        else:
            labels_file = tables.open_file(labels_file_path, mode='w')
            image_file = tables.open_file(image_file_path, mode='w')
            labels_atom = tables.Atom.from_dtype(np.dtype(np.float32, (0, 14)))
            image_atom = tables.Atom.from_dtype(np.dtype(np.float64, (0, 3, 512, 512)))
            labels_array = labels_file.create_earray(labels_file.root, "labels", labels_atom, (0, 14))
            image_array = image_file.create_earray(image_file.root, "image", image_atom, (0, 3, 512, 512))

            with make_reader(data_dir, transform_spec=transform, hdfs_driver='libhdfs') as reader:
                for row in reader:
                    image_np = np.moveaxis(row[1], -1, 0)  # convert the numpy from (H, W,C) to (C, H, W) because this is the shape the model expect
                    image_array.append(np.expand_dims(image_np, 0))
                    labels_array.append(np.expand_dims(row[0], 0))  # expanding dims here because it expect the shape [1, (input shape)]
                reader.stop()
                reader.join()
            labels_file.close()
            image_file.close()

        self.label_data = tables.open_file(labels_file_path, mode='r')
        self.image_data = tables.open_file(image_file_path, mode='r')

    def __getitem__(self, idx):
        return torch.from_numpy(self.image_data.root.image[idx]).float(), torch.from_numpy(self.label_data.root.labels[idx])

    def __len__(self):
        return self.image_data.root.image.shape[0]

    def _transform_row(self, input_row):
        transform = transforms.Compose([
            transforms.ToTensor()  # Read Numpy as as W*H*C, converted to C*H*W via ToTensor function
            #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # In addition, the petastorm pytorch DataLoader does not distinguish the notion of
        # data or target transform, but that actually gives the user more flexibility.
        result_row = {
            'Path': input_row['Path'],
            'origin': input_row['origin'],
            'pid': input_row['pid'],
            'Sex': input_row['Sex'],
            'Age': input_row['Age'],
            'Frontal_Lateral': input_row['Frontal_Lateral'],
            'AP_PA': input_row['AP_PA'],
            'No_Finding': input_row['No_Finding'],
            'Enlarged_Cardiomediastinum': input_row['Enlarged_Cardiomediastinum'],
            'Cardiomegaly': input_row['Cardiomegaly'],
            'Lung_Opacity': input_row['Lung_Opacity'],
            'Lung_Lesion': input_row['Lung_Lesion'],
            'Edema': input_row['Edema'],
            'Consolidation': input_row['Consolidation'],
            'Pneumonia': input_row['Pneumonia'],
            'Atelectasis': input_row['Atelectasis'],
            'Pneumothorax': input_row['Pneumothorax'],
            'Pleural_Effusion': input_row['Pleural_Effusion'],
            'Pleural_Other': input_row['Pleural_Other'],
            'Fracture': input_row['Fracture'],
            'Support_Devices': input_row['Support_Devices'],
            'Labels': input_row['Labels'],
            'Resized_np': input_row['Resized_np']
        }
        return result_row



if __name__ == "__main__":
    dt = PreConvertedChestXpertDataSet({
        "label_npy": "/bdh-spring-2020-project-CheXpert/sample_outputs/data/labels.h5",
        "image_npy": "/bdh-spring-2020-project-CheXpert/sample_outputs/data/image.h5"
    })
    for index in range(len(dt)):
        img, label = dt[index]
        img = img.numpy().astype(np.uint8)
        img = np.moveaxis(img, 0, -1)

        print(img.shape)
        print(label.shape)
        print(len(dt))
        # print (label)
        cv2.imshow("test_img", img)
        cv2.waitKey(0)
