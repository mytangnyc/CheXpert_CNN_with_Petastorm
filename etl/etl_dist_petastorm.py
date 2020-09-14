#!/usr/bin/env python
# coding: utf-8
"""
Import common packages including os for filesystem, pd for pandas dataframes, np for numpy arrays.
pyspark Sparksessions is used to to create a spark session.
pyspark.sql.types tells Spark the types of data being loaded into the Spark dataframe.
pyspark.sql.functions contains functions used to register a User Defined Function (UDF) applied to spark columns.
We use the PIL package to read the jpg chest x-rays and apply transformations including resizes.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType, IntegerType
import pyspark.sql.functions as F
from PIL import Image

"""
The Petastorm package is used to standardize spark data to load in a common format into the Pytorch libraries.
Petastorm has readers for Pytorch and Tensorflow allowing us flexibility to use different libraries if we want.
FileSystemResolver is used to resolve the hdfs locations so that images can be output into an hdfs folder.
"""
from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.pytorch import DataLoader
from petastorm.fs_utils import FilesystemResolver

"""
Image_size is the target image width and height we use to normalize all images.
Repartitions splits the Spark data into partitions for parallel calculation.
"""
image_size = 512
REPARTITIONS = 10

"""
LOCAL_PATH_DATA defines the local path where our training and validation images are.
UNISCHEMA_OUTPUT defines the local filesystem path where Petastorm will output the parquet files it generates.
PATH_DATA_OUTPUT defines the local filesystem output file path and creates a directory if one doesn't currently exist.
"""
LOCAL_PATH_DATA = "../CheXpert-v1.0-small/"
UNISCHEMA_OUTPUT = "file:///bdh-spring-2020-project-CheXpert/sample_outputs/rdd_data"
PATH_DATA_OUTPUT = "../sample_outputs/rdd_data"
os.makedirs(PATH_DATA_OUTPUT, exist_ok=True)

"""
uni_schema defines the Petastorm schema that we use to generate the output.
The UnischemaFields define the name, numpy type, spark type, and whether null values are allowed in those fields.
"""
uni_schema = Unischema('DataSchema', [
   UnischemaField('Path', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('origin', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('pid', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('Sex', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('Age', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('Frontal_Lateral', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('AP_PA', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('No_Finding', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Enlarged_Cardiomediastinum', np.float32, (), ScalarCodec(FloatType()), False),
   UnischemaField('Cardiomegaly', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Lung_Opacity', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Lung_Lesion', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Edema', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Consolidation', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Pneumonia', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Atelectasis', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Pneumothorax', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Pleural_Effusion', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Pleural_Other', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Fracture', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Support_Devices', np.int64, (), ScalarCodec(FloatType()), False),
   UnischemaField('Labels', np.int64, (14,), NdarrayCodec(), False),
   UnischemaField('Resized_np', np.float64, (image_size, image_size, 3), NdarrayCodec(), False)
])


def normalize(arr):
    """
    Applies a linear normalization on the 3 RGB channels.
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # We do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (1.0/(maxval-minval))
    return arr

def convert_to_np(file):
    """
    The image file is read using the PIL library and converted to a 3 channel RGB for the densenet.
    The file is resized and downsampled with the antialias filter, converted into a numpy array, and normalized.
    """
    image = Image.open(file).convert('RGB') # convert to rgb because we need 3 channels for densnet
    resized_image = image.resize((image_size, image_size), Image.ANTIALIAS)  # resize
    resized_np = np.array(resized_image) #Read as W*H*Channel
    return normalize(resized_np)

def row_generator(x):
    """
    The row_generator generates a single Spark row which is then materialized into a petastorm data structure.
    convert_to_np is applied to each image filename to generate the img_np array.
    The Labels column holds all features of the image.
    """
    img_np = convert_to_np(x[0])
    return {'Path': x[0],
            'origin': x[1],
            'pid': x[2],
            'Sex': x[3],
            'Age': x[4],
            'Frontal_Lateral': x[5],
            'AP_PA': x[6],
            'No_Finding': x[7],
            'Enlarged_Cardiomediastinum': x[8],
            'Cardiomegaly': x[9],
            'Lung_Opacity': x[10],
            'Lung_Lesion': x[11],
            'Edema': x[12],
            'Consolidation': x[13],
            'Pneumonia': x[14],
            'Atelectasis': x[15],
            'Pneumothorax': x[16],
            'Pleural_Effusion': x[17],
            'Pleural_Other': x[18],
            'Fracture': x[19],
            'Support_Devices': x[20],
            'Labels': np.array([x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20]]),
            'Resized_np': img_np}

def main():
    #Print the current time to help us track how long the script takes
    print(datetime.now())

    """
    Create a spark session
    Update the 'spark.driver.memory as required. Too low and one can get a hadoop.MemoryManager error.'
    Spark splits the data into REPARTITIONS (10) partitions and executes calculations on those partitions in parallel
    """
    spark = SparkSession.builder.config('spark.driver.memory', '100g').master('local'+'['+str(REPARTITIONS)+']').getOrCreate()
    sc = spark.sparkContext

    """
    Loads the train.csv into a pandas dataframe.
    Cleans up the AP/PA column adding 'LAT' for lateral images.
    Undefined nan values were replaced with 0 and uncertain -1 values replaced with 1 so that we could classify these
    values to reduce the false negative rate. We wanted to reduce the chances of the algorithm classifying someone as
    when in fact they are not.
    Cleaned up the PATH column so that it referred to the correct filepath where our chest x-ray images are found.
    Extracted the patient ID from the image path and added that as the pid column.
    Created a origin column containing the hdfs input filepath in case we read in from an hdfs filepath in the future.
    """
    replace_str = 'CheXpert-v1.0-small/'
    hdfs_replace_str = '../'
    hdfs_replacement_str = ''
    HDFS_PATH_DATA_OUTPUT = 'hdfs://bootcamp.local:9000/user/local/output'
    HDFS_PATH_DATA = "/user/local/input/CheXpert-v1.0-small/train/"
    HDFS_PREFIX_FILE = "hdfs://bootcamp.local:9000/user/local/input/"

    df_csv = pd.read_csv(os.path.join(LOCAL_PATH_DATA, 'train.csv'))
    df_csv['AP/PA']=df_csv['AP/PA'].fillna('LAT') #Replace the AP/PA nans with LAT
    df_csv = df_csv.fillna(0) #Replace remaining nans with 0
    df_csv = df_csv.replace(-1, 1) #Replace -1s with 1
    df_csv['Path'] = df_csv['Path'].apply(lambda x: x.replace(replace_str, LOCAL_PATH_DATA))
    df_csv.insert(loc=1, column='pid', value=df_csv['Path'].astype(str))
    df_csv['pid'] = df_csv['pid'].apply(lambda x: int(x.split(os.sep)[3][-5:]))
    df_csv.insert(loc=1, column='origin', value=HDFS_PREFIX_FILE + df_csv['Path'].astype(str))
    df_csv['origin'] = df_csv['origin'].apply(lambda x: x.replace(hdfs_replace_str, hdfs_replacement_str))

    """
    Ensure the pandas dataframe treated the feature values as int rather than float datatypes.
    """
    df_csv['No Finding'] = df_csv['No Finding'].astype('int')
    df_csv['Enlarged Cardiomediastinum'] = df_csv['Enlarged Cardiomediastinum'].astype('int')
    df_csv['Cardiomegaly'] = df_csv['Cardiomegaly'].astype('int')
    df_csv['Lung Opacity'] = df_csv['Lung Opacity'].astype('int')
    df_csv['Lung Lesion'] = df_csv['Lung Lesion'].astype('int')
    df_csv['Edema'] = df_csv['Edema'].astype('int')
    df_csv['Consolidation'] = df_csv['Consolidation'].astype('int')
    df_csv['Pneumonia'] = df_csv['Pneumonia'].astype('int')
    df_csv['Atelectasis'] = df_csv['Atelectasis'].astype('int')
    df_csv['Pneumothorax'] = df_csv['Pneumothorax'].astype('int')
    df_csv['Pleural Effusion'] = df_csv['Pleural Effusion'].astype('int')
    df_csv['Pleural Other'] = df_csv['Pleural Other'].astype('int')
    df_csv['Fracture'] = df_csv['Fracture'].astype('int')
    df_csv['Support Devices'] = df_csv['Support Devices'].astype('int')

    """
    Using the petastorm library, we generate spark rows, organize them under the petastorm structure, and
    write parquet files to a local filepath. Since the parquet format stores data in columnar form, the rowgroup size
    defines the target size of a block of rows allowing more efficient access to parquet data.
    """
    rowgroup_size_mb = 256
    with materialize_dataset(spark, UNISCHEMA_OUTPUT, uni_schema, rowgroup_size_mb):
        rows_rdd = sc.parallelize(df_csv.values).map(row_generator).map(lambda x: dict_to_spark_row(uni_schema, x))
        sdf = spark.createDataFrame(rows_rdd, uni_schema.as_spark_schema()).coalesce(10)
        sdf.write.mode('Overwrite').parquet(UNISCHEMA_OUTPUT)
        local_sdf_rows = sdf.count()

    """
    Using the petastorm library, we generate spark rows, organize them under the petastorm structure, and
    write parquet files to an hdfs filepath.
    """
    hdfs_output = 'hdfs://bootcamp.local:9000/user/local/output/'

    resolver=FilesystemResolver(hdfs_output, spark.sparkContext._jsc.hadoopConfiguration(), hdfs_driver='libhdfs')
    with materialize_dataset(spark, HDFS_PATH_DATA_OUTPUT, uni_schema, rowgroup_size_mb, filesystem_factory=resolver.filesystem_factory()):
        rows_rdd = sc.parallelize(df_csv.values).map(row_generator).map(lambda x: dict_to_spark_row(uni_schema, x))
        sdf = spark.createDataFrame(rows_rdd, uni_schema.as_spark_schema()).coalesce(10)
        sdf.write.mode('Overwrite').parquet(HDFS_PATH_DATA_OUTPUT)
        hdfs_sdf_rows = sdf.count()

    """
    Print the current time at the end of the script to track how long the script took
    Print the size and filepath of the local and hdfs files generated.
    """
    print(datetime.now())
    print("Complete!")
    print(f"Local file of {local_sdf_rows} rows generated at {UNISCHEMA_OUTPUT}")
    print(f"HDFS file of {hdfs_sdf_rows} rows generated at {HDFS_PATH_DATA_OUTPUT}")

if __name__ == '__main__':
    main()
