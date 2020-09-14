# bdh-spring-2020-project-CheXpert

## Building Docker container

While in this repo/directory, run the following:

```
docker build -t bdh-chexpert .
```

## Running Docker container

While in this repo/directory, run the following:

```
docker run -it --privileged=true --cap-add=SYS_ADMIN -m 8192m -h bootcamp.local --name bdh-chexpert -p 2222:22 -p 9530:9530 -p 8888:8888 -p 8080:8080 -v $PWD:/bdh-spring-2020-project-CheXpert bdh-chexpert /bin/bash
```

## Once in Docker container

Run the following to start dependent services:

```
/scripts/start-services.sh
```

Run the following to activate the conda environment:

```
conda activate bdh-chexpert
```

Run the following to start the Jupyter notebook:

```
jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
```

## Once in the Conda environment:

The etl_dist_petastorm.py script read images from a local directory and outputs parquet files on an HDFS directory.
To update the inputs directory please do the following:

```
Script processes images in sample directory.
Update UNISCHEMA_OUTPUT with the folder containing the train.csv, valid.csv file and images.
Please use file:/// to refer to a local directory.
Example: file:///bdh-spring-2020-project-CheXpert/sample_outputs/rdd_data
```

Files are output to HDFS_PATH_DATA_OUTPUT. Please please make sure to create the following /user/local/output HDFS directory.

```
sudo su - hdfs
hdfs dfs -mkdir /user/local/
hdfs dfs -mkdir /user/local/output/
```

Once the directories are created, we need to allow the user, likely 'root' within our docker container, to have read and write access.

```
hdfs dfs -chown root /user/local/
hdfs dfs -chown root /user/local/output/
```

To change the HDFS directory, please make sure to:
Update HDFS_PATH_DATA and  with the folder containing the csv files and images.
Update HDFS_PREFIX_FILE with the path to the input directry.

```
Please use hdfs:/// to refer to a local directory.
Example: hdfs://bootcamp.local:9000/user/local/output/
```

Make sure to update the Spark REPARTITIONS with the desired number of partitions. A rule of thumb may be total cores * 3 or 4.


# Instructions for Running Stuff in the GCE VMs:

## VM Stuff

SSHing in:

```
gcloud beta compute ssh --zone "us-central1-c" "root@guc3-jameswentest-gputest-9b67" --project "jameswenspotify" --internal-ip -- -A
```

Removing running container and cleaning it up:

```
docker kill bdh-chexpert
docker rm bdh-chexpert
```

Building Docker image:

```
docker build -t bdh-chexpert .
```

Checking how much memory VM has:

```
free -g
free -m
```

GPU/CUDA Setup:

```
apt install cuda
```

GPU/CUDA Verification:

```
cat /proc/driver/nvidia/version
nvcc -V
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## Data Downloading:

Small dataset:

```
nohup curl http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip --output CheXpert-v1.0-small.zip &
nohup unzip CheXpert-v1.0-small.zip &

```

Full dataset:

```
nohup curl http://download.cs.stanford.edu/deep/CheXpert-v1.0.zip --output CheXpert-v1.0.zip &
nohup unzip CheXpert-v1.0.zip &
```

## HDFS Setup:

Switch to `hdfs` user:

```
sudo su - hdfs
```

Create directories in hdfs:

hdfs dfs -mkdir /user/local/
hdfs dfs -mkdir /user/local/output/

Allow root user to own directories:

```
hdfs dfs -chown root /user/local/
hdfs dfs -chown root /user/local/output/
```

## Docker Container:

Note: Use `tail -f nohup.out` or `cat nohup.out` to view detached long duration processes like ETL or training.

Starting detached Docker container (on GCE VM with GPUs):

```
docker run -it --detach --privileged=true --cap-add=SYS_ADMIN --shm-size 400000m -m 400000m --cpus 60 -h bootcamp.local --name bdh-chexpert -p 2222:22 -p 9530:9530 -p 8888:8888 -p 8080:8080 -v $PWD:/bdh-spring-2020-project-CheXpert --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 --device /dev/nvidia2:/dev/nvidia2 --device /dev/nvidia3:/dev/nvidia3 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm bdh-chexpert /bin/bash
```

This command starts the Docker container with the following:

- As a privileged container
- In a detached mode
- Allocates it 40 GB of memory and 60 vCPUs
- Does a few port mappings
- Mounts the current directory as /bdh-spring-2020-project-CheXpert in the container
- Ensures that the Docker container can use the 4 GPUs properly

Entering the Docker container:

```
docker exec -it bdh-chexpert bash
```

### Services and Conda env setup:

```
/scripts/start-services.sh
conda activate bdh-chexpert
```

### ETL:

Ensure proper dataset (`CheXpert-v1.0-small/`) is in repo dir (`/root/bdh-spring-2020-project-CheXpert` in the container).

```
cd etl
rm -rf nohup.out
nohup python etl_dist_petastorm.py &
```

### Training:

Ensure that config.json has the correct parameters set before running training.
If using the Petastorm loader, make sure the data_loader_petastorm values are udpated in the config file.

```
cd deeplearning
nohup ./run.sh &
```

### Grabbing Results from VM:

Tar up whatever you need (ex. contents of `saved` dir with models and logs from training) via:

```
nohup tar -cvzf <desired tar archive>.tar.gz <dir to compress> &
```

Example:

```
nohup tar -cvzf saved-4-26-2020-2-PM.tar.gz saved &
```

Use `gcloud scp` to remote copy it from the VM to your machine's local disk:

```
gcloud beta compute scp --zone "us-central1-c" "root@guc3-jameswentest-gputest-9b67:/root/saved-4-26-2020-2-PM.tar.gz" saved-4-26-2020-2-PM.tar.gz --project "jameswenspotify" --internal-ip
```

## Helpful References for GPU + Docker stuff:

- https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container
- https://xcat-docs.readthedocs.io/en/stable/advanced/gpu/nvidia/verify_cuda_install.html
- https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
