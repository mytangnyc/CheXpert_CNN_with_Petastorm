To Train:
    Modify the Config json to set up the desirable training parameters, specially modify the 
   ```json
        "data_set_config": {
            "label_npy": "<PATH_TO_LABEL_FILE_GENERATED_FROM_ETL>",
            "image_npy": "<PATH_TO_LABEL_FILE_GENERATED_FROM_IMG>"
        }
  ```
Then Go to deeplearning dir: `PYTHONPATH=../:. python train.py`  