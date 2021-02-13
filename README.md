# Polandball Flag Mapping

The premise for this project is to automatically texturize Polandball outlines with any arbitrary flag.

<!-- ![](_readme_images\model_overview.svg "Model overview") -->

<p align="center">
    <img src="_readme_images\model_overview.svg" alt="drawing" width="400" />
</p>

## Training

1. Download the dataset from https://www.kaggle.com/zimonitrome/polandball-characters

2. Put the `balls` and `flags` folders into `./data/`.

3. Run the first pre-processing from the main directory: 
    ```ps
    python .\preprocessing\process_traning_data.py
    ```
    This script can take some time but can be run in multiple instances. <br> Also consider not using 100% of the dataset.

4. Train the GMM in phase 1:
    ```ps
    python ./training/train_GMM_phase1.py
    ```
    Note that no CLI options are provided. Training parameters are set in each training file.

    The trained model will be saved in `./training/checkpoints/GMM_P1_***/***.pt`.

5. Provide the path for the trained GMM_P1 model (above step) in `./training/train_GMM_phase2.py` line XX.

    Train the GMM in phase 2:
    ```ps
    python ./training/train_GMM_phase2.py
    ```

6. Run the second pre-processing step now that a valid GMM model is available:
    ```ps
    python ./preprocessing/process_traning_data_BSM.py
    ```

7. Train the BSM:
    ```ps
    python .\training\train_BSM.py
    ```

8. Done! Make note of where your saved model (`.pt`) files are stored for use in inference.
