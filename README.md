# Polandball Flag Mapping

The premise for this project is to automatically texturize Polandball outlines with any arbitrary flag.

<p align="center">
    <img src="_readme_images\model_overview.svg" width="400" />
</p>

## Dependencies

This project has only been tested using Anaconda environments. Packages can probably be installed in vanilla Python as well.

Install the `polandball_mapping` environment through `environment.yml`:

```ps
conda env create -f environment.yml
```

## Training

1. Download the dataset from https://www.kaggle.com/zimonitrome/polandball-characters

2. Put the `balls` and `flags` folders into `./data/`.

3. Run the first pre-processing from the main directory: 
    ```ps
    python ./preprocessing/process_traning_data.py
    ```
    This script can take some time but can be run in multiple instances. <br> Also consider not using 100% of the dataset.

4. Train the GMM in phase 1:
    ```ps
    python ./training/train_GMM_phase1.py
    ```
    Note that no CLI options are provided. Training parameters are set in each training file.

    The trained model will be saved in `./training/checkpoints/GMM_P1_***/***.pth`.

5. Move the trained model to `main_weights` and rename it to `GMM.pth`.

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
    python ./training/train_BSM.py
    ```

8. Done! Make note of where your saved model (`.pth`) files are stored for use in inference.


## Inference

To try the model, please refer to `inference_demo.ipynb`.

The pre-trained weights `./main_weights/GMM.pth` and `./main_weights/BSM.pth` must exist (will soon be published!).

## Examples


<img src="_readme_images\architecture.svg" width="100%" />
<p align="center">Architecture of the two models.</p>
<br>
<img src="_readme_images\balls_grid.svg" width="100%" />
<p align="center">Inputs and outputs using different settings.</p>
<br>
<img src="_readme_images\FID_ACC_bar.svg" width="100%" />
<p align="center">Performance for each different set of settings.</p>

## Cite

```
@inproceedings{arvidsson2021texture,
  title={Texture Mapping of Flags onto Polandball Characters using Convolutional Neural Nets},
  author={Arvidsson, Simon and Gabrielsson, Patrick and Johansson, Ulf},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--7},
  year={2021},
  organization={IEEE}
}
```
