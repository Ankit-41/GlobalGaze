# GlobalGaze: Efficient Global Context Extraction for Image Restoration

The official PyTorch implementation of the paper

**GlobalGaze: Efficient Global Context Extraction for Image Restoration**  
*Amirhosein Ghasemabadi, Muhammad Kamran Janjua, Mohammad Salameh, Chunhua Zhou, Fengyu Sun, Di Niu*  
Accepted at Transactions on Machine Learning Research (TMLR), 2024.

## Installation

This implementation is based on BasicSR, an open-source toolbox for image/video restoration tasks, incorporating methods from NAFNet, Restormer, and Multi Output Deblur.

### Requirements

- Python 3.9.5
- PyTorch 1.11.0
- CUDA 11.3

To install the required packages, run:
```bash
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Required Packages

- addict
- future
- lmdb
- numpy
- opencv-python
- Pillow
- pyyaml
- requests
- scikit-image
- scipy
- tb-nightly
- tqdm
- yapf
- ptflops
- matplotlib

## Quick Start

We have provided `demo-denoising.ipynb` to demonstrate how to load images from the validation dataset and use the model to restore images.

## GlobalGaze Implementation

- The implementation of the GlobalGaze Net, GlobalGaze block, and the Global Context Extractor module can be found in `/GlobalGaze/basicsr/models/archs/GGNet_arch.py`.
- The implementation of the Multi-Head GlobalGaze Net can be found in `/GlobalGaze/basicsr/models/archs/GGNetMultiHead_arch.py`.

## Denoising on SIDD

### 1. Data Preparation

1. Download the training set from the SIDD dataset website and place it in `./datasets/SIDD/Data/`.
2. Download the evaluation data in LMDB format from the Gopro dataset website and place it in `./datasets/SIDD/test/`.

Your dataset directory should look like this:
```plaintext
./datasets/
└── SIDD/
    ├── Data/
    │   ├── 0001/
    │   │   ├── GT_SRGB.PNG
    │   │   ├── NOISY_SRGB.PNG
    │   │   ....
    │   └── 0200/
    │       ├── GT_SRGB.PNG
    │       ├── NOISY_SRGB.PNG    
    ├── train/
    └── test/
        ├── input.imdb
        └── target.imdb
```

3. Use the script `scripts/data_preparation/sidd.py` to crop the train image pairs to 512x512 patches and convert them into LMDB format. The processed images will be saved in `./datasets/SIDD/train/`.

### 2. Training

To train the GlobalGaze model, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/SIDD/GlobalGaze-SIDD.yml --launcher pytorch
```

### 3. Evaluation

Note: Due to file size limitations, pre-trained models are not included in this code submission but will be provided with an open-source release of the code.

To evaluate the pre-trained model, use:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/test.py -opt ./options/test/SIDD/GlobalGaze-SIDD.yml --launcher pytorch
```

### 4. Model Complexity and Inference Speed

To get the parameter count, MAC, and inference speed, run:
```bash
python GlobalGaze/basicsr/models/archs/GGNet_arch.py
```

## Gaussian Image Denoising

### 1. Data Preparation

Clone the Restormer's GitHub project and follow their instructions to download the train and test datasets.

### 2. Training

1. Copy the `GlobalGaze/basicsr/models/archs/GGNet_Gaussian_arch.py` to `Restormer/basicsr/models/archs/`.
2. Copy the training option files from `GlobalGaze/options/train/Gaussian/` to `Restormer/Denoising/Options/`.

Follow Restormer's training instructions and train models on different noise levels.

### 3. Evaluation

Note: Pre-trained models will be released soon.

To evaluate the pre-trained model, start by adjusting the noise level (sigma=15, 25, or 50), the paths to the trained model, and the training option file within the code. Once modified, execute:
```bash
python GlobalGaze/basicsr/test_gaussian_color_denoising.py
```

### 4. Model Complexity and Inference Speed

To get the parameter count, MAC, and inference speed, run:
```bash
python GlobalGaze/basicsr/models/archs/GGNet_Gaussian_arch.py
```

## Deblurring on GoPro

### 1. Data Preparation

1. Download the training set from the GoPro dataset website and place it in `./datasets/GoPro/train`.
2. Download the evaluation data in LMDB format from the GoPro dataset website and place it in `./datasets/GoPro/test/`.

Your dataset directory should look like this:
```plaintext
./datasets/
└── GoPro/
    ├── train/
    │   ├── input/
    │   └── target/
    └── test/
        ├── input.imdb
        └── target.imdb
```

3. Use the script `scripts/data_preparation/gopro.py` to crop the train image pairs to 512x512 patches and convert them into LMDB format.

### 2. Training

To train the GlobalGaze Multi-Head model, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/GoPro/GlobalGazeMH-GoPro.yml --launcher pytorch
```

To fine-tune the trained GlobalGazeMH model on larger patches:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/GoPro/GlobalGazeMH-GoPro-finetune_largerPatch.yml --launcher pytorch
```

### 3. Evaluation

Note: Pre-trained models will be released soon.

To evaluate the pre-trained model, use:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/test.py -opt ./options/test/GoPro/GlobalGazeMH-GoPro.yml --launcher pytorch
```

### 4. Model Complexity and Inference Speed

To get the parameter count, MAC, and inference speed, run:
```bash
python GlobalGaze/basicsr/models/archs/GGNetMultiHead_arch.py
```

## Visualizing the Training Logs

You can use TensorBoard to track the training status:
```bash
tensorboard --logdir=/GlobalGaze/logs
```
