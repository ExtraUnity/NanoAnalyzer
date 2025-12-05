# NanoAnalyzer
**NanoAnalyzer** is a user-friendly application for automated segmentation and analysis of supported nanoparticles in electron microscopy images (TEM and STEM).  
It combines a U-Net–based deep learning model with an intuitive graphical interface, allowing materials researchers to:

- segment nanoparticles without writing code,
- process entire folders of images in batch,
- extract particle-level statistics (area, equivalent circular diameter, size distributions),
- and optionally train new models on their own annotated datasets.

NanoAnalyzer is developed by Christian Vedel Petersen & Nikolaj Nguyen originally as part of their bachelor's thesis at the Technical University of Denmark, and later as part of a study on **accessible deep learning for automated segmentation of supported nanoparticles in electron microscopy**.

## Table of Contents
1. [Installation Guide](https://github.com/ExtraUnity/NanoparticleAnalysis#installation-guide)
2. [User Guide](https://github.com/ExtraUnity/NanoparticleAnalysis#user-guide)
   - [Segmenting Images](https://github.com/ExtraUnity/NanoparticleAnalysis#segmenting-images)
   - [Batch Processing](https://github.com/ExtraUnity/NanoparticleAnalysis#batch-processing)
   - [Training a New Model](https://github.com/ExtraUnity/NanoparticleAnalysis#training-a-new-model)
   - [Data Format for Training](https://github.com/ExtraUnity/NanoparticleAnalysis#data-format-for-training)
   - [Test Model](https://github.com/ExtraUnity/NanoparticleAnalysis#test-model)
4. [License](https://github.com/ExtraUnity/NanoparticleAnalysis#license)
5. [Contact](https://github.com/ExtraUnity/NanoparticleAnalysis#contact)



## Installation Guide

### Downloading executables
The application has been exported as an executable for Windows. 
Download these through [Releases](https://github.com/ExtraUnity/NanoparticleAnalysis/releases). These also have pre-trained models ready for use.
Simply download and unzip the application. Then open the **NanoAnalyzer.exe** to start the program. Alternatively use the installer provided.
If only CPU segmentation is needed, we recommend downloading the CPU-only version of the application.

### Running the source code
To run the source code, do the following steps:
0. Clone/download the repository
1. Install Conda
2. Create the Conda environment
3. Run the main.py

#### Installing Conda
An installation guide for Conda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

#### Creating the conda environment
To create the conda environment, run the following commands in the terminal:
1. ```conda env create -f environment.yml```
2. ```conda activate nanoanalyzer```

#### Running the application
To run the application from the source code, run ```python main.py```

## User Guide

### Segmenting Images
1. Load an image
   - Click **Open Image** from the files tab and select a TEM or STEM image (e.g. `.tif`, `.dm3`, `.dm4`)
   - The image should display, and scale information should be shown above the image.
   - Alternatively, you can set a scale by pressing **Set Scale**
   
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/f6f48524-c27e-4d81-b3ac-4e6beba5b45a" />
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/03915506-94ca-42b7-a797-d9d6ccc2c5e5" />

3. Run segmentation
   - Simply click **Run Segmentation** to segment the image
      Note that images are downscaled to a maximum of 1024x1024 for computational efficiency.
     
4. View results
   - After segmentation, the program will display the segmentation and write the statistics information to the folder `data/<name_of_image>/` (from the same directory as the .exe file)
      - The statistics includes the per-particle metrics **Area**, **Diameter (equivalent circular diameter)** and **Particle ID** (for identification on segmentation map)
   - The segmentation can be viewed side by side in a fullscreen window by pressing **Fullscreen Image**
   - Summary statistics are viewed in the application. You can also export the data as CSV files under **Export**.
     
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/97b10898-1236-4c19-b2e8-19d43479c6e3" />


### Batch Processing
1. Click **Analyze > Run segmentation on folder**
2. Select folder to segment
   - All images in the folder should have readable units (e.g. `nm`, `μm`) in their metadata, otherwise the program will not be able to segment the folder.
4. Select output folder
   The program will then segment the entire folder in the background. A folder for each segmentation will be created, along with a collected statistics text file.
   

### Training a New Model
NanoAnalyzer allows you to train new models on your own annotated data directly from the interface.
1. Prepare your dataset

   See [Data Format for Training](https://github.com/ExtraUnity/NanoparticleAnalysis#data-format-for-training)
2. Open the training tab, **Model > Train New Model**
3. Set training options
   - Choose the dataset folders
   - Optionally adjust
     - Separate test set
     - Number of epochs
     - Learning rate
     - Use of data augmentation (Random cropping, random rotation, random brightness adjustment, random contrast adjustment)
     - Use of early stopping (25 epochs)
4. Click **Train Model**
   - The application will
      - validate the dataset (matching sizes and binary masks),
      - split it into training/validation/test sets (60/20/20),
      - train the U-Net model,
      - evaluate performance on the test set and display final model results along with example segmentations.
        
   During training, the application will provide continuous statistics (after each epoch) on training loss and validation loss, along with the best epoch.
   At any time, the user can stop the training, and the application will stop the training at the nearest checkpoint and save the best model.
5. Load the new model
   - After training, the new model is saved locally (`data/models/`).
   - You can load the model by clicking **Model > Load Model** and selecting the model `.pt` file.
### Data Format for Training
A training dataset consists of two folders:
- `images/` - The raw TEM/STEM images
- `masks/` - Corresponding binary annotation masks
  
Requirements:
- Each image in `images/` must have a corresponding mask in `masks/` with:
   - The same filename
   - The same dimensions
- Masks should have binary pixel values
   - 0 = background
   - 1 (or 255) = foreground (particle)
     
Large images will automatically be downscaled to 1024x1024 during training to fit the downscaling during inference.

### Test Model
You can test the currently loaded model against a new dataset.
1. Click **Model > Test Model**
2. Select the test images folder
3. Select the test masks folder
   
The test dataset should be formatted according to [Data Format for Training](https://github.com/ExtraUnity/NanoparticleAnalysis#data-format-for-training).

## License
Copyright © 2025, Christian Vedel Petersen & Nikolaj Nguyen

NanoAnalyzer is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

See the [LICENSE](./LICENSE) file for the full text of the GPLv3 license.

## Contact
For questions, bug reports, or feature requests, please contact:
- **Name:** Christian Vedel Petersen  
- **Email:** s224810@dtu.dk 
  
You are also welcome to open issues or pull requests directly in this repository.
