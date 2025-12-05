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
   - [Viewing Results and Statistics](https://github.com/ExtraUnity/NanoparticleAnalysis#viewing-results-and-statistics)
   - [Training a New Model](https://github.com/ExtraUnity/NanoparticleAnalysis#training-a-new-model)
   - [Data Format for Training](https://github.com/ExtraUnity/NanoparticleAnalysis#data-format-for-training)
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

2. Choose a model
   - Be default, the **pretrained model** is loaded.
   - You can later switch to a **custom-trained model** and load it by pressing **Model > Load Model** (see [Training a New Model](https://github.com/ExtraUnity/NanoparticleAnalysis#training-a-new-model))
     
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/9f9ea52c-35e3-467a-8b46-9770b3302de1" />

3. Run segmentation
   - Simply click **Run Segmentation** to segment the image
     
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/95578d03-67b5-4b7e-8e4f-6dbe0a012d27" />

4. View results
   - After segmentation, the program will display the segmentation and write the statistics information to the folder `data/<name_of_image>/` (from the same directory as the .exe file)
   - The segmentation can be viewed side by side in a fullscreen window by pressing **Fullscreen Image**
   - Summary statistics are viewed in the application. You can also export the data as CSV files under **Export**.
     
      <img width="443" height="262" alt="image" src="https://github.com/user-attachments/assets/97b10898-1236-4c19-b2e8-19d43479c6e3" />


### Batch Processing

### Viewing Results and Statistics

### Training a New Model

### Data Format for Training


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
