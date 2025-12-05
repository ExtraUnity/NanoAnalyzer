# NanoAnalyzer
**NanoAnalyzer** is a user-friendly application for automated segmentation and analysis of supported nanoparticles in electron microscopy images (TEM and STEM).  
It combines a U-Net–based deep learning model with an intuitive graphical interface, allowing materials researchers to:

- segment nanoparticles without writing code,
- process entire folders of images in batch,
- extract particle-level statistics (area, equivalent circular diameter, size distributions),
- and optionally train new models on their own annotated datasets.

NanoAnalyzer is developed by Christian Vedel Petersen & Nikolaj Nguyen originally as part of their bachelor's thesis at the Technical University of Denmark, and later as part of a study on **accessible deep learning for automated segmentation of supported nanoparticles in electron microscopy**.

## Table of Contents
1. [Installation Guide](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#installation-guide)
2. [User Guide](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#user-guide)
   - [Segmenting Images](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#segmenting-images)
   - [Batch Processing](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#batch-processing)
   - [Viewing Results and Statistics](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#viewing-results-and-statistics)
   - [Training a New Model](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#training-a-new-model)
   - [Data Format for Training](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#data-format-for-training)
4. [License](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#license)
5. [Contact](https://github.com/ExtraUnity/NanoparticleAnalysis/main/README.md#contact)



## Installation Guide

### Downloading executables
The application has been exported as executables for Windows. Download these through [Releases](https://github.com/ExtraUnity/NanoparticleAnalysis/releases). These also have pre-trained models ready for use.
If you want to build the project using the source code, follow the guides below.

### Running the source code
## Install Conda and import the environment.yml based on your OS
0. Conda installation guide https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 
1. Install pyinstaller with 'pip install pyinstaller'

## Windows:
0. Navigate to root of project folder
1. Run the following: 'pyinstaller --noconfirm --noconsole --name NanoAnalyzer  --add-data "src/data/model/UNet_best_06-06.pt;src/data/model" main.py '
2. If you do not have a pre-trained model, do not run with the --add-data argument
3. To run the application, navigate to dist folder and execute/open: 'main.exe'

## Linux (Ubuntu): 
0. Navigate to root of project folder
1. Give build script exec. permission: 'sudo chmod +x build_app_mkl.sh'
2. Run the script 'build_app_mkl.sh'
3. To run the application, navigate to the 'NanoAnalyzer' folder and run NanoAnalyzer.

## MacOS: 
0. Navigate to root of project folder
1. Give build script exec. permission: 'sudo chmod +x build_app_mac.sh'
2. Run the script 'build_app_mac.sh'
3. To run the application, navigate to the 'NanoAnalyzer' folder and run NanoAnalyzer.

## User Guide

### Segmenting Images

### Batch Processing

### Viewing Results and Statistics

### Training a New Model

### Data Format for Training


## License

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
