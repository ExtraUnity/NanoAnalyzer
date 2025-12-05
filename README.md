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
