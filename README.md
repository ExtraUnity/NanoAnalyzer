# NanoparticleAnalysis
Bachelor Project 2025

## License

NanoAnalyzer is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

See the [LICENSE](./LICENSE) file for the full text of the GPLv3 license.

# Installation Guide

## Downloading executables
The application has been exported as executables for Windows, Mac and Linux. Download these through "Releases". These also have pre-trained models ready for use.
If you want to build the project using the source code, follow the guides below.

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
