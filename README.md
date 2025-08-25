This repository contains the codes for the paper:
"A Physics-Informed Dynamic and Constrained Machine Learning Framework for CO₂ Saturation Prediction in Geological Carbon Storage Reservoirs"

Reproducibility Instructions:
1. Clone this repository (or download it directly from Zenodo).
2. Download the dataset from Figshare: https://doi.org/10.6084/m9.figshare.26962108.v4
3. Unzip the dataset into a folder named dataset/ in the project root directory.
4. Update paths: Before running the scripts, please update the file_folder and data_directory variables in the code to match the location of your files.
5. Run training and testing:
python train.py  
python test.py  
6. Pre-trained models (Model A, Model B, Model C, and Model A with penalty, as reported in our manuscript) are available on Figshare: https://doi.org/10.6084/m9.figshare.26962108.v4. Using these models, you can skip the train.py step.
