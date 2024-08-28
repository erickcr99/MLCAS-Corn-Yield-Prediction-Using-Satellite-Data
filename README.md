# MLCAS-Corn-Yield-Prediction-Using-Satellite-Data
MLCAS Corn Yield Prediction Using Satellite Data model
Satellite Yield Prediction
This repository contains a pipeline for predicting crop yield per acre using satellite imagery data from 2022 and 2023. The project involves loading, preprocessing, and analyzing satellite images, training a Convolutional Neural Network (CNN) model, and validating its performance using validation and test datasets.

Table of Contents
Overview
Requirements
Setup
Data Preparation
Model Training
Validation and Testing
Results
Usage
Contributing
License
Overview
This project aims to predict crop yield per acre based on satellite images. It uses a CNN model trained on labeled data from 2022 and 2023. The trained model is then used to predict yields for validation and test datasets, and the predictions are updated in the respective CSV files.

Requirements
Python 3.x
TensorFlow 2.x
Pandas
NumPy
Rasterio
Matplotlib
Scikit-learn
Setup
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/satellite-yield-prediction.git
cd satellite-yield-prediction
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Set up your data directories as specified in the script:

2022/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv
2023/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv
2023_validacion/2023/Satellite/
Test/Satellite/
Data Preparation
Ensure that your data is correctly placed in the directories specified in the script. The data should include satellite images in .TIF format and CSV files containing the ground truth crop yield data.

Model Training
To train the CNN model:

Run the training script:

bash
Copy code
python train_model.py
The model will be trained using the combined data from 2022 and 2023. The training process includes splitting the data into training, validation, and test sets.

Validation and Testing
After training, the model is validated and tested using the respective datasets:

Validation: The model's predictions for the validation dataset are generated and stored in the corresponding CSV file.
Testing: Similarly, predictions for the test dataset are generated and stored in the CSV file.
To perform validation and testing:

Run the validation and testing script:
bash
Copy code
python validate_and_test.py
Results
The results of the model's predictions (in terms of crop yield per acre) are updated in the CSV files provided in the 2023_validacion and Test directories.

Usage
You can modify the paths to the datasets in the script as per your directory structure. The key functions include:

load_and_preprocess_tif(image_path): Loads and preprocesses the satellite images.
create_cnn_model(input_shape): Creates and compiles the CNN model.
split_data(data, test_size=0.2, val_size=0.1): Splits the dataset into training, validation, and test sets.
build_image_path(row, base_path): Constructs the image path based on CSV data.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
