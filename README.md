# MLCAS-Corn-Yield-Prediction-Using-Satellite-Data
MLCAS Corn Yield Prediction Using Satellite Data model
This repository contains the implementation for predicting corn yield per acre using satellite data as part of the MLCAS 2024 challenge. The project includes a trained model, a notebook with exploratory data analysis, and the resulting CSV files for validation and test datasets.

## Repository Structure

- `Model.py`: This script contains the final CNN model for predicting yield using satellite imagery. It is designed to load, preprocess, and predict yield values based on the multispectral satellite images.
- `Notebook_code.ipynb`: A Jupyter Notebook that documents the exploratory data analysis, data preprocessing steps, model development process, and the training procedure. This notebook serves as a detailed walkthrough from raw data to the final model.
- `test_HIPS_HYBRIDS_2023_V2.3.csv`: The CSV file with predicted yield values for the test dataset. These predictions were generated using the trained model.
- `val_HIPS_HYBRIDS_2023_V2.3.csv`: The CSV file with predicted yield values for the validation dataset. These predictions were also generated using the trained model.
- `README.md`: This file, providing an overview of the repository and instructions on how to use the files.

## Requirements

- Python 3.10
- TensorFlow 2.x
- Pandas
- NumPy
- Rasterio
- Matplotlib
- Scikit-learn

## Model
The Model.py script implements a Convolutional Neural Network (CNN) designed to predict the yield per acre based on satellite images from the 2022 and 2023 datasets. The model was trained and validated on these datasets and is now used to make predictions on unseen data.

## Key Features:
Preprocessing: The model script includes functions to load and preprocess satellite images.
Training: The CNN model is trained on combined data from 2022 and 2023, with image normalization and resizing.
Prediction: The script can be used to predict yield on new datasets, with predictions saved back to CSV files.

## Usage
To use the model on new data:

1. Ensure your data is structured similarly to the provided CSVs.
2. Modify the paths in Model.py to point to your new datasets.
3. Run the script to generate predictions.
4. 
For detailed steps and code, refer to Notebook_code.ipynb.

## Results
The val_HIPS_HYBRIDS_2023_V2.3.csv and test_HIPS_HYBRIDS_2023_V2.3.csv contain the yield predictions for the validation and test datasets, respectively. These files can be used to compare against actual yields or as part of a larger analysis.

## Contributing
Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a Pull Request.

