# Cloud Type Classification Project

## Overview
This project involves developing a machine learning model to classify images of clouds into different types. The dataset used includes images of clouds labeled into seven categories, and the model is trained using PyTorch. The project showcases skills in data preprocessing, model training, and evaluation, as well as handling a competition-style test and submission workflow.

## Project Structure
- **data/**: Contains the datasets including `train.csv` and `test.csv`.
- **images/**: Directory containing subdirectories `train` and `test` for image data.
- **cloud_train/**: Directory where training images are sorted into subdirectories based on cloud type.
- **cloud-type-classification2/**: Directory for Python scripts and other project files.
- **classification.py**: Main script for data processing, model training, and prediction.
- **submit.csv**: Template for competition submission file.

## Features
- **Data Preprocessing**: Organizing images based on labels and applying data augmentation techniques.
- **Model Training**: Implementing a Convolutional Neural Network (CNN) using PyTorch.
- **Evaluation**: Using precision and recall metrics to evaluate model performance.
- **Prediction**: Generating predictions for test images and preparing submission files for competition.

## Technologies Used
- Python
- PyTorch
- pandas
- torchvision
- matplotlib
- torchmetrics

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Kingtilon1/Cloud-classification.git
    cd cloud-type-classification
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Data Preprocessing
The script reads `train.csv` to organize training images into appropriate subdirectories based on their labels. Run the following command to start preprocessing:
```bash
python classification.py preprocess
```

## Training the Model
To train the model, run:
```bash
python classification.py train
```
This command will train the CNN model using the preprocessed training data and save the trained model.

## Evaluating the Model
To evaluate the model using validation data, run:
```bash
python classification.py evaluate
```
This will calculate and print precision and recall metrics for the validation set.

## Making Predictions
To generate predictions for the test data and prepare the submission file, run:
```bash
python classification.py predict
```

This will output a final_submission.csv file ready for submission to the competition.

## Example Code
### Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])
```

### Model Definition
```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
```

### Model Training Loop

```python
for epoch in range(10):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Results 
The model achieved high recall and precision metrics, indicating its effectiveness in correctly classifying cloud types.
The final model was submitted to a competition, demonstrating practical application of the developed solution.

## Contact
For any questions or inquiries, please contact me at bobbtilon@gmail.com
