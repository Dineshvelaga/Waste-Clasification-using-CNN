# CNN Model for Plastic Waste Classification

<h1 align="center">Hi there, I'm Dinesh Velaga </h1>
<h3 align="center">Senior Undergrad Student | CSE | AIML | AWS</h3>



---



## Overview  
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into various categories. The primary goal is to enhance waste management systems by improving the segregation and recycling process using deep learning technologies.  

---

## Table of Contents  
- [Project Description](#project-description)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Weekly Progress](#weekly-progress)  
- [How to Run](#how-to-run)  
- [Technologies Used](#technologies-used)  
- [Future Scope](#future-scope)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Description  
Effective trash segregation is essential to addressing the rising worldwide concern about plastic pollution. To enable automated garbage management, this study uses a CNN model to categorize plastic waste into discrete groups.  

## Dataset  
Sashaank Sekar's **Waste Classification Data** used as the dataset for this research. There are 25,077 captioned photos in all, split into two categories: **Recyclable** and **Organic**. The purpose of this dataset is to support machine learning tasks related to trash categorization.  

### Key Details:
- **Total Images**: 25,077  
  - **Training Data**: 22,564 images (85%)  
  - **Test Data**: 2,513 images (15%)  
- **Classes**: Organic and Recyclable  
- **Purpose**: To aid in automating waste management and reducing the environmental impact of improper waste disposal.
  
### Approach:  
- Studied waste management strategies and white papers.  
- Analyzed the composition of household waste.  
- Segregated waste into two categories (Organic and Recyclable).  
- Leveraged IoT and machine learning to automate waste classification.  

### Dataset Link:  
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

*Note: Ensure appropriate dataset licensing and usage guidelines are followed.*  


## Model Architecture  
The CNN architecture includes:  
- **Convolutional Layers:** Feature extraction  
- **Pooling Layers:** Dimensionality reduction  
- **Fully Connected Layers:** Classification  
- **Activation Functions:** ReLU and Softmax  

### Basic CNN Architecture  
Below is a visual representation of the CNN architecture used in this project:  

<p align="center">
  <img src="https://github.com/Dineshvelaga/Waste-Clasification-using-CNN/blob/main/CNNimage.jpg" alt="Basic CNN Architecture" style="width:80%;">
</p>

## Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** Configurable (default: 25)  
- **Batch Size:** Configurable (default: 32)  

Data augmentation techniques were utilized to enhance model performance and generalizability.  

## Weekly Progress  
This section will be updated weekly with progress details and corresponding Jupyter Notebooks.  

### Week 1: Libraries, Data Import, and Setup  
- **Date:** 20th January 2025 - 27th January 2025  
- **Activities:**  
  - Imported the required libraries and frameworks.  
  - Set up the project environment.  
  - Explored the dataset structure.  
  - Note: If the file takes too long to load, you can view the Kaggle notebook directly [Kaggle Notebook](https://www.kaggle.com/code/hardikksankhla/cnn-plastic-waste-classification).  

- **Notebooks:**  
  - [Week1-Libraries-Importing-Data-Setup.ipynb](Week1-Libraries-Importing-Data-Setup.ipynb)  
  - [Kaggle Notebook](https://www.kaggle.com/code/hardikksankhla/cnn-plastic-waste-classification)  

### Week 2: TBD  
*Details to be added after completion.*  

### Week 3: TBD  
*Details to be added after completion.*  

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Dineshvelaga/Waste-Clasification-using-CNN
   cd Waste-Clasification-using-CNN
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the training script:  *Details to be added after completion.*  
   ```bash  
   python train.py  
   ```  
4. For inference, use the following command:  *Details to be added after completion.*  
   ```bash  
   python predict.py --image_path /path/to/image.jpg  
   ```  

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  

## Future Scope  
- Expanding the dataset to include more plastic waste categories.  
- Deploying the model as a web or mobile application for real-time use.  
- Integration with IoT-enabled waste management systems.  



