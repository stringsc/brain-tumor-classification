## Key Features
- **Machine Learning Models**: Utilized Xception and Convolutional Neural Networks (CNN) for classification.
- **High Accuracy**: Achieved 97% accuracy through model tuning and optimization.
- **Data Augmentation**: Enhanced robustness by applying data augmentation techniques to increase variability in training data.
- **Hyperparameter Optimization**: Fine-tuned hyperparameters such as learning rate, batch size, and model architecture to improve performance.

---

## Implementation Details
1. **Data Preparation**:
   - Preprocessed a dataset containing MRI scans of brain tumors.
   - Applied data augmentation to enhance training data diversity and prevent overfitting.

2. **Model Selection**:
   - Leveraged **Xception** and **CNN** models for classification.
   - Compared and optimized model performance using validation metrics.

3. **Optimization**:
   - Iteratively fine-tuned model parameters.
   - Increased batch sizes for better gradient stability during training.

4. **Evaluation**:
   - Achieved 97% accuracy on test data.
   - Evaluated robustness with saliency maps and probability distributions.

---

## Usage
### Requirements
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Matplotlib

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Use the Streamlit web application to classify tumors:
   ```bash
   streamlit run app.py
   ```

---

## Results
- **Accuracy**: 97%
- **Visualization**:
   - Saliency maps highlight regions of interest in MRI scans.
   - Probability bar graphs provide confidence scores for each category.

---
