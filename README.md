# MLmodels

A curated collection of Jupyter Notebook implementations for common machine learning tasks, including image classification, natural language processing, time series forecasting, and medical diagnostics. Each notebook demonstrates end-to-end workflows—data loading, preprocessing, model building, training, evaluation, and visualization—using popular Python libraries.

## Table of Contents

1. [Projects](#projects)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Environment & Dependencies](#environment--dependencies)  
5. [Repository Structure](#repository-structure)  
6. [Contributing](#contributing)  
7. [License](#license)  

## Projects

### 1. MNIST Digit Classification using Neural Network  
An introduction to image classification. Builds a fully connected neural network on the MNIST dataset to recognize handwritten digits.

### 2. Digit Classification (Scikit-Learn)  
Applies classical machine learning algorithms (e.g., SVM, Random Forest) to the MNIST digit dataset for performance comparison.

### 3. Breast Cancer Detection  
Binary classification to predict malignant vs. benign tumors using the UCI Breast Cancer Wisconsin dataset. Covers feature selection, model tuning, and ROC analysis.

### 4. Sentiment Analysis on 1.6 Million Tweets  
Performs text preprocessing, tokenization, and trains a sentiment classifier on a large-scale tweets dataset using NLTK and scikit-learn.

### 5. Exploring Tokenization Techniques with NLTK  
Compares different tokenization strategies (word, sentence, regex) and their impact on downstream NLP tasks.

### 6. RNN Model for Stock Data  
Time series forecasting of stock prices with a simple recurrent neural network. Includes data normalization, sequence preparation, and performance metrics.

### 7. COVID-19 Detection  
Uses image data (e.g., chest X-rays) to train a convolutional neural network for automated detection of COVID-19 infection.

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/sparshdubey95/MLmodels.git
   cd MLmodels
   ```

2. (Recommended) Create a virtual environment:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** If `requirements.txt` is not present, install common libraries:  
   > ```bash
   > pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn nltk
   > ```

## Usage

1. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
2. Open the notebook of interest (e.g., `MNIST_Digit_classification_using_NN.ipynb`).
3. Run each cell sequentially.  
4. Review results and modify hyperparameters as desired.

## Environment & Dependencies

- Python 3.7+  
- Jupyter Notebook  
- Core libraries:  
  - numpy  
  - pandas  
  - scikit-learn  
  - tensorflow, keras  
  - matplotlib, seaborn  
  - nltk  

## Repository Structure

| File/Folder                                                 | Description                                                    |
|-------------------------------------------------------------|----------------------------------------------------------------|
| MNIST_Digit_classification_using_NN.ipynb                   | Neural network on MNIST digit dataset                          |
| digit_classification.ipynb                                  | Classical ML models (SVM, RF) for MNIST                        |
| breast_cancer_detection.ipynb                               | Tumor classification with UCI Breast Cancer dataset            |
| Sentiment_Analysis_on_1_6_Million_Tweets.ipynb              | Sentiment classifier on tweets                                 |
| Exploring_Tokenization_Techniques_with_NLTK.ipynb           | Comparison of NLTK tokenizers                                  |
| RNN_Model_for_Stock_Data.ipynb                              | Recurrent neural network for stock price prediction            |
| Covid_19_Detection.ipynb                                    | CNN for COVID-19 detection from medical images                 |
| README.md                                                   | Project overview and instructions                              |

## Contributing

Contributions, issues, and feature requests are welcome!  
1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/foo`)  
3. Commit your changes (`git commit -m 'Add foo feature'`)  
4. Push to the branch (`git push origin feature/foo`)  
5. Open a Pull Request  

## License

Distributed under the MIT License. See `LICENSE` for more information.

[1] https://github.com/sparshdubey95/MLmodels/blob/main/Covid_19_Detection.ipynb
[2] https://github.com/sparshdubey95/MLmodels/blob/main/Exploring_Tokenization_Techniques_with_NLTK.ipynb
[3] https://github.com/sparshdubey95/MLmodels/blob/main/MNIST_Digit_classification_using_NN.ipynb
[4] https://github.com/sparshdubey95/MLmodels/blob/main/README.md
[5] https://github.com/sparshdubey95/MLmodels/blob/main/RNN_Model_for_Stock_Data.ipynb
[6] https://github.com/sparshdubey95/MLmodels/blob/main/Sentiment_Analysis_on_1_6_Million_Tweets.ipynb
[7] https://github.com/sparshdubey95/MLmodels/blob/main/breast_cancer_detection.ipynb
[8] https://github.com/sparshdubey95/MLmodels/blob/main/digit_classification.ipynb
