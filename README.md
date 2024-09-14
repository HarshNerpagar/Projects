# Spam Classification Using Machine Learning and Deep Learning

In this notebook, we explore two different approaches to classify emails as spam or legitimate: a traditional machine learning method (Logistic Regression) and a deep learning method (Recurrent Neural Network, RNN).

## 1. Traditional Machine Learning Approach

### Step 1: Data Preparation

1. **Download the Dataset**
   We use publicly available datasets for spam and legitimate emails from the SpamAssassin public corpus.

2. **Extract the Dataset**
   The datasets are compressed in tar.bz2 format. We extract them to access the email files.

3. **Load and Combine Data**
   Emails are read from the extracted directories and labeled as 'phishing' (spam) or 'legitimate' (ham). The data is combined into a single DataFrame.

### Step 2: Text Preprocessing

1. **Tokenization and Cleaning**
   The email text is tokenized into words, and non-alphabetical tokens are removed. The text is converted to lowercase, and stopwords are removed.

2. **Lemmatization**
   Words are lemmatized to their base forms using NLTK’s `WordNetLemmatizer`.

### Step 3: Feature Extraction

1. **TF-IDF Vectorization**
   We convert the cleaned text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) to represent the importance of words in the emails.

### Step 4: Model Training

1. **Train-Test Split**
   The dataset is split into training and testing sets.

2. **Logistic Regression**
   We train a Logistic Regression model on the training data.

### Step 5: Evaluation

1. **Model Evaluation**
   We evaluate the model's performance using accuracy, precision, recall, and F1 score on the test set.

2. **Prediction**
   We use the trained model to predict whether new emails are spam or legitimate.

## 2. Deep Learning Approach (Recurrent Neural Network - RNN)

### Step 1: Text Preprocessing for Deep Learning

1. **Tokenization and Sequencing**
   The text is tokenized, and each word is converted to an integer index. The sequences are padded to ensure uniform length.

2. **Label Encoding**
   Email labels are encoded into numerical format for model training.

### Step 2: Create TensorFlow Datasets

1. **Convert to TensorFlow Datasets**
   We create TensorFlow datasets for training and testing from the preprocessed sequences and labels.

### Step 3: Build and Train the RNN Model

1. **Model Architecture**
   - **Embedding Layer:** Converts word indices into dense vectors of fixed size.
   - **SimpleRNN Layers:** Apply recurrent layers to capture temporal dependencies in the email sequences.
   - **Dense Layer:** Outputs the final classification score.

2. **Compile and Train**
   The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is then trained on the training dataset.

### Step 4: Evaluation

1. **Model Evaluation**
   We evaluate the RNN model's performance on the test set using accuracy, precision, and F1 score.

2. **Prediction**
   We use the trained RNN model to classify new emails.

### Example Usage

To predict whether a new email is spam or legitimate using the Logistic Regression model:

```python
new_email = "Congratulations! You've won a free prize. Click here to claim."
print(f'This email is: {predict_email(new_email)}')
