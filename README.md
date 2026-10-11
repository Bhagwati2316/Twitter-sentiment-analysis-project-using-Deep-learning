# Twitter-sentiment-analysis-project-using-Deep-learning
Using Deep Learning Recurrent Neural Network Predicting the Sentiment of the Tweets of users


# 🐦 Twitter Tweets Sentiment Analysis using Deep Learning

A production-ready Deep Learning pipeline utilizing Recurrent Neural Networks (RNN) to accurately classify the sentiment of Twitter tweets into three strategic categories: **Negative**, **Neutral**, or **Positive**. This project includes a complete model development pipeline and an interactive deployment interface powered by Streamlit.

---

## 🚀 Project Overview & Architecture Evolution

During the engineering process, the model's architecture went through three clear development phases based on validation performance metrics:

*   **Phase 1 (The Baseline Model - 36% Accuracy):** The initial setup fed raw tokenized vocabulary integers directly into a standalone `SimpleRNN` layer (`input_shape=(99,1)`). This approach failed to break past a baseline accuracy of ~36% because the single-feature token space could not capture the **semantic relationships** or contextual alignment between words. 
*   **Phase 2 (The Semantic Translator - 98.2% Training Accuracy):** To capture semantic meaning, a dense **Keras Embedding Layer** was introduced right at the front. By transforming sparse token indices into rich, continuous 128-dimensional vector spaces, words with similar contextual meanings mapped closely together. This immediately broke the accuracy bottleneck, driving training accuracy up to **98.2%**.
*   **Phase 3 (The Overfitting Challenge - 68.4% Validation Performance):** While training metrics soared, evaluating the test logs revealed a classic overfitting plateau. The model effectively memorized the training sequences perfectly, but its generalize capacity on unseen test validation text stalled at **68.4%**. 

### Final Production Model Blueprint (`model_final`)
The final verified layout incorporates a strict `mask_zero=True` flag to optimize pipeline speeds by forcing the recurrent memory loop to skip empty padding arrays:

1.  **Input Layer:** Expects tokenized integer text arrays bounded at a maximum sequence length of **99 timesteps** (words).
2.  **Embedding Layer:** Maps the vocabulary index space into a continuous, dense 128-dimensional matrix while masking zero-padded tokens (`mask_zero=True`).
3.  **Hidden Recurrent Block:** A standalone `SimpleRNN` with **32 hidden neurons** that iteratively compresses sequential word matrices into a single 32-dimensional summary vector (`return_sequences=False`).
4.  **Output Layer:** A `Dense` layer with **3 structural nodes** driven by a **Softmax activation function** to deliver normalized class probability distributions.

---


### Live demo :  https://twitter-sentiment-analysis-project-using-deep-learning-34rkam8.streamlit.app/

<img width="1895" height="731" alt="Image" src="https://github.com/user-attachments/assets/58372666-7033-4f8b-b0fc-7c8fdf65b517" />

<img width="1530" height="720" alt="Image" src="https://github.com/user-attachments/assets/22d51e05-aba3-4f7b-8ce9-f701666ebe84" />

<img width="1602" height="667" alt="Image" src="https://github.com/user-attachments/assets/ba373d0a-da64-4934-b86e-0b9c2d6e621e" />


## 🛠️ Tech Stack & Requirements

The project dependencies are strictly managed within the isolated tracking environment:

*   **Deep Learning Framework:** TensorFlow 2.x / Keras
*   **Data Wrangling:** Pandas, NumPy, Scikit-Learn
*   **Application Deployment:** Streamlit
*   **Serialization Pipeline:** Pickle

To install the required libraries, run:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Directory Structure

```text
├── dataset/
│   └── twitter_training.csv  # Raw text corpus (sampled to 15,000 records)
├── Tokenizer.pkl             # Serialized text tokenizer map
├── model.h5                  # Final compiled trained Keras network parameters
├── streamlit_app.py          # Interactive web UI execution code script
├── requirements.txt          # Python dependency checklist configuration file
└── README.md                 # Project system documentation
```

---

## 💻 Web App Deployment & Inference Pipeline

The project features a lightweight interface built with **Streamlit** for real-time inference. When a user inputs a tweet, the text undergoes sequence padding to match the model's expected shape before computing target outputs.

### Running the App Locally:

Ensure your trained configuration parameters (`model.h5` and `Tokenizer.pkl`) are placed in the root directory, then execute the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

### Inference Flow:

1. Input custom text strings into the interface text area.
2. The pipeline converts text strings to integer token strings using the saved token mappings.
3. Pads sequences uniformly to a `maxlen=99` signature using standard `post` padding configurations.
4. Generates predictions using the compiled target probability matrix.

---

## 📈 Future Engineering Enhancements

To break past the 68.4% validation ceiling and mitigate the current overfitting bottleneck, upcoming iterations will implement the following features:
*   **Gated Recurrent Architectures:** Upgrading the memory cell block from `SimpleRNN` to an **LSTM** or **GRU** block to better maintain historical gradients over the 99-timestep window.
*   **Regularization Tuning:** Injecting an explicit `Dropout(0.3)` or `SpatialDropout1D` wrapper after the embedding vector transformations.
*   **Early Stopping:** Integrating Keras training callbacks to instantly halt weights processing when the `val_loss` starts to diverge.
