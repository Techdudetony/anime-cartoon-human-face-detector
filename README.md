# Anime-Cartoon-Human Face Detector 

A multi-class face image classifier that predicts whether an input image contains an **anime**, **cartoon**, or **real human** face using a Convolutional Neural Network (CNN).

---

## Dataset

This project uses custom-sampled data from Kaggle sources:

- **Anime Faces** – from [`splcher/animefacedataset`](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **Cartoon Faces** – from [`alhumayri/cartoon-classifier`](https://www.kaggle.com/datasets/alhumayri/cartoon-classifier)
- **Human Faces** – from [`ashwingupta3012/real-human-faces`](https://www.kaggle.com/datasets/ashwingupta3012/real-human-faces)

To avoid large downloads, each category is limited to 300 samples using `kagglehub`.

---

##  Features

-  Custom CNN built with TensorFlow/Keras
-  Dataset sampled and preprocessed from Kaggle via script
-  Organized into `dataset/` subfolders (`anime/`, `cartoon/`, `human/`)
-  Supports training, evaluation, and prediction
-  Visualizations for performance (accuracy, confusion matrix)

---

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/Techdudetony/anime-cartoon-human-face-detector.git
   cd anime-cartoon-human-face-detector

2. Create a virtual environment (*if not already present*)
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use: .venv\Scripts\activate

3. Install dependencies
    ```bash
    pip install -r requirements.txt

## Usage

1. Run the dataset downloader and sampling script:  
    ```bash
    python utils/download_sample.py

2. Train your model
    ```bash
    python main.py

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © [Techdudetony](https://github.com/Techdudetony)

---

## 🙌 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for the datasets  
- `kagglehub` for simplified API access  
- You, the reader, for exploring this repo! 😊
