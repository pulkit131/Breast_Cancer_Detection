# 🩺 Breast Cancer Mass Classification

This is a **Streamlit app** that uses a pretrained **DenseNet121** model to classify breast cancer masses from mammogram images.
Deployed URL : https://breastcancerdetection-efpcuhrumydn3ajyzhq7av.streamlit.app/

---

## 🚀 Features
- Upload an image (`.jpg`, `.jpeg`, `.png`).
- Model (`densenet121.h5`) predicts the **pathology class**.
- Displays prediction + confidence score.

---

## 📂 Project Structure
```
breastcancer/
│── app.py                   # Streamlit app
│── densenet121.h5           # Trained model (must be placed here)
│── label_encoder_pathology.pkl (optional) # Saved LabelEncoder
│── requirements.txt         # Dependencies
```

---

## 🔧 Installation

1. Clone this repo or copy the files into a folder.

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the trained model file (`densenet121.h5`) in the same folder as `app.py`.  
   If you are using the `.pkl` encoder, also place `label_encoder_pathology.pkl` here.

---

## ▶️ Running the App
Run the following command:
```bash
streamlit run app.py
```

You’ll see a local URL like:
```
Local URL: http://localhost:8501
```

Open it in your browser, upload an image, and view the prediction 🎉

---

