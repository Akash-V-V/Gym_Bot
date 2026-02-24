# GymBot — BERT Chatbot Frontend

A full-stack chatbot interface for your BERT-based gym intent classifier.

## Project Structure

```
gymbot/
├── app.py              ← Flask backend
├── static/
│   └── index.html      ← Chat UI
└── model/
    └── gym_model.pth   ← Your trained BERT weights (place here)
```

## Setup

### 1. Install dependencies
```bash
pip install flask torch transformers
```

### 2. Place your model
Copy your trained model file to the `model/` folder:
```bash
mkdir model
cp /path/to/New_Bert_gym_model.pth model/gym_model.pth
```

Or set a custom path via environment variable:
```bash
export MODEL_PATH=/path/to/your/gym_model.pth
```

### 3. Run the server
```bash
python app.py
```

### 4. Open the chatbot
Visit **http://localhost:5000** in your browser.

---

## How It Works

- User types a message → sent to Flask `/chat` endpoint
- Flask tokenizes it with `BertTokenizer` and runs your trained model
- The model predicts one of **8 intents**:
  | ID | Intent |
  |----|--------|
  | 0  | gym_hours |
  | 1  | membership_info |
  | 2  | class_schedule |
  | 3  | equipment_info |
  | 4  | personal_training |
  | 5  | pricing |
  | 6  | location_directions |
  | 7  | general_inquiry |
- A pre-written response for that intent is returned and displayed

## Customizing Responses
Edit the `RESPONSES` dict in `app.py` to change what the bot says for each intent.

## Customizing Intent Labels
If your model's class order differs, update the `INTENT_LABELS` list in `app.py` to match the label encoding used during training.

---

## Model Training Source Code

The BERT intent classification model used in this project was created and trained using the notebook:

Bert_Gym_model.ipynb

### What the Notebook Contains

- Data preprocessing and label encoding
- Tokenization using BertTokenizer
- Model setup using BertForSequenceClassification
- Fine-tuning on a custom gym intent dataset
- Training and validation loop
- Model evaluation
- Saving trained model weights as a .pth file

After training, the final model weights were exported and saved as:

gym_model.pth

This file is loaded inside app.py during runtime for inference.

### To Re-train the Model

1. Open Bert_Gym_model.ipynb
2. Run all cells sequentially
3. Ensure the trained model is saved as:
   gym_model.pth
4. Move the file into the model/ directory before running the Flask application

---
