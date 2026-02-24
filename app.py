from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__, static_folder='static')

# ── Intent labels (8 classes) — index = class ID ───────────────────────────────
# {'class_schedule': 0, 'contact_info': 1, 'facilities': 2, 'location': 3,
#  'membership_fee': 4, 'offers': 5, 'timings': 6, 'trainer_info': 7}
INTENT_LABELS = [
    "class_schedule",   # 0
    "contact_info",     # 1
    "facilities",       # 2
    "location",         # 3
    "membership_fee",   # 4
    "offers",           # 5
    "timings",          # 6
    "trainer_info",     # 7
]

# ── Canned responses per intent ────────────────────────────────────────────────
RESPONSES = {
    "class_schedule": (
        "📅 We run a variety of group classes daily!\n"
        "• Morning: Yoga (6 AM), Spin (7 AM)\n"
        "• Afternoon: HIIT (12 PM), Pilates (2 PM)\n"
        "• Evening: Zumba (6 PM), Boxing (7 PM)\n"
        "Full schedules are posted on our app and at the front desk."
    ),
    "contact_info": (
        "📞 You can reach us at:\n"
        "• Phone: (555) 123-4567\n"
        "• Email: info@gymbot.com\n"
        "• Instagram: @GymBot\n"
        "Our team is available during gym hours to assist you!"
    ),
    "facilities": (
        "🏋️ Our facilities include:\n"
        "• Free weights & Olympic lifting platforms\n"
        "• Cardio zone (treadmills, bikes, rowers)\n"
        "• Cable machines & functional training area\n"
        "• Sauna, steam room & recovery lounge\n"
        "• Locker rooms with showers\n"
        "All equipment is sanitized regularly."
    ),
    "location": (
        "📍 We're located at 123 Fitness Blvd, Downtown.\n"
        "• Free parking in the lot behind the building\n"
        "• 2 blocks from Central Metro Station (Blue Line)\n"
        "Drop by anytime — we'd love to give you a tour!"
    ),
    "membership_fee": (
        "💳 Our membership plans:\n"
        "• Basic — $29/month (equipment access)\n"
        "• Pro — $49/month (equipment + group classes)\n"
        "• Elite — $79/month (everything + personal training)\n"
        "Day passes are $15. Student & senior discounts available!"
    ),
    "offers": (
        "🎉 Current offers & deals:\n"
        "• First month FREE for new members\n"
        "• Refer a friend → get 1 month at 50% off\n"
        "• Annual plan → save 20% vs monthly\n"
        "• Student discount: 15% off with valid ID\n"
        "Offers may be time-limited — ask at the front desk!"
    ),
    "timings": (
        "🕐 Gym timings:\n"
        "• Monday – Friday: 5 AM – 11 PM\n"
        "• Saturday: 6 AM – 10 PM\n"
        "• Sunday: 7 AM – 9 PM\n"
        "Holiday hours may vary — check our app for updates!"
    ),
    "trainer_info": (
    "🎯 Our certified personal trainers:\n"
    "• 1-on-1 and small-group sessions available\n"
    "• 30-min or 60-min session packages\n"
    "• Specialties: weight loss, muscle gain, rehab, sports performance\n\n"
    "👨 Male Trainer: Rahul Mehta – Strength & Conditioning\n"
    "👩 Female Trainer: Priya Sharma – Weight Loss & Functional Training\n\n"
    "✨ Both male and female trainers available upon request.\n"
    "Book a FREE introductory consultation at the front desk!"
)
}

# ── Load model (lazy, on first request) ────────────────────────────────────────
tokenizer = None
model = None
MODEL_PATH = os.environ.get("MODEL_PATH", "model/gym_model.pth")

def load_model():
    global tokenizer, model
    if tokenizer is not None:
        return
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(INTENT_LABELS)
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

def predict_intent(text: str) -> str:
    load_model()
    inputs = tokenizer(
        text, max_length=32, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        ).logits
    idx = torch.argmax(logits, dim=1).item()
    return INTENT_LABELS[idx]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400
    try:
        intent = predict_intent(user_msg)
        reply  = RESPONSES.get(intent, "I'm not sure about that. Please call us at (555) 123-4567 or email info@gymbot.com for help!")
        return jsonify({"reply": reply, "intent": intent})
    except Exception as e:
        return jsonify({"reply": "Sorry, something went wrong. Please try again.", "intent": "error"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
