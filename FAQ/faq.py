
import tkinter as tk
from tkinter import scrolledtext
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQs for an online shopping store
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return eligible products within 7 days of delivery."
    },
    {
        "question": "How can I track my order?",
        "answer": "Open My Orders on your account to track your order."
    },
    {
        "question": "What payment methods are accepted?",
        "answer": "We accept credit cards, debit cards, UPI and net banking."
    },
    {
        "question": "How can I reset my password?",
        "answer": "Click Forgot Password on the login page and follow the instructions."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "Contact our support team through the Help section on our website."
    },
    {
        "question": "How long does delivery take?",
        "answer": "Delivery usually takes 3 to 7 business days."
    },
    {
        "question": "Can I cancel my order?",
        "answer": "You can cancel eligible orders from the My Orders section before shipping."
    },
    {
        "question": "Is cash on delivery available?",
        "answer": "Cash on delivery is available for eligible products and locations."
    }
]

# Text preprocessing using NLTK
def preprocess(text):
    text = text.lower()
    tokens = nltk.wordpunct_tokenize(text)
    tokens = [
        word for word in tokens
        if any(char.isalnum() for char in word)
    ]
    return " ".join(tokens)


# Prepare FAQ questions
questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]

# Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)


# Find the best matching answer
def get_answer(user_question):
    cleaned_question = preprocess(user_question)

    user_vector = vectorizer.transform([cleaned_question])

    similarities = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    best_match = similarities.argmax()
    best_score = similarities[best_match]

    # Reject weak matches
    if best_score < 0.20:
        return (
            "Sorry, I could not find a suitable answer. "
            "Please try asking in a different way."
        )

    return faqs[best_match]["answer"]


# Display messages in the chat window
def send_message(event=None):
    question = entry.get().strip()

    if not question:
        return

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + question + "\n")
    entry.delete(0, tk.END)

    if question.lower() in ["bye", "exit", "quit"]:
        answer = "Goodbye! Thank you for chatting with us."
    else:
        answer = get_answer(question)

    chat_area.insert(tk.END, "Bot: " + answer + "\n\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.see(tk.END)


# Create the graphical interface
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("550x500")
root.configure(bg="#eef2ff")

title = tk.Label(
    root,
    text="FAQ Chatbot",
    font=("Arial", 20, "bold"),
    bg="#eef2ff",
    fg="#3730a3"
)
title.pack(pady=10)

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 11),
    state=tk.DISABLED
)
chat_area.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root, bg="#eef2ff")
entry_frame.pack(fill=tk.X, padx=15, pady=10)

entry = tk.Entry(entry_frame, font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
entry.bind("<Return>", send_message)

send_button = tk.Button(
    entry_frame,
    text="Send",
    command=send_message,
    bg="#4f46e5",
    fg="white",
    font=("Arial", 11, "bold")
)
send_button.pack(side=tk.RIGHT)

chat_area.config(state=tk.NORMAL)
chat_area.insert(
    tk.END,
    "Bot: Hello! Welcome to our online store.\n"
    "Ask me about orders, payments, returns or delivery.\n\n"
)
chat_area.config(state=tk.DISABLED)

root.mainloop()
