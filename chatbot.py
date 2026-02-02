import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "Hello, how can I help you?",
    "Python is a high-level programming language used for AI, machine learning, and data science.",
    "Python is easy to learn and has simple syntax.",
    "Python supports multiple programming paradigms.",
    "NLP stands for Natural Language Processing.",
    "Natural Language Processing helps computers understand human language.",
    "AI stands for Artificial Intelligence.",
    "Thank you for chatting with me.",
    "Goodbye! Have a great day."
]

greeting_inputs = ("hello", "hi", "hey", "greetings")
greeting_responses = ["Hi there!", "Hello!", "Hey!", "Greetings!"]

def greet(sentence):
    for word in sentence.lower().split():
        if word in greeting_inputs:
            return random.choice(greeting_responses)

def chatbot_response(user_input):
    user_input = user_input.lower()
    corpus.append(user_input)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf[-1], tfidf).flatten()
    corpus.remove(user_input)


    threshold = 0.2

    responses = [
        corpus[i]
        for i, score in enumerate(similarity[:-1])
        if score > threshold
    ]

    if not responses:
        return "Sorry, I didn't understand that."
    else:
        return "\n".join(responses)

print("🤖 AI Chatbot: Hello! Type 'bye' to exit.")

while True:
    user_text = input("You: ")

    if user_text.lower() == "bye":
        print("🤖 AI Chatbot: Goodbye! 👋")
        break
    elif greet(user_text):
        print("🤖 AI Chatbot:", greet(user_text))
    else:
        print("🤖 AI Chatbot:\n", chatbot_response(user_text))
