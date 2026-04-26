import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")

# Label
data_fake["class"] = 0
data_true["class"] = 1

# Remove last 10 rows
data_fake = data_fake.iloc[:-10]
data_true = data_true.iloc[:-10]

# Combine
data = pd.concat([data_fake, data_true], axis=0)
data = data[["text", "class"]]
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

data["text"] = data["text"].apply(wordopt)

# Split
x = data["text"]
y = data["class"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

# Vectorize
vectorization = TfidfVectorizer(stop_words="english", max_df=0.7)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train
LR = LogisticRegression(class_weight="balanced", max_iter=1000)
LR.fit(xv_train, y_train)

# Evaluate
pred_lr = LR.predict(xv_test)
print("Accuracy:", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

# Save
pickle.dump(LR, open("model.pkl", "wb"))
pickle.dump(vectorization, open("vectorizer.pkl", "wb"))
print("✅ model.pkl and vectorizer.pkl saved successfully!")