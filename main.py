import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint


df = pd.read_csv(r"C:\hafta4\fraud-detection\data\raw\creditcard.csv")

print("### Shape ###")
print(df.shape)

print("\n### Types ###")
print(df.dtypes)

print("\n### Head ###")
print(df.head())

print("\n### Tail ###")
print(df.tail())

print("\n### Missing Values ###")
print(df.isnull().sum())

print(f"\nAny Missing? {df.isnull().any().any()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

print("\n### Quantiles ###")
print(df.describe([0, 0.5, 0.95, 0.99, 1]).T)

# 2. Basit İnceleme

print("\nClass Value Counts:")
print(df["Class"].value_counts())

df.hist(bins=30, figsize=(20, 20))
plt.show()


# 3. Özellik Ölçekleme

new_df = df.copy()
new_df["Amount"] = RobustScaler().fit_transform(new_df["Amount"].to_numpy().reshape(-1, 1))

time = new_df["Time"]
new_df["Time"] = (time - time.min()) / (time.max() - time.min())

# 4. Karıştırma ve Bölme

new_df = new_df.sample(frac=1, random_state=1)
train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]

print("\nTrain/Test/Val Class Counts:")
print(train["Class"].value_counts(), test["Class"].value_counts(), val["Class"].value_counts())

train_np, test_np, val_np = train.to_numpy(), test.to_numpy(), val.to_numpy()

x_train, y_train = train_np[:, :-1], train_np[:, -1]
x_test, y_test = test_np[:, :-1], test_np[:, -1]
x_val, y_val = val_np[:, :-1], val_np[:, -1]


# 5. Logistic Regression

logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(x_train, y_train)
print("\nLogistic Regression Report:")
print(classification_report(y_val, logistic_model.predict(x_val), target_names=["No Fraud", "Fraud"]))

# 6. Basit Neural Network

shallow_nn = Sequential()
shallow_nn.add(InputLayer((x_train.shape[1],)))
shallow_nn.add(Dense(2, activation="relu"))
shallow_nn.add(BatchNormalization())
shallow_nn.add(Dense(1, activation="sigmoid"))

checkpoint = ModelCheckpoint(filepath="shallow_nn.keras", save_best_only=True)
shallow_nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
shallow_nn.summary()

shallow_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=[checkpoint])

def neural_net_predictions(model, x):
    return (model.predict(x).flatten() > 0.5).astype(int)

print("\nShallow NN Report:")
print(classification_report(y_val, neural_net_predictions(shallow_nn, x_val), target_names=["No Fraud", "Fraud"]))


# 7. Diğer Modeller

rf_model = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf_model.fit(x_train, y_train)
print("\nRandom Forest Report:")
print(classification_report(y_val, rf_model.predict(x_val), target_names=["No Fraud", "Fraud"]))

gbc_model = GradientBoostingClassifier(n_estimators=50, learning_rate=1, random_state=0)
gbc_model.fit(x_train, y_train)
print("\nGradient Boosting Report:")
print(classification_report(y_val, gbc_model.predict(x_val), target_names=["No Fraud", "Fraud"]))

svc_model = LinearSVC(class_weight="balanced")
svc_model.fit(x_train, y_train)
print("\nLinear SVC Report:")
print(classification_report(y_val, svc_model.predict(x_val), target_names=["No Fraud", "Fraud"]))

# ======================
# 8. Veri Dengeleme
# ======================
not_frauds = new_df.query("Class == 0")
frauds = new_df.query("Class == 1")
balanced_df = pd.concat([not_frauds.sample(len(frauds), random_state=1), frauds]).sample(frac=1, random_state=1)

balanced_df_np = balanced_df.to_numpy()
x_train_b, y_train_b = balanced_df_np[:700, :-1], balanced_df_np[:700, -1]
x_test_b, y_test_b = balanced_df_np[700:842, :-1], balanced_df_np[700:842, -1]
x_val_b, y_val_b = balanced_df_np[842:, :-1], balanced_df_np[842:, -1]


# 9. Denge Sonrası Modeller

logistic_model_b = LogisticRegression(max_iter=500)
logistic_model_b.fit(x_train_b, y_train_b)
print("\nBalanced Logistic Regression Report:")
print(classification_report(y_val_b, logistic_model_b.predict(x_val_b), target_names=["No Fraud", "Fraud"]))

rf_model_b = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf_model_b.fit(x_train_b, y_train_b)
print("\nBalanced Random Forest Report:")
print(classification_report(y_val_b, rf_model_b.predict(x_val_b), target_names=["No Fraud", "Fraud"]))

gbc_model_b = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=2, random_state=0)
gbc_model_b.fit(x_train_b, y_train_b)
print("\nBalanced Gradient Boosting Report:")
print(classification_report(y_val_b, gbc_model_b.predict(x_val_b), target_names=["No Fraud", "Fraud"]))

svc_model_b = LinearSVC(class_weight="balanced")
svc_model_b.fit(x_train_b, y_train_b)
print("\nBalanced Linear SVC Report:")
print(classification_report(y_val_b, svc_model_b.predict(x_val_b), target_names=["No Fraud", "Fraud"]))
