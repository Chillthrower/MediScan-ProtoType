import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
from django.conf import settings
import os
DATA_PATH = os.path.join(settings.BASE_DIR, "app", "Training.csv")
data = pd.read_csv(DATA_PATH).dropna(axis=1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)
# dump(final_svm_model, 'final_svm_model.joblib')
# dump(final_nb_model, 'final_nb_model.joblib')
# dump(final_rf_model, 'final_rf_model.joblib')
symptoms = X.columns.values
prediction_classes = encoder.classes_
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": prediction_classes
}

def predictDisease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.capitalize())
        if index is not None:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = final_rf_model.predict(input_data)
    nb_prediction = final_nb_model.predict(input_data)
    svm_prediction = final_svm_model.predict(input_data)

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

    rf_disease_name = data_dict["predictions_classes"][rf_prediction[0]]
    nb_disease_name = data_dict["predictions_classes"][nb_prediction[0]]
    svm_disease_name = data_dict["predictions_classes"][svm_prediction[0]]
    final_disease_name = data_dict["predictions_classes"][final_prediction]

    predictions = {
        "rf_model_prediction": rf_disease_name,
        "naive_bayes_prediction": nb_disease_name,
        "svm_model_prediction": svm_disease_name,
        "final_prediction": final_disease_name
    }
    return predictions
print(predictDisease("itching,vomiting,yellowish_skin"))