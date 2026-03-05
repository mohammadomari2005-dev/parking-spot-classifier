import pickle
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from prepare_data import load_data 
from config import *



def train():
    data, labels = load_data()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, stratify=labels)
    classifier = SVC()

    parameters = [{'C' : C_VALUES,
                  'gamma' : GAMMA_VALUES}]
    
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train)

    best_estimator = grid_search.best_estimator_
    y_prediction = best_estimator.predict(x_test)
    score = accuracy_score(y_prediction, y_test)

    # Train accuracy
    y_train_prediction = best_estimator.predict(x_train)
    train_score = accuracy_score(y_train_prediction, y_train)
    print('Train accuracy: {}%'.format(str(train_score * 100)))

    # Test accuracy
    print('Test accuracy: {}%'.format(str(score * 100)))

    # Precision & Recall
    print('\nClassification Report:')
    print(classification_report(y_test, y_prediction, target_names=CATEGORIES))


    pickle.dump(best_estimator, open(MODEL_PATH, 'wb'))
    print('Model saved to {}'.format(MODEL_PATH))


if __name__ == "__main__":
    train()