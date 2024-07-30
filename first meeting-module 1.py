from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Load the data
data = pd.read_csv('path_combine_final_outcome_vars 2.csv')

# Encode the 'fracture' column
label_encoder = LabelEncoder()
data['fracture'] = label_encoder.fit_transform(data['fracture'])

# Split the data into features and target
X = data.drop(columns=['fracture'])
y = data['fracture']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Function to create the model (required for KerasClassifier)
def create_model(neurons=32, dropout_rate=0.5, l2_reg=0.01, learning_rate=0.001, optimizer='adam'):
    model = Sequential([
        Dense(neurons, input_dim=12, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dense(neurons//2, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dense(neurons//4, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.AUC()])
    return model

# Create the KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150],
    'model__neurons': [64, 128, 256],
    'model__dropout_rate': [0.3, 0.5, 0.7],
    'model__l2_reg': [0.001, 0.01, 0.1],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__optimizer': ['adam', 'rmsprop']
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, verbose=1, n_jobs=-1)

# Fit the RandomizedSearchCV
random_search_result = random_search.fit(X_train, y_train)


# Base models with additional models like XGBoost and LightGBM
base_models = [
    ('nn', KerasClassifier(model=create_model, epochs=50, batch_size=32, verbose=0)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
]

# Meta-model
meta_model = LogisticRegression()

# Stacking Classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the Stacking Classifier
stacking_clf.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC: {auc_roc}")
