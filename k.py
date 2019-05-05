import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve
from keras.callbacks import EarlyStopping

dataset1 = np.loadtxt("./data/Stroke_drop_rate2.csv", delimiter=',', skiprows=1)
dataset2 = np.loadtxt("./data/Stroke_drop_rate5.csv", delimiter=',', skiprows=1)

x_train_temp1 = dataset1[:, 1:]
x_train1 = np.delete(x_train_temp1, 8, 1)
y_train1 = dataset1[:, 9]

x_train2_temp = dataset2[:100000, 1:]
x_train2 = np.delete(x_train2_temp, 8, 1)
y_train2 = dataset2[:100000, 9]

x_train_merged = np.vstack([x_train1, x_train2])
y_train_merged = np.hstack([y_train1, y_train2])


x_test_temp = dataset2[100000:, 1:]
X_test = np.delete(x_test_temp, 8, 1)
Y_test = dataset2[100000:, 9]

x_val = X_test[:20000]
y_val = Y_test[:20000]
x_test = X_test[20000:]
y_test = Y_test[20000:]


model = Sequential()
model.add(Dense(12, input_dim=28, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping()
model.fit(x_train_merged, y_train_merged, epochs=50, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping])

scores = model.evaluate(x_test, y_test)


print('')
print('loss : ' + str(scores[0]))
print('accuray : ' + str(scores[1]))

"""
fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(x_test)[:])

plt.plot(fpr, tpr, 'o-')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver operating characteristic')
plt.show()
"""
