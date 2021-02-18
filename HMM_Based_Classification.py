# import necessary libraries
import pandas as pd  # for data reading and manipulation
import numpy as np  # for data reading and manipulation
import matplotlib.pyplot as plt  # for plotting data
from sklearn.metrics import f1_score, accuracy_score  # for accuracy
from sklearn.decomposition import PCA  # for dimensionality reduction
from hmmlearn import hmm  # solving problem using Hidden Markov Model

# read the train data
train_data = pd.read_csv('train.csv')

# read the test data
test_data = pd.read_csv('test.csv')

walk_down_up = train_data[train_data['Activity'] == 'WALKING_UPSTAIRS']
train_data = train_data.drop(walk_down_up.index, axis=0)
walk_down_up = train_data[train_data['Activity'] == 'WALKING_DOWNSTAIRS']
train_data = train_data.drop(walk_down_up.index, axis=0)

walk_up = test_data[test_data['Activity'] == 'WALKING_UPSTAIRS']
test_data.drop(walk_up.index, axis=0, inplace=True)
walk_down = test_data[test_data['Activity'] == 'WALKING_DOWNSTAIRS']
test_data.drop(walk_down.index, axis=0, inplace=True)

# train_data
# test_data

# ___________________________________________________________________________________________________________________checkpoint

data = train_data.iloc[:, 0:561].values  # get values for different columns
# print(data.shape)

cov_data = np.cov(data, rowvar=False)  # find covariance
print(np.linalg.det(cov_data))  # determinant of covariance is zero for independent data

# perform dimensionality reduction to remove data with less information using PCA (Principal Component Analysis)
dim_reduce_pca = PCA(n_components=100)
cov_pca = dim_reduce_pca.fit(train_data.iloc[:, 0:561].values)

data_train_pca = cov_pca.transform(train_data.iloc[:, 0:561].values)
df_train_red = pd.DataFrame(data_train_pca)
# print(data_train_pca) 

# ___________________________________________________________________________________________________________________checkpoint

df_train_red['Subject'] = train_data['subject']
df_train_red['Activity'] = train_data['Activity']
df_train_red.dropna(inplace=True)
# df_train_red.tail()

df_train_red_STAND = df_train_red[df_train_red['Activity'] == 'STANDING']
df_train_red_SIT = df_train_red[df_train_red['Activity'] == 'SITTING']
df_train_red_LAY = df_train_red[df_train_red['Activity'] == 'LAYING']
df_train_red_WALK = df_train_red[df_train_red['Activity'] == 'WALKING']

print(df_train_red_STAND.shape)
print(df_train_red_SIT.shape)
print(df_train_red_LAY.shape)
print(df_train_red_WALK.shape)

test_data.dropna(inplace=True)

data_test_red = cov_pca.transform(test_data.iloc[:, 0:561].values)
df_test_red = pd.DataFrame(data_test_red)

df_test_red['Subject'] = test_data['subject']
df_test_red['Activity'] = test_data['Activity']
df_test_red.dropna(inplace=True)

# calculating true labels
labels_true = []
# count = 0
for i in range(df_test_red.shape[0]):
    if df_test_red['Activity'].iloc[i] == 'STANDING':
        labels_true.append(0)
    elif df_test_red['Activity'].iloc[i] == 'SITTING':
        labels_true.append(1)
    elif df_test_red['Activity'].iloc[i] == 'LAYING':
        labels_true.append(2)
    elif df_test_red['Activity'].iloc[i] == 'WALKING':
        labels_true.append(3)
    else:
        print(df_test_red['Activity'].iloc[i])
    # count += 1
labels_true = np.array(labels_true)


# print(count)


# implementing hmm
# since there are 4 activity so fitting hmm for each activity
def HMM_F1score(N, M, labels_true):
    hmm_stand = hmm.GMMHMM(n_components=N, n_mix=M, covariance_type='diag')
    hmm_sit = hmm.GMMHMM(n_components=N, n_mix=M, covariance_type='diag')
    hmm_lay = hmm.GMMHMM(n_components=N, n_mix=M, covariance_type='diag')
    hmm_walk = hmm.GMMHMM(n_components=N, n_mix=M, covariance_type='diag')

    hmm_stand.fit(df_train_red_STAND.iloc[:, 0:100].values)
    hmm_sit.fit(df_train_red_SIT.iloc[:, 0:100].values)
    hmm_lay.fit(df_train_red_LAY.iloc[:, 0:100].values)
    hmm_walk.fit(df_train_red_WALK.iloc[:, 0:100].values)

    # calculating F1 score
    labels_predict = []
    for i in range(len(df_test_red)):
        log_likelihood_value = np.array([hmm_stand.score(df_test_red.iloc[i, 0:100].values.reshape((1, 100))),
                                         hmm_sit.score(df_test_red.iloc[i, 0:100].values.reshape((1, 100))),
                                         hmm_lay.score(df_test_red.iloc[i, 0:100].values.reshape((1, 100))),
                                         hmm_walk.score(df_test_red.iloc[i, 0:100].values.reshape((1, 100)))])
        labels_predict.append(np.argmax(log_likelihood_value))
    labels_predict = np.array(labels_predict)

    F1 = f1_score(labels_true, labels_predict, average='micro')
    acc = accuracy_score(labels_true, labels_predict)
    return F1, acc


states = np.arange(1, 20, 1)

F1_value_states = []
acc_value_states = []
max_acc = (0, 0, 0)
for i in states:
    print("HMM has been trained for num_states= {}".format(i))
    f1, acc = HMM_F1score(i, 1, labels_true)
    if (f1, acc) > max_acc:
        max_acc = (f1, acc, i)
    F1_value_states.append(f1)
    acc_value_states.append(acc)
fig, ax = plt.subplots(2, 1)
print(max_acc)

ax[0].plot(F1_value_states)
ax[1].plot(acc_value_states)

plt.show()

f_test = []
acc_test = []

for i in range(1, 6):
    f1, acc1 = HMM_F1score(3, i, labels_true)
    f_test.append(f1)
    acc_test.append(acc1)

fig, ax = plt.subplots(2, 1)

ax[0].plot(f_test)
ax[1].plot(acc_test)

plt.show()

f1_val, acc_val = HMM_F1score(3, 8, labels_true)
print(f1_val)
print(acc_val)
