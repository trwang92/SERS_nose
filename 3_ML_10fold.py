import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

import os

# dataloader and extract the range of 1300 cm⁻¹ to 1400 cm⁻¹ for features
def load_from_json(datapath):
    sensor = []
    label_lst = []
    sensor_data = []

    with open(datapath, 'r', encoding="utf8") as fin:
        data = eval(fin.read())
        for item in data:
            label = int(item['label'])
            label_lst.append(label)

            item_id = item['id']
            item_type = item['type']

            a = item['a']
            b = item['b']
            c = item['c']
            d = item['d']
            e = item['e']
            f = item['f']
            # sensor_lst = []
            # sensor_lst.extend(a)
            # sensor_lst.extend(b)
            # sensor_lst.extend(c)
            # sensor_lst.extend(d)
            # sensor_lst.extend(e)
            # sensor_lst.extend(f)

            sensor_lst = []
            sensor_a = []
            sensor_b = []
            sensor_c = []
            sensor_d = []
            sensor_e = []
            sensor_f = []

            indices = range(771 - 578, 839 - 578)  # index range of 1300 cm⁻¹ to 1400 cm⁻¹

            for idx in indices:
                if idx < len(a):
                    sensor_a.append(a[idx])
                if idx < len(b):
                    sensor_b.append(b[idx])
                if idx < len(c):
                    sensor_c.append(c[idx])
                if idx < len(d):
                    sensor_d.append(d[idx])
                if idx < len(e):
                    sensor_e.append(e[idx])
                if idx < len(f):
                    sensor_f.append(f[idx])

            sensor_lst.extend(sensor_a)
            sensor_lst.extend(sensor_b)
            sensor_lst.extend(sensor_c)
            sensor_lst.extend(sensor_d)
            sensor_lst.extend(sensor_e)
            sensor_lst.extend(sensor_f)
            sensor.append(sensor_lst)

            sensor_data.append({
                "id": item_id,
                "label": label,
                "type": item_type,
                "a": sensor_a,
                "b": sensor_b,
                "c": sensor_c,
                "d": sensor_d,
                "e": sensor_e,
                "f": sensor_f,
            })


    return np.array(sensor), np.array(label_lst), sensor_data

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding="utf8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)

file_list = os.listdir('./data/compare')

temperature1 = '25'
temperature2 = '25'

# BYS means 2,4-DNPA
test_object1 = 'TNT'
test_object2 = 'BYS'

sensor = 'all'

# 假设数据文件路径为 'data.json'
datapath = './data/compare/' + temperature1 + test_object1 + temperature2 + test_object2 + '/augmented_data.json'

# filtered_data_out_path = './data/compare/' + temperature1 + test_object1 + temperature2 + test_object2 + '/1300-1400.json'

X, y, filtered_data = load_from_json(datapath)

# save_to_json(filtered_data, filtered_data_out_path)

# define the model list
models = {
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# store the average accuracy and confusion matrix for each model
mean_accuracies = {}
confusion_matrices = {}
roc_curves = {}
# store the precision-recall data for each model
pr_curves = {}

# 10-fold cross validation
kf = KFold(n_splits=10)

for model_name, model in models.items():
    accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # training
        model.fit(X_train, y_train)

        # validation
        y_pred = model.predict(X_val)
        y_score = model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

        all_y_true += list(y_val)
        all_y_pred += list(y_pred)
        all_y_scores += list(y_score)

    # mean accuracy
    mean_accuracy = np.mean(accuracies)
    mean_accuracies[model_name] = mean_accuracy
    print(f"{model_name} Mean Validation Accuracy over 10 folds: {mean_accuracy:.4f}")

    # confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    confusion_matrices[model_name] = cm

    # ROC curve
    fpr, tpr, _ = roc_curve(y, all_y_scores)
    roc_auc = auc(fpr, tpr)
    roc_curves[model_name] = (fpr, tpr, roc_auc)

    # precision and recall
    # precision, recall, _ = precision_recall_curve(y_bin[:, 1], all_y_scores)
    # average_precision = average_precision_score(y_bin[:, 1], all_y_scores)
    # pr_curves[model_name] = (precision, recall, average_precision)

    # plot Precision-Recall Curve
    def plot_precision_recall_curve(precision, recall, average_precision, model_name, file_name):
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label='Average Precision = {:.2f}'.format(average_precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            'Precision-Recall Curve of the ' + model_name + ' testing different substances at' + temperature1 + ' degrees Celsius with Sensor ' + sensor)
        plt.legend(loc='best')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.grid()
        # 保存为高质量 PNG
        plt.savefig(
            './data/compare/' + file_name + '/' + sensor + '-' + file_name + '-' + model_name + '-PR.png',
            dpi=300)
        plt.show()

# plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix for {model_name} at ' + temperature1 + ' degrees Celsius with Sensor ' + sensor)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('./data/compare/' + temperature1 + test_object1 + temperature2 + test_object2 + '/' + sensor + '-' + temperature1 + test_object1 + temperature2 + test_object2 + '-' + model_name + '-confusion.png', dpi=300)
    plt.show()

# plot roc curve
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for ' + model_name + ' testing different substances at ' + temperature1 + ' degrees Celsius with Sensor ' + sensor)

    plt.savefig(
        './data/compare/' + temperature1 + test_object1 + temperature2 + test_object2 + sensor + '-'  + temperature1 + test_object1 + temperature2 + test_object2 + '-' + model_name + '-roc.png',
        dpi=300)
    plt.legend(loc='lower right')
    plt.show()

# plot all confusion matrixes and roc curves
for model_name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, model_name)

for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
    plot_roc_curve(fpr, tpr, roc_auc, model_name)

# plot all Precision-Recall Curves
# for model_name, (precision, recall, average_precision) in pr_curves.items():
#     plot_precision_recall_curve(precision, recall, average_precision, model_name)
