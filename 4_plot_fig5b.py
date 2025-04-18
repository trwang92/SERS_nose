import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_data(file_path):

    matrix = []
    with open(file_path) as fileob:
        lines = fileob.readlines()

        # 找到第一个有效的 data2 值的行索引
        start_index = None
        for i in range(89, len(lines)):
            data = lines[i].split(';')
            if data[3] != '' and data[3].replace('.', '', 1).isdigit():
                start_index = i
                break

        if start_index is not None:
            for line in lines[start_index:]:
                data = line.split(';')
                if data[3] != '':
                    data2_value = float(data[3])
                    if 1299.57 <= data2_value <= 1400.28:
                        matrix.append(data)
                    elif data2_value > 1400.28:
                        break

    matrix = np.array(matrix)

    if matrix.size > 0:

        data1 = np.where(matrix[:, 7] == '', np.nan, matrix[:, 7]).astype(float)  # Dark Subtracted
        data2 = np.where(matrix[:, 3] == '', np.nan, matrix[:, 3]).astype(float)  # Raman Shift

        for i in range(len(data2)):
            print(f"data2 = {data2[i]}, data1 = {data1[i]}")
    else:
        print("error")

    return data2, data1

# plot splicing spectra
def plot_sensor(data2, data1, sensor, color):

    plt.figure(figsize=(6, 10))
    plt.plot(data2, data1, linestyle='-', color=color)

    plt.xlim(1300, 1400)
    plt.ylim(min(data1), max(data1) + 10)

    plt.xticks(np.arange(1300, 1410, 20))
    plt.yticks(np.arange(int(min(data1) // 100) * 100, int(max(data1) // 100) * 100 + 100, 100))

    plt.title(sensor)
    plt.xlabel('Raman Shift')
    plt.ylabel('Dark Subtracted')

    plt.savefig('./fig5_b/' + sensor + '.eps', format='eps')

    plt.show()


# data path of the sensor
file_path_f = './data/25duTNT_original/20220428-f-TNT-25du-20min-001.txt'

x_f, y_f = extract_data(file_path_f)

colors = ['b', 'g', 'r', 'c', 'm', 'y']  # different colors

plot_sensor(x_f, y_f, 'Sensor b', colors[1])
