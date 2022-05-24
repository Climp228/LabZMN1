import sklearn.cluster as scl, statistics as stat, random as rand
from matplotlib import pyplot as plt

# Задаємо к-сть елементів, які потрібно обробити
num_of_elements = 70
# Задаємо к-сть кластерів, яку ми хочемо отримати
num_of_clusters = 8


def most_common(array: list):
    temp = []
    counter = 0
    for i in range(0, len(array)):
        if array.count(array[i]) > counter:
            counter = array.count(array[i])
    for i in range(0, len(array)):
        if array.count(array[i]) == counter and array[i] not in temp:
            temp.append(array[i])
    return temp


# Згенеровуємо вибірку даних, за допомогою генератора випадкових чисел
array = []
x_array = []
y_array = []
for i in range(0, num_of_elements):
    array.append([rand.randint(0, num_of_elements), rand.randint(0, num_of_elements)])
print("Масив:\n", array, "\n")

# Вивід графіку випадково згенерованих міток
for i in range(0, num_of_elements):
    x_array.append(array[i][0])
    y_array.append(array[i][1])
plt.scatter(x_array, y_array)
plt.title("Випадково згенеровані мітки")
plt.show()

cl_array = scl.KMeans(n_clusters=num_of_clusters, random_state=0).fit(array)
cl_labels = cl_array.labels_
cl_centers = cl_array.cluster_centers_
print("Кластерні мітки:\n", cl_labels, "\n")
print("Центри кластерів:\n", cl_centers)

# Вивід у консоль значення кластерів
for i in range(0, num_of_clusters):
    temp = []
    temp2 = []
    temp3 = []
    for w in range(0, len(cl_labels)):
        if cl_labels[w] == i:
            temp.append(array[w])
            temp2.append(array[w][0])
            temp3.append(array[w][1])
    print("\nКластер " + str(i + 1) + ":\n", temp)
    print("X values:", temp2)
    print("Y values:", temp3)
    print("Min X:", min(temp2))
    print("Max X:", max(temp2))
    print("Mean X:", stat.mean(temp2))
    print("Median X:", stat.median(temp2))
    print("Most common X:", *most_common(temp2))
    print("Mode X:", stat.mode(temp2))
    print("Min Y:", min(temp3))
    print("Max Y:", max(temp3))
    print("Mean Y:", stat.mean(temp3))
    print("Median Y:", stat.median(temp3))
    print("Most common Y:", *most_common(temp3))
    print("Mode Y:", stat.mode(temp3))

    plt.scatter(temp2, temp3)
# Вивід на графік к-сть кластерів
plt.title(str(num_of_clusters) + " - кластерів")
plt.show()

# Вивід на графік центри кластерів
x_centers = []
y_centers = []
for i in range(0, num_of_clusters):
    x_centers.append(cl_centers[i][0])
    y_centers.append(cl_centers[i][1])
for i in range(0, num_of_clusters):
    plt.scatter(x_centers[i], y_centers[i])
plt.title(str(num_of_clusters) + " - центри кластерів")
plt.show()
