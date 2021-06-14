

data_file = open("/Users/roman/Yandex.Disk-Ioannis05.localized/Self-government/Neuronet/mnist_test/mnist_test.csv", "r")
data_list = data_file.readlines()
all_values = data_list[0].split(',')
# Не забыть закрыть файл во избежание ошибок, случайных изменений и зависаний
data_file.close()



image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
# Проверка на работу
# Подготовка данных для нашей сети и их загрузка туда
# Переводим значения цветовых кодов из диапазона 0-255 в диапазон 0.01-1.0
scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# print(scaled_input)


# количество выходных узлов - 10
onodes = 10
targets = np.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)


print(targets)
print(n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))



