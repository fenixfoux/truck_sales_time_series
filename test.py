import matplotlib.pyplot as plt

# Пример данных
height = [160, 170, 155, 180, 165]
weight = [60, 70, 55, 75, 62]

plt.scatter(height, weight)
plt.xlabel('Рост')
plt.ylabel('Вес')
plt.title('Диаграмма рассеяния Рост vs Вес')
plt.show()

