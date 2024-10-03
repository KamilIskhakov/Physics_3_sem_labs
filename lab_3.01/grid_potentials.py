import matplotlib.pyplot as plt
"""
Скрипт для создания сетки с найденными точками. Идеально подойдет для тех,
у кого есть айпад, чтобы нарисовать нужные линии, не мучаясь c выставлениями
точек.
"""
# Размер сетки в сантиметрах
width_cm = 31
height_cm = 20

# Количество делений по горизонтали и вертикали (в сантиметрах)
x_divisions_cm = 10
y_divisions_cm = 10

# # Количество делений по горизонтали и вертикали (в миллиметрах)
# x_divisions_mm = x_divisions_cm * 10
# y_divisions_mm = y_divisions_cm * 10

# Создание сетки
fig, ax = plt.subplots(dpi=300) # Увеличение разрешения

# Определение границ сетки
ax.set_xlim(0, width_cm)
ax.set_ylim(0, height_cm)

# Добавление линий сетки в сантиметрах
for i in range(width_cm ):
  ax.axvline(i , color='lightgray', linestyle='-', linewidth=0.8)

for i in range(height_cm ):
  ax.axhline(i , color='lightgray', linestyle='-', linewidth=0.8)


# # Добавление линий сетки в миллиметрах
# for i in range(x_divisions_mm + 1):
#   ax.axvline(i * width_cm / x_divisions_mm, color='lightgrey', linestyle=':', linewidth=0.5)
#
# for i in range(y_divisions_mm + 1):
#   ax.axhline(i * height_cm / y_divisions_mm, color='lightgrey', linestyle=':', linewidth=0.5)

# Определение точек
points = [(3.7,18), (8, 18), (13.2, 18),(18, 18),
          (23, 18),(30, 18),(3.3, 16),(7.8, 16),
          (13.6, 16),(16.6, 16),(22, 16),(27.4, 16),
          (2, 14),(6, 14),(11, 14),(18.5, 14),
          (23.1, 14),(27.9, 14),(2.9, 12),(6.7, 12),
          (10.7, 12),(17.1, 12),(22.7, 12),(27.5, 12),
          (2.4, 10),(5.9, 10),(9.9, 10),(16.5, 10),
          (22.4, 10),(27.1, 10),(2.2, 8),(6.1, 8),
          (11, 8),(17, 8),(22.9, 8),(27.5, 8),
          (1.6, 6),(6, 6),(11.2, 6),(16.8, 6),
          (22.3, 6),(27.5, 6),(1, 4),(6.1, 4),(11.5, 4),
          (17, 4),(22.5, 4),(28, 4),(1.8, 2),(6.9, 2),
          (12.4, 2),(17.8, 2),(22.5, 2),(28.6, 2)]

# Отметка точек на сетке
for point in points:
  ax.plot(point[0], point[1], 'ro', markersize=2) # Изменение размера точек

# Установка заголовка графика
ax.set_title('Эквипотенциальные и силовые линии для плоского конденсатора')

# Установка меток осей с шагом 2
ax.set_xticks(range(0, int(width_cm) + 1, 2))
ax.set_yticks(range(0, int(height_cm) + 1, 2))

# Установка меток осей
ax.set_xlabel('X (см)')
ax.set_ylabel('Y (см)')

# Отображение графика
plt.show()
