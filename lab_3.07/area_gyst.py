import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology, measure

# Масштабы
horizontal_division_ms = 50  # 50 мс на одно деление
vertical_division_mv = 50  # 50 мВ на одно деление

# Масштаб времени и напряжения на пиксель
x_scale = horizontal_division_ms  # мс/пиксель
y_scale = vertical_division_mv     # мВ/пиксель

# Загрузка изображения
I_in = io.imread('other_gyst.png')

# Удаление альфа-канала, если он присутствует
if I_in.shape[-1] == 4:
    I_in = I_in[..., :3]

# Преобразование в черно-белое изображение
I_gray = color.rgb2gray(I_in)

# Преобразование в бинарное изображение с порогом яркости 180
threshold = 180 / 255
I_bw = I_gray > threshold


# Восстановление разрывов петли (дилатация)
BWd1 = morphology.dilation(I_bw)

# Фильтрация объектов меньше 100 пикселей
BWd2 = morphology.remove_small_objects(BWd1, min_size=25)

# Заполнение области внутри петли (дилатация и эрозия)
disk_footprint = morphology.disk(50)
BWd3 = morphology.dilation(BWd2, footprint=disk_footprint)
BWd4 = morphology.erosion(BWd3, footprint=disk_footprint)

# Отображение результата
plt.figure()
plt.imshow(BWd4, cmap='gray')
plt.title('Processed Image')
plt.axis('off')
plt.show()

# Вычисление площади в пикселях
regions = measure.regionprops(morphology.label(BWd4))
areas_pixels = [region.area for region in regions]

# Перевод площади в мс × мВ
area_ms_mv = [area * (x_scale * y_scale) for area in areas_pixels]

print(f"Areas in pixels: {areas_pixels}")
print(f"Areas in ms × mV: {sum(area_ms_mv)*0.0000001}")