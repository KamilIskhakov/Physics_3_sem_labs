import numpy as np
import matplotlib.pyplot as plt
from pylatex import Section, Tabular, NoEscape, Figure, Table, MultiColumn
from pylatex.package import Package
from pylatex import Document
import math

class LabReport:
    def __init__(self, filename, font_size=26):
        self.doc = Document(geometry_options={"left": "3cm", "right": "3cm", "bottom": "2cm", "top": "1cm"})
        self.filename = filename
        self.font_size = font_size

        # Добавляем пакеты для поддержки кириллицы и математики
        self.doc.packages.append(Package('babel', options='russian'))
        self.doc.packages.append(Package('amsmath'))
        self.doc.packages.append(Package('amssymb'))
        self.doc.packages.append(Package('amsfonts'))
        self.doc.packages.append(Package('geometry'))
        self.doc.packages.append(Package('mathtext'))

        # Инициализация переменных класса
        self.Bc_values = []
        self.e_m_values = []
        self.U_values = []
        self.delta_U_values = []
        self.delta_B_c_values = []
        self.delta_r_a_values = []
        self.critical_points = []


    def create_title_page(self, title, author):
        """
        Функция для создания титульного листа.
        :param title: Название лабораторной.
        :param author: Выполнявшие данную лабораторную работу.
        """
        # Центрирование и установка заголовка
        self.doc.append(NoEscape(r'\begin{center}'))
        self.doc.append(NoEscape(r'.\\'))
        self.doc.append(NoEscape(r'\vspace{12cm}'))
        self.doc.append(NoEscape(r'\textbf{' + title + '}'))
        self.doc.append(NoEscape(r'\end{center}'))

        # Центрирование и добавление автора
        self.doc.append(NoEscape(r'\begin{center}'))
        self.doc.append(author)
        self.doc.append(NoEscape(r'\end{center}'))

        # Добавляем дату
        self.doc.append(NoEscape(r'\begin{center}'))
        self.doc.append(NoEscape(r'\today'))
        self.doc.append(NoEscape(r'\end{center}'))

        self.doc.append(NoEscape(r'\newpage'))

    def add_text_with_math(self, text, math_expr=None):
        """
        Функция для указания названия формулы и ее самой в формате displaymath.
        :param text: Название формулы.
        :param math_expr: Нужная формула; Записываете ее как если бы записывали в латехе (причем без долларов), но перед "" обязательно поставьте r (r"").
        """
        if math_expr:
            self.doc.append(NoEscape(text + r"\begin{displaymath}" + math_expr + r"\end{displaymath}"))
        else:
            self.doc.append(text)
        self.doc.append(NoEscape(r'\newline'))

    def create_summary(self):
        """
        Функция для создания подзаголовка с итогами.
        """
        with self.doc.create(Section('Выводы')):
            self.doc.append("")

    def create_data(self):
        """
        Функция для создания подзаголовка с используемыми формулами.
        """
        with self.doc.create(Section('Основные формулы:')):
            self.doc.append("")

    def create_calculus(self):
        """
        Функция для создания подзаголовка расчетов.
        """
        with self.doc.create(Section('Расчеты:')):
            self.doc.append("")

    def lab_construction(self, image_filename, caption):
        """
        Функция для создания подзаголовка с описанием лабораторной установки.
        :param image_filename: Название нужного изображения.
        :param caption: Описание изображения.
        """
        with self.doc.create(Section('Лабораторная установка')):
            self.add_figure(image_filename, caption)

    def add_custom_latex(self, latex_code):
        """
        Функция для добавления текста с формулами, написанными как в LaTeX.
        :param latex_code: LaTeX-код, содержащий текст, формулы и т.д. Обязательно перед "" поставьте r
        """
        self.doc.append(NoEscape(latex_code))
        self.doc.append(NoEscape(r'\newline'))

    def add_figure(self, image_filename, caption):
        """
        Функция для добавления изображения в центр.
        :param image_filename: Название файла нужного изображения.
        :param caption: Описание изображения.
        """
        with self.doc.create(Figure()) as fig:
            fig.add_image(image_filename, width='350px')
            fig.add_caption(caption)

    def add_graph(self, image_filename, caption):
        """
        Функция для добавления графика в центр.
        :param image_filename: Название файла нужного графика.
        :param caption: Описание графика.
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_filename, width='400px')
            fig.add_caption(NoEscape(caption))

    def add_table(self, title, headers, data):
        """
        Функция для добавления таблицы.
        :param title: Заголовок таблицы.
        :param headers: Заголовки столбцов.
        :param data: Данные таблицы.
        """
        with self.doc.create(Section(title)):
            with self.doc.create(Tabular('|c|c|c|c|')) as table:
                table.add_hline()
                table.add_row(
                    [
                        NoEscape(headers[0]),
                        NoEscape(headers[1]),
                        NoEscape(headers[2]),
                        NoEscape(headers[3]),
                    ]
                )
                table.add_hline()
                for row in data:
                    table.add_row(row)
                    table.add_hline()

    def add_section(self, title, content):
        """
        Функция для добавления раздела с заголовком и содержимым.
        :param title: Заголовок раздела.
        :param content: Содержимое раздела.
        """
        with self.doc.create(Section(title)):
            self.doc.append(content)

    def build(self):
        """
        Функция для завершения изменений в документе.
        """
        self.doc.generate_pdf(self.filename, clean_tex=False)

    def get_doc(self):
        """
        Функция для получения pdf и tex файлов нашего документа
        """
        return self.doc

    @staticmethod
    def represent_as_power_of_10(number):
        if number == 0:
            return NoEscape("$0$")

        exponent = int(math.log10(abs(number)))
        coefficient = number / (10 ** exponent)
        coefficient_rounded = round(coefficient, 3)

        return NoEscape(f"${coefficient_rounded} \\cdot 10^{{{exponent}}}$")

    @staticmethod
    def calculate_Bc(IL, params):
        """Вычисление магнитной индукции."""
        mu0 = 4 * np.pi * 10**(-7)
        N, l, d = params['N'], params['l'], params['d']
        return mu0 * N * IL / np.sqrt(l ** 2 + d ** 2)

    @staticmethod
    def calculate_em(U, Bc, ra):
        """Вычисление удельного заряда электрона."""
        return 8 * U / (Bc ** 2 * ra ** 2)

    @staticmethod
    def line_equation(x1, y1, x2, y2):
        """Функция для нахождения уравнения прямой по двум точкам."""
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    @staticmethod
    def find_intersection(m1, b1, m2, b2):
        """Функция для нахождения точки пересечения двух прямых."""
        if m1 == m2:
            raise ValueError("Прямые параллельны и не пересекаются")
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y

    def plot_graph(self, file_name="linear_approximation.png"):
        """
        Построение графика зависимости B^2_c от U и сохранение его в файл.
        :param file_name: Имя файла для сохранения графика
        """
        coefficients = np.polyfit(self.sin_ratios,self.magnetic_fields, 1)  # [a, b] для y = ax + b
        poly_eq = np.poly1d(coefficients)
        y_approx = poly_eq(self.sin_ratios)

        # Строим график
        plt.figure()
        plt.plot(self.sin_ratios, self.magnetic_fields, 'bo', label='Исходные данные')  # Исходные данные
        plt.plot(self.sin_ratios, y_approx, 'r-', label='Аппроксимация по прямой')  # Аппроксимация
        plt.xlabel(r'$\gamma_i = \frac{\sin(\alpha_i)}{\sin(\varphi - \alpha_i)}$', fontsize=10)
        plt.ylabel(r'$B_c, \text{мкТл}$', fontsize=12)
        plt.title(r'График зависимости $B_c = B_c(\gamma_i)$', fontsize=12)
        plt.legend()
        plt.grid(True)

        # Сохраняем график
        graph_filename = file_name
        plt.savefig(graph_filename)
        plt.close()

        # Возвращаем коэффициенты и имя файла с графиком
        return coefficients, graph_filename

    def make_me_p_2_5(self, data, params):
        """Основной код для обработки данных и создания графиков."""
        for U, values in data.items():
            IL, Ia = values['IL'], values['Ia']

            # Найти критический ток ILc
            diff = np.diff(Ia)
            diff = list(map(abs, diff))
            idx = np.argmax(diff)

            # Вычисления
            Bc = self.calculate_Bc(IL[idx], params) * 10 ** (3)
            e_m = self.calculate_em(U, Bc * 10 ** (-3), params['ra'])

            # Сохранение данных для графиков
            self.Bc_values.append(Bc)
            self.e_m_values.append(e_m)
            self.U_values.append(U)

            # Пример значений погрешностей (замените на реальные значения)
            delta_U = 0.1  # В
            delta_B_c = 0.001  # Тл
            delta_r_a = 0.0001  # м

            self.delta_U_values.append(delta_U)
            self.delta_B_c_values.append(delta_B_c)
            self.delta_r_a_values.append(delta_r_a)

            # Создание сетки графиков
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # График Ia от IL с выделением участков a и b
            axs[0].plot(IL, Ia, 'o-', label="Экспериментальные данные")

            # Нарисовать прямую "Критический ток"
            m_critical, b_critical = self.line_equation(IL[idx - 3], Ia[idx - 3], IL[idx - 2], Ia[idx - 3])
            x_critical = np.linspace(IL[idx - 3], IL[-1], 100)
            y_critical = m_critical * x_critical + b_critical
            axs[0].plot(x_critical, y_critical, 'g--')

            # Нарисовать прямую "Касательный ток"
            m_tangent, b_tangent = self.line_equation(IL[idx], Ia[idx], IL[idx + 1], Ia[idx + 1])
            x_tangent = np.linspace(IL[idx - 1], IL[-1], 100)
            y_tangent = m_tangent * x_tangent + b_tangent
            axs[0].plot(x_tangent, y_tangent, 'b--')
            x_intersect = 0
            # Найти точку пересечения
            try:
                x_intersect, y_intersect = self.find_intersection(m_critical, b_critical, m_tangent, b_tangent)
                axs[0].plot(x_intersect, y_intersect, 'ro', label=f"Критическая точка = {round(x_intersect, 3)}")
            except ValueError as e:
                print(f"Ошибка: {e}")

            self.critical_points.append([U, round(x_intersect, 3), round(Bc, 3), self.represent_as_power_of_10(e_m)])
            #plt.title(r'График зависимости $B_c = B_c(\gamma_i)$', fontsize=12)
            axs[0].set_xlim([-0.01, IL[-1] + 0.01])
            axs[0].set_ylim([-0.01, np.max(Ia) + 0.03])
            axs[0].set_xlabel(r"Ток в соленоиде, A")
            axs[0].set_ylabel(r"Анодный ток, A")
            axs[0].set_title(r"График $I_a$ от $I_L$ при U = " + f"{U} В")
            axs[0].grid(True)
            axs[0].legend()

            # График Ia/IL от IL
            Ia_over_IL = np.array(Ia) / np.array(IL)
            axs[1].plot(IL[1:], Ia_over_IL[1:], 'o-')
            axs[1].set_xlabel(r"Ток в соленоиде, A")
            axs[1].set_ylabel(r"$I_a / I_L$, A")
            axs[1].set_title(r"График $I_a / I_L$ от $I_L$ при U = " + f"{U} В")
            axs[1].grid(True)

            # Сохранение и закрытие фигуры
            plt.savefig(f'combined_plot_U_{U}.png')
            plt.close()
            self.add_graph(f'combined_plot_U_{U}.png',
                           r"Комбинированный график зависимости анодного тока и $I_a/I_L$ от тока в соленоиде при $U$ = "+f"{U} В")

        self.doc.append(NoEscape(r'Получившиеся графики в полной мере наглядно '
                                 r'показывают когда траектория электронов '
                                 r'становится касательной к поверхности анода.'))
        # Таблица результатов
        self.add_table(r"Результаты эксперимента",
                       [r"$U$, В", r"$I_{L_c}$, мА", r"$B_c$, мТл", r"$e/m$, Кл/кг"],
                       self.critical_points)

        # Среднее значение и погрешность
        mean_em = np.mean(self.e_m_values)

        self.doc.append(NoEscape(r' \vspace{0.5cm} Среднее значение $\langle \frac{e}{m} \rangle$: \text{ ' + self.represent_as_power_of_10(mean_em) + r' Кл/кг}' ))

        # График Bc^2 от U
        plt.figure()
        plt.plot(self.U_values, [(x * 10 ** (-3)) ** 2 for x in self.Bc_values], 'o-', label='Данные')
        plt.xlabel("Анодное напряжение, В")
        plt.ylabel(r"$B_c^2$, $\text{Тл}^2$")
        plt.title(r"График зависимости $B_c^2$ от $U$")
        plt.grid(True)

        # Линейная аппроксимация
        U_values = np.array(self.U_values)
        Bc2_values = np.array([(x*10**(-3)) ** 2 for x in self.Bc_values])
        coefficients = np.polyfit(U_values, Bc2_values, 1)
        slope = coefficients[0]  # Угловой коэффициент
        intercept = coefficients[1]  # Свободный член

        # Создание аппроксимирующей прямой
        fit_line = slope * U_values + intercept

        # Добавление аппроксимирующей прямой на график
        plt.plot(U_values, fit_line, label=f'Аппроксимация данных', linestyle='--')
        plt.legend()

        # Сохранение графика
        plt.savefig('Bc2_vs_U.png')
        plt.close()


        # Добавление графика в вашу систему (предполагается, что этот метод уже определен)
        self.add_graph('Bc2_vs_U.png', r"График зависимости $B_c^2$ от анодного напряжения $U$")
        self.doc.append(NoEscape(
            r' Угловой коэффициент аппроксимирующей прямой $k$ = ' + self.represent_as_power_of_10(slope) + "."))
        self.doc.append(NoEscape(r' \\ \vspace{1cm} Следовательно, $\frac{e}{m}$ = '
                                 r'$\frac{8 k}{r_a^2}$ =  ' +
                                 self.represent_as_power_of_10(8/(slope*params['ra']**2)) + r' Кл/кг.'))
        # Вызов calculus_error_report после завершения цикла
        self.calculus_error_report()

        self.doc.append(NoEscape(r'Получившиеся относительные погрешности для каждого из конкретного значения напряжения оказались'
                                 r' приблизительно равны 6 процентам, что является допустимым.'))

    def calculus_error_report(self):
        with self.doc.create(Section('Оценка погрешностей')):
            self.doc.append(NoEscape(r'Оценим погрешность удельного заряда электрона: '))
            self.doc.append(NoEscape(r' Вычислим для каждого из значений анодного напряжения и критического тока:'))
            self.doc.append(NoEscape(
                r'$$\Delta \left( \frac{e}{m} \right) = \sqrt{ \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial U} \Delta U \right)^2 + \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial B_c} \Delta B_c \right)^2 + \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial r_a} \Delta r_a \right)^2 }$$'))
            self.doc.append(
                NoEscape(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial U} = \frac{8}{B_c^2 r_a^2}$$'))
            self.doc.append(
                NoEscape(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial B_c} = -\frac{16U}{B_c^3 r_a^2}$$'))
            self.doc.append(
                NoEscape(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial r_a} = -\frac{16U}{B_c^2 r_a^3}$$'))

            mean_em = np.mean(self.e_m_values)

            for U, delta_U, B_c, delta_B_c, r_a, delta_r_a in zip(self.U_values, self.delta_U_values, self.Bc_values,
                                                                  self.delta_B_c_values,
                                                                  [params['ra']] * len(self.U_values),
                                                                  self.delta_r_a_values):
                # Преобразуем значения в систему СИ
                U_SI = U  # U уже в вольтах (В)
                delta_U_SI = delta_U  # delta_U уже в вольтах (В)
                B_c_SI = B_c * 1e-3  # B_c в миллитеслах (мТл), преобразуем в тесла (Тл)
                delta_B_c_SI = delta_B_c * 1e-3  # delta_B_c в миллитеслах (мТл), преобразуем в тесла (Тл)
                r_a_SI = r_a  # r_a уже в метрах (м)
                delta_r_a_SI = delta_r_a  # delta_r_a уже в метрах (м)

                delta_em = self.calculate_delta_em(U_SI, delta_U_SI, B_c_SI, delta_B_c_SI, r_a_SI, delta_r_a_SI)
                relative_error = (delta_em / mean_em) * 100

                self.doc.append(NoEscape(f'Оценим погрешность для U = {U} В: '))
                self.doc.append(NoEscape(
                    r'$$\Delta \left( \frac{e}{m} \right) = \sqrt{ \left( \frac{8 \Delta U}{B_c^2 r_a^2} \right)^2 '
                    r'+ \left( \frac{16U \Delta B_c}{B_c^3 r_a^2} \right)^2 + '
                    r'\left( \frac{16U \Delta r_a}{B_c^2 r_a^3} \right)^2 } = \text{' +
                    self.represent_as_power_of_10(delta_em)
                    + r'} \text{Кл/кг}$$'
                ))
                self.doc.append(NoEscape(f'Относительная погрешность: {relative_error:.2f}'+ r'\%'))
                self.doc.append(NoEscape(r'\newline'))  # Добавляем новую строку


    @staticmethod
    def calculate_delta_em(U, delta_U, B_c, delta_B_c, r_a, delta_r_a):
        """
        Вычисляет значение Δ(e/m) в системе СИ.

        :param U: Значение U в вольтах (В).
        :param delta_U: Значение ΔU в вольтах (В).
        :param B_c: Значение B_c в тесла (Тл).
        :param delta_B_c: Значение ΔB_c в тесла (Тл).
        :param r_a: Значение r_a в метрах (м).
        :param delta_r_a: Значение Δr_a в метрах (м).
        :return: Значение Δ(e/m) в Кл/кг.
        """
        term1 = (8 * delta_U) / (B_c ** 2 * r_a ** 2)
        term2 = (16 * U * delta_B_c) / (B_c ** 3 * r_a ** 2)
        term3 = (16 * U * delta_r_a) / (B_c ** 2 * r_a ** 3)

        delta_em = math.sqrt(term1 ** 2 + term2 ** 2 + term3 ** 2)

        return delta_em

    @staticmethod
    def calculate_delta_em(U, delta_U, B_c, delta_B_c, r_a, delta_r_a):
        """
        Вычисляет значение Δ(e/m) в системе СИ.

        :param U: Значение U в вольтах (В).
        :param delta_U: Значение ΔU в вольтах (В).
        :param B_c: Значение B_c в тесла (Тл).
        :param delta_B_c: Значение ΔB_c в тесла (Тл).
        :param r_a: Значение r_a в метрах (м).
        :param delta_r_a: Значение Δr_a в метрах (м).
        :return: Значение Δ(e/m) в Кл/кг.
        """
        term1 = (8 * delta_U) / (B_c ** 2 * r_a ** 2)
        term2 = (16 * U * delta_B_c) / (B_c ** 3 * r_a ** 2)
        term3 = (16 * U * delta_r_a) / (B_c ** 2 * r_a ** 3)

        delta_em = math.sqrt(term1 ** 2 + term2 ** 2 + term3 ** 2)

        return delta_em
if __name__ == "__main__":
    report = LabReport("lab_report", 29)

    # Титульный лист
    report.create_title_page("Лабораторная работа № 3.03: Определение удельного заряда электрона", "Исхаков Камиль Фархатович")
    report.create_data()
    # Текст части 1
    report.add_text_with_math("Магнитная индукцию внутри соленоида:",
                              r" B_c = \mu_0 I_c N \frac{1}{\sqrt{(l^2+d^2)}}")
    report.add_custom_latex(r"где $\mu_0 =  4 \pi 10^{-7} $ Гн/м "
                            r"– магнитная постоянная, "
                            r"$N$ – число витков соленоида, $l$ – его длина, "
                            r"$d$ – его диаметр.")
    report.add_text_with_math("Формула удельного заряда:",
                              r"\frac{e}{m} = \frac{8 U}{B_c^2 r_a^2}")
    report.add_custom_latex(r"где $e$ – заряд электрона, $m$ – его масса, "
                            r"$r_a$ – радиус анода, $U$ – анодное напряжение")

    # Текст части 2
    # report.lab_construction("", "Параметры установки:R=0,15м–"
    #                             "радиус катушек;n=100– число витков в каждой из катушек")

    # Текст части 3
    report.create_calculus()
    params = {'N': 1500, 'l': 0.036, 'd': 0.037, 'ra': 0.003}
    data = {
        12: {
            'IL': [0.00000001, 0.039, 0.063, 0.094, 0.124, 0.150, 0.177, 0.218, 0.250, 0.280,
                   0.313, 0.339, 0.370, 0.404, 0.431, 0.470, 0.505, 0.536, 0.570, 0.6],
            'Ia': [0.279, 0.278, 0.277, 0.278, 0.279, 0.280, 0.278, 0.263, 0.207, 0.159, 0.122,
                   0.103, 0.082, 0.067, 0.059, 0.049, 0.043, 0.039, 0.036, 0.034]
        },
        14: {
            'IL': [0.00000001, 0.030, 0.049, 0.086, 0.109, 0.144, 0.172, 0.200, 0.225, 0.251, 0.275, 0.303,
                   0.330, 0.359, 0.393, 0.425, 0.448, 0.477, 0.500, 0.523],
            'Ia': [0.353, 0.352, 0.351, 0.351, 0.352, 0.353, 0.352, 0.348, 0.336, 0.272, 0.220,
                   0.181, 0.153, 0.128, 0.101, 0.087, 0.078, 0.069, 0.064, 0.059]
        },
        15: {
            'IL': [0.00000001, 0.033, 0.057, 0.082, 0.106, 0.136, 0.156, 0.185, 0.208, 0.236,
                   0.267, 0.282, 0.303, 0.330, 0.358, 0.385, 0.412, 0.441, 0.469, 0.5],
            'Ia': [0.389, 0.388, 0.389, 0.388, 0.389, 0.392, 0.392, 0.389, 0.383, 0.365, 0.266,
                   0.239, 0.207, 0.174, 0.149, 0.123, 0.106, 0.093, 0.082, 0.074]
        }
    }

    # Создание таблиц и графиков для каждого значения анодного напряжения
    report.make_me_p_2_5(data, params)

    # Аппроксимация и добавление графика в отчет

    # Итоги
    report.create_summary()
    report.add_text_with_math(r"Табличное значение удельного заряда электрона:",r"\frac{e}{m} = 1,76\cdot10^{11} \textit{ Кл/кг}")
    report.add_custom_latex(NoEscape(r"В ходе работы был определен удельный заряд электрона. Табличное и "
                              r"экспериментальное значение удельного заряда электрона в нашем случае"
                              r" получилось почти идентичными. Во время выполнения всей лабораторной работы значения получались "
                              r"вполне реалистичными, кроме задания с построением зависимости $B^2_c$ от $U$, поскольку значение удельного заряда, вычисленный через коэффициент наклона прямой оказалось в полтора раза больше табличного значения. "
                              r"Расхождение теоретического значения удельного заряда от экспериментального может быть из-за влияния облака заряда, накапливающегося в диоде. "))
    # Генерация PDF
    report.build()