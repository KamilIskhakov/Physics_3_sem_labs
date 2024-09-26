from statistics import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import scale
from pylatex import Document, Section, Subsection, Command, Tabular, NoEscape, Figure, Table, MultiColumn, Math
from pylatex.utils import italic, bold
from pylatex.package import Package
from pylatex import Document
from io import BytesIO
import scipy.stats as stats



class LabReport:
    def __init__(self, filename, font_size= 26):
        self.doc = Document(geometry_options={"left": "3cm", "right": "3cm", "bottom": "2cm", "top":"1cm"})
        self.filename = filename
        self.font_size = font_size
        # Добавляем пакеты для поддержки кириллицы и математики
        self.doc.packages.append(Package('babel', options='russian'))
        self.doc.packages.append(Package('amsmath'))
        self.doc.packages.append(Package('amssymb'))
        self.doc.packages.append(Package('amsfonts'))
        self.doc.packages.append(Package('geometry'))
        self.doc.packages.append(Package('mathtext'))


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
            self.doc.append(NoEscape(text + r"\begin{displaymath}" +  math_expr + r"\end{displaymath}"))
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

    def add_figure(self, image_filename):
        """
        Функция для добавления изображения в центр.
        :param image_filename: Название файла нужного изображения.
        """
        with self.doc.create(Figure()) as fig:
            fig.add_image(image_filename, width='350px')

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


class MeasurementTable:
    def __init__(self, doc, angles, currents_1, currents_2, phi):
        """
        Инициализация данных для расчетов и создания таблицы.
        :param doc: Данный документ
        :param angles: Массив углов α_i в градусах
        :param currents_1: Массив значений тока I_1
        :param currents_2: Массив значений тока I_2
        :param phi: Угол φ в градусах
        """
        self.doc = doc
        self.angles = angles
        self.currents_1 = currents_1
        self.currents_2 = currents_2
        self.phi = phi
        self.average_current = self.give_average_current() # Среднее значение тока <I>
        self.sin_ratios = self.calculate_sin_ratios()  # Вычисление sin(α_i)/sin(φ - α_i)
        self.magnetic_fields = self.calculate_magnetic_fields()  # Расчет Bc

    def give_average_current(self):
        """
        Вычисление среднего значения силы тока при одном и том же угле α_i.
        """
        mas = []
        for i in range(len(self.currents_1)):
            mas.append((currents_1[i]+currents_2[i])/2)
        return mas

    def calculate_sin_ratios(self):
        """
        Вычисление отношения sin(α_i)/sin(φ - α_i) для каждого угла α_i.
        """
        mas = []
        for i in range(len(self.currents_1)):
            mas.append(np.sin(np.radians(self.angles[i])) / np.sin(np.radians(self.phi - self.angles[i])))
        return mas

    def calculate_magnetic_fields(self):
        """
        Расчет магнитного поля B_c на основе среднего тока и синусных отношений для каждого угла α_i.
        """
        mas = []
        for i in range(len(self.currents_1)):
            mas.append(1.25663706*(1/10**6)*((4/5)**(3/2)) * ((self.average_current[i] * 100)/0.15)*10**3) # тут же сила тока была в мА, поэтому в конце вместо 10^6 получили 10^3
        return mas

    def create_latex_table(self):
        """
        Создание таблицы в формате LaTeX.
        """

        with self.doc.create(Table(position='h!')) as table:
            with self.doc.create(Tabular("|c|c|c|c|c|c|")) as tabular:
                table.add_caption("Результаты прямых измерений")
                tabular.add_hline()
                tabular.add_row(
                    [
                        NoEscape(r"$\varphi = \ldots^\circ$"),
                        MultiColumn(3, align="|c|", data="Ток в катушках, мА"),
                        "", ""
                    ]
                )
                tabular.add_hline()
                tabular.add_row(
                    [
                        NoEscape(r"$\alpha_i$"),
                        NoEscape("$I_1$"),
                        NoEscape("$I_2$"),
                        NoEscape(r"$\langle I \rangle$"),
                        NoEscape(r"$\frac{\sin (\alpha_i)}{\sin (\varphi - \alpha_i)}$"),
                        NoEscape(r"$B_c, \text{мкТл}$"),
                    ]
                )
                tabular.add_hline()
                # Заполнение строк таблицы
                for i in range(len(self.angles)):
                    tabular.add_row(
                        [
                            f"{self.angles[i]}°",
                            f"{self.currents_1[i]:.2f}",
                            f"{self.currents_2[i]:.2f}",
                            f"{self.average_current[i]:.2f}",  # Здесь используем i для индексации
                            f"{self.sin_ratios[i]:.2f}",
                            f"{self.magnetic_fields[i]:.2f}"
                        ]
                    )
                    tabular.add_hline()
        self.doc.append(NoEscape(r'\newpage'))

    def plot_graph(self, file_name="linear_approximation.png"):
        """
        Построение графика зависимости Bc от γ_i и сохранение его в файл.
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
    def give_errors(self):
        linear_regress = LinearRegressionWithErrors(self.sin_ratios, self.magnetic_fields)
        linear_regress.fit()
        linear_regress.display_results()
        return linear_regress.calculate_error()


class LinearRegressionWithErrors:
    def __init__(self, x, y, confidence_level=0.95):
        """
        Инициализация класса с экспериментальными данными и уровнем доверия.

        x: массив значений независимой переменной (например, угол γ)
        y: массив значений зависимой переменной (например, магнитное поле Bc)
        confidence_level: уровень доверия для расчета доверительных интервалов (по умолчанию 95%)
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.confidence_level = confidence_level
        self.b0 = None  # Свободный член
        self.b1 = None  # Угловой коэффициент
        self.s_b1 = None  # Стандартная ошибка углового коэффициента
        self.y_pred = None  # Предсказанные значения

    def fit(self):
        """
        Метод для вычисления коэффициентов линейной регрессии с использованием метода наименьших квадратов (МНК).
        """
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        self.b1 = np.sum((self.x - x_mean) * (self.y - y_mean)) / np.sum((self.x - x_mean) ** 2)
        self.b0 = y_mean - self.b1 * x_mean

        # Вычисление предсказанных значений
        self.y_pred = self.b0 + self.b1 * self.x

        # Оценка ошибок
        residuals = self.y - self.y_pred
        s_squared = np.sum(residuals ** 2) / (self.n - 2)
        self.s_b1 = np.sqrt(s_squared / np.sum((self.x - x_mean) ** 2))

    def calculate_error(self):
        """
        Метод для расчета погрешности углового коэффициента с использованием коэффициента Стьюдента.
        """
        degrees_of_freedom = self.n - 2
        t_value = stats.t.ppf(1 - (1 - self.confidence_level) / 2, degrees_of_freedom)
        # Погрешность с коэффициентом Стьюдента
        error_b1 = t_value * self.s_b1
        return error_b1, t_value

    def display_results(self):
        """
        Метод для вывода результатов на экран.
        """
        if self.b1 is None or self.s_b1 is None:
            print("Необходимо сначала выполнить метод fit() для вычисления регрессии.")
            return

        error_b1, t_value = self.calculate_error()

        print(f"Угловой коэффициент (B1): {self.b1:.4f}")
        print(f"Свободный член (B0): {self.b0:.4f}")
        print(f"Погрешность углового коэффициента: ±{error_b1:.4f}")
        print(f"Коэффициент Стьюдента для доверительного уровня {self.confidence_level * 100}%: {t_value:.4f}")

    def plot(self):
        """
        Метод для построения графика зависимости и линии регрессии.
        """
        if self.y_pred is None:
            print("Необходимо сначала выполнить метод fit() для вычисления регрессии.")
            return

        plt.scatter(self.x, self.y, label="Экспериментальные данные")
        plt.plot(self.x, self.y_pred, color='red', label=f"МНК: y={self.b0:.2f} + {self.b1:.2f}x")
        plt.xlabel("Угол γ")
        plt.ylabel("Магнитное поле Bc")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    report = LabReport("lab_report",29)

    # Титульный лист
    report.create_title_page("Лабораторная работа № 3.13: Магнитное поле Земли", "Исхаков Камиль Фархатович")
    report.create_data()
    # Текст части 1
    report.add_text_with_math("Отношение между направлениями пробного поля и земного"
                              " магнитного поля:",
                              r"{\frac{\sin{\alpha}}{\sin{\phi-\alpha}}} "
                              r"= {\frac{B_h}{B_c}}")
    report.add_custom_latex(r"где $B_h, B_c$ – направления пробного поля и земного "
                            r"магнитного поля соответственно; $\phi$ – угол между $B_c$ "
                            r"и $B_h$, а $\alpha$ – угол между направлением "
                            r"результирующего поля и земного магнитного поля.")
    report.add_text_with_math("Величина магнитной индукции на оси одного "
                              "кругового тока:",
                              r"B(x) = \frac{\mu_0 I}{2} \frac{R^2}{(x^2+R^2)^{3/2}}")
    report.add_custom_latex(r"где $I$ – сила тока, $R$ – средний радиус каждой катушки,$\mu_0$ – магнитная постоянная, $x$ – расстояние от центра контура")
    report.add_text_with_math("Модуль вектора направления пробного поля:",
                              r"B_c = \mu_0 \left(\frac{4}{5}\right)^{3/2} \frac{I n}{R}")
    report.add_custom_latex(r"где $n$ – число витков в каждой катушке")

    # Текст части 2
    # report.lab_construction("", "Параметры установки:R=0,15м–"
    #                             "радиус катушек;n=100– число витков в каждой из катушек")

    # Текст части 3
    report.create_calculus()
    angles = np.arange(10, 150, 10)  # значения углов α_i
    currents_1 = np.array([6.4,12.5,15.0,17.7,19.4,21.1,21.9, 23.2,25.2,26.3,28.9,30.8,33.7,40.9]) # значения тока I_1
    currents_2 = np.array([7.7,12.6,16.0,18.2,19.4,20.4,22.3,23.6,25.1,26.3,28.2,30.3,33.8,39.5]) # значения тока I_2
    phi = 161  # значение угла φ
    table = MeasurementTable(report.get_doc(),angles,currents_1,currents_2, phi)
    table.create_latex_table()
    # Аппроксимация и добавление графика в отчет
    coefficients, graph_filename = table.plot_graph()
    coefficients[0],coefficients[1] = round(coefficients[0],2),round(coefficients[1],2)
    report.add_figure(graph_filename)
    errors = table.give_errors()
    report.add_custom_latex(
        r"Угловой коэффициент "+f" a = {coefficients[0]:.4f}"+" мкТл"+r", что должно "
                                                         r"соответствовать величине"
                                                         r" магнитного поля Земли. "
                                                         r"Из методического пособия "
                                                         r"А.Ф. Костко В.А. Самолетов "
                                                         r"'ФИЗИКА ЛАБОРАТОРНЫЕ РАБОТЫ ПО ЭЛЕКТРИЧЕСТВУ И МАГНЕТИЗМУ'"
                                                         r", можно получить информацию о том, какова "
                                                         r"величина магнитного поля Земли в Санкт-Петербурге, "
                                                         r"которая равна 15,4 мкТл."
                                                         r" Найдем погрешность углового коэффициента, а также"
                                                         r" доверительный интервал при $\alpha = 0.95$:\vspace{0.5cm}\\"
                                                         r"\textbf{Погрешность углового коэффициента}: "+ r" $\pm$"+f"{errors[0]:.4f}"+ r" мкТл\vspace{0.5cm} \\"
                                                         r"\textbf{Доверительный интервал} для $95.0\%$: "+f"{coefficients[0]:.4f}"+ r" $\pm$"+f"{errors[1]:.4f}"+ r" мкТл\vspace{0.5cm}" )
    # Итоги
    report.create_summary()
    report.add_text_with_math(r"Во время выполнения лабораторной работы удалось установить диапазон величины магнитного поля Земли, в котором содержится истинное ее значение."
                              r"Также была сделана линейная аппроксимация данных, которая хорошо описывает зависимость."
                          f"Коэффициенты линейной аппроксимации: a = {coefficients[0]:.4f}, b = {coefficients[1]:.4f}")

    # Генерация PDF
    report.build()
