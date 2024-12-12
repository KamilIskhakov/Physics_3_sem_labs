import math

import numpy as np
import matplotlib.pyplot as plt
from pylatex import Section,NoEscape, Figure
from pylatex.package import Package
from pylatex import Document
from math import log10, floor


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
        with self.doc.create(Figure(position="h!")) as fig:
            fig.add_image(image_filename, width='350px')

    def add_figure_witn_page(self, image_filename):
        """
        Функция для добавления изображения в центр и созданием новой страницы.
        :param image_filename: Название файла нужного изображения.
        """
        with self.doc.create(Figure(position="h!")) as fig:
            fig.add_image(image_filename, width='350px')
            self.doc.append(NoEscape(r'\newpage'))

    def calc_electric_field_error(self, phi_1, phi_2, d):
        """
        Функция для вычисления погрешности электрической напряженности.
        """
        delta_E = math.sqrt(
            ((2*0.1) / (3*d) )** 2 + (( 2*(phi_2 - phi_1) * 0.001) / ( (3*d ** 2))) ** 2)

        return round(delta_E,3)

    def calc_sigma(self, E):
        """
        Функция для вычисления поверхностной плотности электрического заряда на электроде.
        """
        sigma = -8.85*E*(10)**(-2)

        return round(sigma,3)
    def plot_potential_graphs(self,mas_1, mas_2,file_name="2_potentials.png"):
        """
        Рисует два графика зависимости потенциала от координаты
        для двух исследованных конфигураций поля.
          :param mas_1: Список координат и потенциалов для плоского конденсатора.
          :param mas_2: Список координат и потенциалов для кольца.
          :param file_name: Название графика
        """
        x1, y1 = zip(*mas_1)
        x2, y2 = zip(*mas_2)

        plt.plot(x1, y1, label="Плоский конденсатор",)
        plt.plot(x2, y2, label="Кольцо")

        plt.xlabel("Координата X (см)")
        plt.ylabel("Потенциал Phi (В)")
        plt.title("Зависимость потенциала от координаты для двух конфигураций поля")
        plt.legend()
        plt.grid(True)

        # Сохраняем график
        graph_filename = file_name
        plt.savefig(graph_filename)
        plt.close()

        return graph_filename

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

class MeasurementRounding:
    def __init__(self, value, error):
        """
        Инициализация класса с измеренным значением и абсолютной погрешностью.
        :param value - измеренное значение
        :param error - абсолютная погрешность
        """
        self.value = value
        self.error = error

    @staticmethod
    def significant_digits(x):
        """
        Определяет количество значащих цифр в числе.
        """
        if x == 0:
            return 0
        return floor(log10(abs(x))) + 1

    @staticmethod
    def round_to_significant_figures(x, sig_figs):
        """
        Округляет число до указанного количества значащих цифр.
        """
        if x == 0:
            return 0
        # Определяем масштаб
        scale = floor(log10(abs(x)))
        factor = 10 ** (sig_figs - scale - 1)
        return round(x * factor) / factor

    def round_error(self):
        """
        Округляет погрешность в зависимости от первой значащей цифры.
        """
        first_digit = int(str(self.error)[0])  # Первая значащая цифра
        if first_digit in [1, 2, 3]:
            return self.round_to_significant_figures(self.error, 2)
        else:
            return self.round_to_significant_figures(self.error, 1)

    def round_value(self, error_rounded):
        """
        Округляет измеренное значение до того же разряда, что и погрешность.
        """
        return round(self.value, -floor(log10(error_rounded)))


    def get_rounded_measurement(self):
        """
        Возвращает округленное измеренное значение и погрешность.
         """
        error_rounded = self.round_error()
        value_rounded = self.round_value(error_rounded)
        return value_rounded, error_rounded


if __name__ == "__main__":
    report = LabReport("lab_report",29)

    # Титульный лист
    report.create_title_page("Лабораторная работа № 3.01: Изучение электростатического поля методом моделирования", "Исхаков Камиль Фархатович")
    report.create_data()
    # Текст части 1
    report.add_text_with_math("Средняя напряженность между двумя точками, лежащими на одной"
                              "силовой линии:",
                              r"\langle E_{12} \rangle= \frac{\phi_1-\phi_2}{l_{12}}")
    report.add_custom_latex(r"где $\phi_1, \phi_2$ – потенциалы в выбранных точках, а $l_{12}$ – расстояния между "
                            r"данными точками")
    report.add_text_with_math("Поверхностная плотность зарядов проводника:",
                              r"\sigma' = -\epsilon_0 \frac{\Delta \phi}{\Delta l_n}")
    report.add_custom_latex(r"где $\epsilon_0$ – постоянная электрическая постоянная, $\Delta \phi$"
                            r"– изменение потенциала при смещении на малое расстояние $\Delta l_n$"
                            r"по нормали к поверхности проводника")

    # Текст части 2
    # report.lab_construction("", "Параметры установки:R=0,15м–"
    #                             "радиус катушек;n=100– число витков в каждой из катушек")

    # Текст части 3
    report.create_calculus()
    E_center = round(2/(0.16-0.100),3)
    report.add_custom_latex(r"$E_{\text{центра(16 10)}} = "
                            r"\frac{\phi_1-\phi_2}{l_{12}} = "
                            r"\frac{2}{0.162-0.100} = $" + str(E_center))
    E_loc  = round((12.01-11.25)/(0.286-0.265), 3)

    report.add_custom_latex(r"$E_{\text{окр+}} = "
                            r"\frac{\phi_1-\phi_2}{l_{12}} = "
                            r"\frac{12.01-11.25}{0.286-0.265} = $" + str(E_loc))

    E_center_error = MeasurementRounding(E_center,
                                         report.calc_electric_field_error
                                         (4,2,0.160-0.100)).get_rounded_measurement()[0]
    E_loc_error = MeasurementRounding(E_loc,
                                         report.calc_electric_field_error
                                         (12.01, 11.25, 0.286-0.265)).get_rounded_measurement()[0]
    report.add_text_with_math("Расчет погрешностей измерений:",r""
                                                               r"\Delta E = "
                                                               r"\sqrt{\left(\frac{2\cdot \Delta \phi_i}{3 l}\right)^2+"
                                                               r"\left(\frac{2(\phi_2-\phi_1)\cdot \Delta l_i}{3 l^2}\right)^2}")
    report.add_custom_latex(r"$\Delta E_{\text{центра(16 10)}}$ = " + str(E_center_error)+r" В")
    report.add_custom_latex(r"$\Delta E_{\text{окр+}}$ = " + str(E_loc_error) + r" В")
    report.add_custom_latex(r"$\sigma'_{+} = - \epsilon_0 \frac{\Delta \phi}{\Delta l_n} $= "
                            r"" + str(report.calc_sigma(E_loc)) + " В/м")
    E_minus = round((2.5-1.98)/(0.03-0.0195), 3)

    report.add_custom_latex(r"$\sigma'_{-} = - \epsilon_0 \frac{\Delta \phi}{\Delta l_n} $= "
                            r"" + str(report.calc_sigma(E_minus)) + " В/м")
    arrow = [(2.4, 2), (5.9, 4), (9.9, 6),(10.5,  6.9),(15, 6.9), (16.5, 8), (22.4, 10), (27.1, 12)]
    ring = [(2.1, 2), (5.7, 4), (9.2, 6),(10.0, 7.5),(20, 7.5), (22, 8), (24.2, 10), (28.1, 12)]

    report.add_figure_witn_page(report.plot_potential_graphs(arrow,ring))
    report.add_figure("potential_field.jpg")
    report.add_custom_latex(r"Зеленые линии – эквипотенциальные линии; "
                            r"Синие линии – силовые линии.")
    report.create_summary()
    report.add_custom_latex(r"В результате выполнения лабораторной работы "
                            r"было смоделировано электрическое поле с помощью "
                            r"эквипотенциальных поверхностей. Стоит отметить, что верхняя "
                            r"часть смоделированного поля в последнем рисунке искривлена сильнее, чем "
                            r"нижняя. Это может быть связано с тем, что источник питания был расположен как"
                            r" раз ближе к верхней половине установки. Также искривления могли появиться из-за того,"
                            r" что недистиллированная вода могла неравномерно покрывать установку, вследствие чего"
                            r" одна из сторон источников питания была менее погружена другой."
                            r" \\ На графике зависимости потенциала $\phi$ "
                            r"от икса для кольца и плоского конденсатора видно плоское плато, которое соответствует"
                            r" значению напряженности конденсатора и кольца (и области, заключенной внутри кольца)"
                            r". Причем для конденсатора характерно, что в окрестности кончика стрелки значение потенциала"
                            r" растет более быстро, чем в хвостике. Это соответствует действительности, поскольку "
                            r"количество зарядов на кончике будет больше, чем в самом конце. Расчитанные погрешности не имеют особых всплесков."
                            r"")

    # Генерация PDF
    report.build()
