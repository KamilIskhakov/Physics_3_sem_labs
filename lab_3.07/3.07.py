import numpy as np
import matplotlib.pyplot as plt
from pylatex import Document, NoEscape, Figure, Tabular, Section, MultiColumn, Package, Table, Command
import math
from typing import List

from abc import ABC, abstractmethod

class IDocumentHandler(ABC):
    @abstractmethod
    def append_content(self, content):
        pass

    @abstractmethod
    def generate_pdf(self, filename):
        pass

class IDataProcessor(ABC):
    @abstractmethod
    def process_data(self, data, params):
        pass

    @abstractmethod
    def calculate_errors(self, U_values, delta_U_values, Bc_values, delta_Bc_values, ra, delta_ra):
        pass

class IPlotter(ABC):
    @abstractmethod
    def plot_graph(self, x_data, y_data, filename, xlabel, ylabel, title, legend_labels=None):
        pass

class DocumentHandler(IDocumentHandler):
    def __init__(self, filename):
        self.doc = Document(geometry_options={"left": "3cm", "right": "3cm", "bottom": "2cm", "top": "1cm"})
        self.doc.packages.append(Package('babel', options='russian'))
        self.doc.packages.append(Package('amsmath'))
        self.doc.packages.append(Package('amssymb'))
        self.doc.packages.append(Package('amsfonts'))
        self.doc.packages.append(Package('geometry'))
        self.doc.packages.append(Package('mathtext'))
        self.filename = filename

    def append_content(self, content):
        self.doc.append(NoEscape(content))

    def add_text_with_math(self, content, math_expr=None):
        """
        Функция для указания названия формулы и ее самой в формате displaymath.
        :param content: Название формулы.
        :param math_expr: Нужная формула; Записываете ее как если бы записывали в латехе (причем без долларов), но перед "" обязательно поставьте r (r"").
        """
        if math_expr:
            self.doc.append(NoEscape(content + r"\begin{displaymath}" + math_expr + r"\end{displaymath}"))
        else:
            self.doc.append(content)
        self.doc.append(NoEscape(r'\newline'))

    def add_graph(self, image_filename, caption):
        """
        Функция для добавления графика в центр.
        :param image_filename: Название файла нужного графика.
        :param caption: Описание графика.
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_filename, width='300px')
            fig.add_caption(NoEscape(caption))


    def generate_pdf(self, filename):
        self.doc.generate_pdf(filename, clean_tex=False)

class DataProcessor(IDataProcessor):
    def __init__(self, params):
        self.params = params

    @staticmethod
    def represent_as_power_of_10(number):
        if number == 0:
            return NoEscape("$0$")

        exponent = int(math.log10(abs(number)))
        coefficient = number / (10 ** exponent)
        coefficient_rounded = round(coefficient, 3)

        return NoEscape(f"${coefficient_rounded} \\cdot 10^{{{exponent}}}$")

    def calculate_X(self,kx,ky,f):
        return kx*ky*((1665*470000*0.47*(10**(-6)))/(970*68))*f

    def calculate_P(self,Xi,S):
        return Xi*S
    def calculate_H_for_max_mu(self):
        index_mu = 0
        mu_buf = 0
        for i in range(len(self.values_mu)):
            if self.values_mu[i] > mu_buf:
                index_mu = i
                mu_buf = self.values_mu[i]

        return self.values_H[index_mu]
    def calculate_max_mu_th(self):
        mu = math.pi * 4 * 10 ** (-7)
        index_mu = 0
        mu_buf = 0
        for i in range(len(self.values_mu)):
            if self.values_mu[i] > mu_buf:
                index_mu = i
                mu_buf = self.values_mu[i]

        return self.values_B[index_mu]/(mu * self.values_H[index_mu])

    def process_data(self, data, num_key):
        alpha = 313.91
        beta = 3.558
        mu = math.pi * 4 * 10 ** (-7)
        for key, values in data.items():
            if key == num_key:
                if num_key == 1:
                    self.values_U_x_1 = values['U_x']
                    self.values_U_y_1 = values['U_y']
                    self.values_H_c = [x*alpha*10**(-3) for x in self.values_U_x_1]
                    self.values_B_r = [x * beta * 10 **(-3)  for x in self.values_U_y_1]

                elif num_key == 2:
                    self.values_U_x_2 = values['U_x']
                    self.values_U_y_2 = values['U_y']
                    self.values_H_m = [x * alpha * 10 ** (-3) for x in self.values_U_x_2]
                    self.values_B_m = [x * beta * 10 ** (-3)  for x in self.values_U_y_2]
                    self.values_mu_m = [y / x/ mu for x, y in zip(self.values_H_m, self.values_B_m)]

                elif num_key == 3:
                    self.values_U = values['U']
                    self.values_U_x_3 = values['U_x']
                    self.values_U_y_3 = values['U_y']
                    self.values_H = [x * alpha * 10 ** (-3) for x in self.values_U_x_3]
                    self.values_B = [x * beta * 10 ** (-3)  for x in self.values_U_y_3]
                    self.values_mu = [y / x/ mu for x, y in zip(self.values_H, self.values_B)]

    def calculate_errors(self, U_values, delta_U_values, Bc_values, delta_Bc_values, ra, delta_ra):
        mean_em = np.mean(self.e_m_values)
        for U, delta_U, B_c, delta_B_c, r_a, delta_r_a in zip(U_values, delta_U_values, Bc_values, delta_Bc_values, [ra]*len(U_values), delta_ra):
            U_SI = U
            delta_U_SI = delta_U
            B_c_SI = B_c * 1e-3
            delta_B_c_SI = delta_B_c * 1e-3
            r_a_SI = r_a
            delta_r_a_SI = delta_r_a
            delta_em = self.calculate_delta_em(U_SI, delta_U_SI, B_c_SI, delta_B_c_SI, r_a_SI, delta_r_a_SI)
            relative_error = (delta_em / mean_em) * 100
            yield (U, delta_em, relative_error)

    @staticmethod
    def calculate_delta_em(U, delta_U, B_c, delta_B_c, r_a, delta_r_a):
        term1 = (8 * delta_U) / (B_c**2 * r_a**2)
        term2 = (16 * U * delta_B_c) / (B_c**3 * r_a**2)
        term3 = (16 * U * delta_r_a) / (B_c**2 * r_a**3)
        return math.sqrt(term1**2 + term2**2 + term3**2)

class Plotter(IPlotter):
    def plot_graph(self, x_data, y_data, filename, xlabel, ylabel, title, legend_labels=None):
        plt.figure()
        plt.plot(x_data, y_data, 'o-', label=legend_labels[0] if legend_labels else None)
        if legend_labels:
            plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

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
        return math.floor(math.log10(abs(x))) + 1

    @staticmethod
    def round_to_significant_figures(x, sig_figs):
        """
        Округляет число до указанного количества значащих цифр.
        """
        if x == 0:
            return 0
        # Определяем масштаб
        scale = math.floor(math.log10(abs(x)))
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
        return round(self.value, -math.floor(math.log10(error_rounded)))


    def get_rounded_measurement(self):
        """
        Возвращает округленное измеренное значение и погрешность.
         """
        error_rounded = self.round_error()
        value_rounded = self.round_value(error_rounded)
        return value_rounded, error_rounded


class ErrorCalculator:
    def __init__(self, t_alpha_n: float, measurements: List[float]):
        """
        Инициализация класса для расчета погрешности.

        :param t_alpha_n: Коэффициент Стьюдента (t_alpha, n) для заданного уровня доверия и числа измерений.
        :param measurements: Список измерений (альфа_i).
        """
        self.t_alpha_n = t_alpha_n
        self.measurements = measurements

    def mean_value(self) -> float:
        """
        Вычисляет среднее значение измерений.

        :return: Среднее арифметическое измерений.
        """
        return sum(self.measurements) / len(self.measurements)

    def calculate_error(self) -> float:
        """
        Вычисляет погрешность по заданной формуле.

        :return: Значение погрешности (Δα).
        """
        n = len(self.measurements)
        mean_alpha = self.mean_value()
        variance_sum = sum((x - mean_alpha) ** 2 for x in self.measurements)
        error = self.t_alpha_n * math.sqrt(variance_sum / (n * (n - 1)))
        return error

# Мейник
if __name__ == "__main__":
    doc_handler = DocumentHandler("lab_report")
    data_processor = DataProcessor(params={'N': 1500, 'l': 0.036, 'd': 0.037, 'ra': 0.003})
    plotter = Plotter()

    doc_handler.append_content(r'\begin{center}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(
        r'{\LARGE\textbf{Лабораторная работа № 3.07: \\ Изучение свойств ферромагнетиков}}\\[1cm]')
    doc_handler.append_content(r'{\Large Исхаков Камиль Фархатович}\\[1cm]')
    doc_handler.append_content(r'{\Large \today}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(r'\end{center}')
    doc_handler.append_content(r'\newpage')

    with doc_handler.doc.create(Section('Основные формулы')):
        doc_handler.add_text_with_math(r"Магнитная проницаемость материала:",r"\mu=\frac{B}{\mu_0 H}")
        doc_handler.append_content(r"где $B$ - индукция магнитного поля в материале, $\mu_0 = 4 \pi \cdot 10^{-7}$ Гн/м - магнитная постоянная, $H$ -  напряженность магнитного поля.\\")
        doc_handler.add_text_with_math(r"Средняя мощность, расходуемая внешним источником тока при циклическом перемагничивании ферромагнитного образца:", r"P = \chi \cdot S_{\text{ПГ}}")
        doc_handler.append_content(r"где $S_{\text{ПГ}}$ - площадь петли гистерезиса, измеренная в делениях шкалы осциллографа, а коэффициент $\chi$ равен: $$\chi = K_x K_y \frac{N_1 R_2 C_1}{N_2 R_1}f$$где $f$ – частота сигнала, подаваемого на первичную обмотку трансформатора, "
                                   r"$K_x, K_y$ цена горизонтального и вертикального деления соответственно, $N_1, N_2$ – число витков первичной и вторичной обмотки соответственно, "
                                   r" $C_1$ – емкость конденсатора, $R_1, R_2$ – сопротивления первого и второго резистора соответственно.\\")

        doc_handler.add_graph('lab3_07_p1.png', r"Принципиальная электрическая схема установки")
    # Ваши данные
    data = {
        1: {'U_x': [108],'U_y': [78.2],'K_x': [100],'K_y': [50]}, # мВ
        2: {'U_x': [317],'U_y': [129]},
        3: {'U': [19,17,15,13,11,9,7,5],'U_x': [295,247,205,175,142,123,106,93.7],'U_y': [126,109,97.2,84.7,71.4,57.9,42.9,31.9]},} # мВ
    data_processor.process_data(data, 1)

    with doc_handler.doc.create(Section("Результаты эксперимента")):
        with doc_handler.doc.create(Table(position='h!')) as table:
            table.append(Command('centering'))
            tabular = Tabular('|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$U_x$, мВ"),
                NoEscape(r"$U_y$, мВ"),
                NoEscape(r"$H_c$, А/м"),
                NoEscape(r"$B_r$, Тл")
            ])
            tabular.add_hline()

            for i in range(len(data_processor.values_U_x_1)):
                tabular.add_row([
                    f"{data_processor.values_U_x_1[i]:.2f}",
                    f"{data_processor.values_U_y_1[i]:.2f}",
                    f"{data_processor.values_H_c[i]:.2f}",
                    f"{data_processor.values_B_r[i]:.2f}"
                ])
                tabular.add_hline()
            table.append(tabular)
            table.add_caption('Измерения 1')
        data_processor.process_data(data, 2)
        with doc_handler.doc.create(Table(position='h!')) as table:
            table.append(Command('centering'))
            tabular = Tabular('|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$U_x$, мВ"),
                NoEscape(r"$U_y$, мВ"),
                NoEscape(r"$H_c$, А/м"),
                NoEscape(r"$B_r$, Тл"),
                NoEscape(r"$\mu_m$")
            ])
            tabular.add_hline()

            for i in range(len(data_processor.values_U_x_2)):
                tabular.add_row([
                    f"{data_processor.values_U_x_2[i]:.2f}",
                    f"{data_processor.values_U_y_2[i]:.2f}",
                    f"{data_processor.values_H_m[i]:.2f}",
                    f"{data_processor.values_B_m[i]:.2f}",
                    f"{data_processor.values_mu_m[i]:.2f}"
                ])
                tabular.add_hline()
            table.append(tabular)
            table.add_caption('Измерения 2')
        doc_handler.add_graph('other_gyst.png', r"Петля гистерезиса")
        doc_handler.append_content(r' Масштаб по оси $X$: справа внизу указано, что 1 деление по'
                                   r' горизонтали соответствует $50$ мс.\\'
                                   r' Масштаб по оси $Y$: в нижней части экрана видно, что 1 деление по вертикали равно $50$ мВ.\\'
                                   r'$S_{\text{ПГ}} = 5.5 \text{дел}^2$\\')
        Xi = data_processor.calculate_X(0.1,0.05,0.5)
        print(Xi)
        doc_handler.append_content(r'\\ $\chi$ = '+f"{data_processor.represent_as_power_of_10(data_processor.calculate_X(0.1,0.05,40))} " + r" Дж/с")
        doc_handler.append_content(r'\\ $P$ = '+
            f"{data_processor.represent_as_power_of_10(data_processor.calculate_P(Xi,5.5))}" + r" Вт")
        data_processor.process_data(data, 3)
        with doc_handler.doc.create(Table(position='h!')) as table:
            table.append(Command('centering'))
            tabular = Tabular('|c|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$U$, В"),
                NoEscape(r"$U_x$, мВ"),
                NoEscape(r"$H$, А/м"),
                NoEscape(r"$U_y$, мВ"),
                NoEscape(r"$B$, Тл"),
                NoEscape(r"$\mu$")
            ])
            tabular.add_hline()

            for i in range(len(data_processor.values_U)):
                tabular.add_row([
                    f"{data_processor.values_U[i]}",
                    f"{data_processor.values_U_x_3[i]:.2f}",
                    f"{data_processor.values_H[i]:.2f}",
                    f"{data_processor.values_U_y_3[i]:.2f}",
                    f"{data_processor.values_B[i]:.2f}",
                    f"{data_processor.values_mu[i]:.2f}"
                ])
                tabular.add_hline()
            table.append(tabular)
            table.add_caption('Результаты прямых измерений и расчетов')

        plotter.plot_graph(data_processor.values_H, data_processor.values_B, 'B_vs_H.png',
                               r"$H$", r"$B$  ", r"График зависимости $B$ от $H$")
        doc_handler.add_graph('B_vs_H.png', r"График зависимости $B$ от $H$")
        plotter.plot_graph(data_processor.values_H, data_processor.values_mu, 'mu_vs_H.png',
                           r"$H$", r"$\mu$  ", r"График зависимости $B$ от $H$")
        doc_handler.add_graph('mu_vs_H.png', r"График зависимости $\mu$ от $H$")
        doc_handler.append_content(r'Максимальное значение магнитной проницаемости: ' + f"{data_processor.calculate_max_mu_th():.2f}" +r"\\")
        doc_handler.append_content(r'Напряженность: ' + f"{data_processor.calculate_H_for_max_mu():.2f}" + r" А/м\\")
        doc_handler.append_content(r'Относительная погрешность мощности: ' + r"$8.3 \%$" + r" Вт")
    # Итоги
    doc_handler.append_content(r"\newpage")
    with doc_handler.doc.create(Section("Выводы")):
        doc_handler.append_content(r'В ходе выполнения данной лабораторной работы были рассчитаны коэрцитивная сила, остаточная индукция, магнитная проницаемость вещества, а также были построены соответсвующие графики зависимости от напряженности.')

    doc_handler.generate_pdf("lab_report_3.07")