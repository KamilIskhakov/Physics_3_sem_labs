
import numpy as np
import matplotlib.pyplot as plt
from pylatex import Document, NoEscape, Figure, Tabular, Section, MultiColumn, Package, Table, Command
import math
from typing import List

# Interfaces
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

# Concrete Implementations

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

    def calculate_E(self,T_values, R_values):
        E_list = []
        par1_list = []
        par2_list = []
        n = len(T_values)

        if n % 2 == 1:  # Если длина массивов нечетная
            middle_index = n // 2
            for i in range(middle_index):  # Для первой половины точек
                T1 = T_values[i]
                R1 = R_values[i]
                T2 = T_values[i + middle_index]
                R2 = R_values[i + middle_index]
                par1_list.append(i+1)
                par2_list.append(i+1++ middle_index)
                alpha = abs(2*((T1*T2) / (T2 - T1))*math.log(R1/R2))
                E_list.append(alpha)

            # Обработка последней точки, сопоставленной с серединной
            T1 = T_values[middle_index]
            R1 = R_values[middle_index]
            T2 = T_values[-1]
            R2 = R_values[-1]
            alpha = abs(2 * ((T1 * T2) / (T2 - T1)) * math.log(R1 / R2))
            E_list.append(alpha)
            par1_list.append(middle_index+1)
            par2_list.append(len(T_values))
        else:  # Если длина массивов четная
            for i in range(n // 2):
                T1 = T_values[i]
                R1 = R_values[i]
                T2 = T_values[i + n // 2]
                R2 = R_values[i + n // 2]
                par1_list.append(i + 1)
                par2_list.append(i + 1 + n // 2)
                alpha = abs(2 * ((T1 * T2) / (T2 - T1)) * math.log(R1 / R2))
                E_list.append(alpha)

        return E_list, par1_list, par2_list

    def calculate_alpha(self,T_values, R_values):
        alpha_list = []
        par1_list = []
        par2_list = []
        n = len(T_values)

        if n % 2 == 1:  # Если длина массивов нечетная
            middle_index = n // 2
            for i in range(middle_index):  # Для первой половины точек
                T1 = T_values[i]
                R1 = R_values[i]
                T2 = T_values[i + middle_index]
                R2 = R_values[i + middle_index]
                par1_list.append(i+1)
                par2_list.append(i+1++ middle_index)
                alpha = abs((R2-R1) / (R1*T2 - R2*T1))
                alpha_list.append(alpha)

            # Обработка последней точки, сопоставленной с серединной
            T1 = T_values[middle_index]
            R1 = R_values[middle_index]
            T2 = T_values[-1]
            R2 = R_values[-1]
            alpha = abs((R2-R1) / (R2*T1 - R1*T2))
            alpha_list.append(alpha)
            par1_list.append(middle_index+1)
            par2_list.append(len(T_values))
        else:  # Если длина массивов четная
            for i in range(n // 2):
                T1 = T_values[i]
                R1 = R_values[i]
                T2 = T_values[i + n // 2]
                R2 = R_values[i + n // 2]
                par1_list.append(i + 1)
                par2_list.append(i + 1 + n // 2)
                alpha = abs((R2-R1) / (R2*T1 - R1*T2))
                alpha_list.append(alpha)

        return alpha_list, par1_list, par2_list

    def process_data(self, data, num_key):
        for key, values in data.items():
            if key == num_key:
                T = values['T']  # К
                Im = values['Im']  # мкА
                U = values['U']  # В

                if num_key == 1:
                    self.T_values_1 = []  # К
                    self.Im_values_1 = []
                    self.U_values_1 = []
                    self.R_values = []  # Ом
                    self.lnR_values = []
                    self.x_values = []
                    self.T_values_1.extend(T)
                    self.Im_values_1.extend(Im)
                    self.U_values_1.extend(U)

                    for i in range(len(T)):
                        R = U[i] / (Im[i] * 10 ** (-6))
                        self.R_values.append(round(R, 3))
                        lnR = np.log(R)
                        self.lnR_values.append(round(lnR, 3))
                        x = round(10 ** 3 / T[i], 3)
                        self.x_values.append(x)

                elif num_key == 2:
                    self.T_values_2 = []  # К
                    self.Im_values_2 = []
                    self.U_values_2 = []
                    self.kR_values = []  # кОм
                    self.t_values = []  # цельсии
                    self.T_values_2.extend(T)
                    self.Im_values_2.extend(Im)
                    self.U_values_2.extend(U)

                    for i in range(len(T)):
                        R = (U[i] / (Im[i] * 10 ** (-6))) * 10 ** (-3)  # кОм
                        self.kR_values.append(round(R, 3))
                        t = T[i] - 273
                        self.t_values.append(t)

    @staticmethod
    def calculate_Bc(IL, params):
        mu0 = 4 * np.pi * 10**-7
        N, l, d = params['N'], params['l'], params['d']
        return mu0 * N * IL / np.sqrt(l**2 + d**2)

    @staticmethod
    def calculate_em(U, Bc, ra):
        return 8 * U / (Bc**2 * ra**2)

    @staticmethod
    def represent_as_power_of_10(number):
        if number == 0:
            return NoEscape("$0$")
        exponent = int(math.log10(abs(number)))
        coefficient = number / (10**exponent)
        coefficient_rounded = round(coefficient, 3)
        return NoEscape(f"${coefficient_rounded} \\cdot 10^{{{exponent}}}$")

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
    # Initialize components
    doc_handler = DocumentHandler("lab_report")
    data_processor = DataProcessor(params={'N': 1500, 'l': 0.036, 'd': 0.037, 'ra': 0.003})
    plotter = Plotter()

    # Create title page
    doc_handler.append_content(r'\begin{center}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(
        r'{\LARGE\textbf{Лабораторная работа № 3.05: \\ Температурная зависимость электрического сопротивления металла и полупроводника}}\\[1cm]')
    doc_handler.append_content(r'{\Large Исхаков Камиль Фархатович}\\[1cm]')
    doc_handler.append_content(r'{\Large \today}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(r'\end{center}')
    doc_handler.append_content(r'\newpage')


    # Add data and formulas
    with doc_handler.doc.create(Section('Основные формулы')):
        doc_handler.add_text_with_math(r"Закон Ома для участка цепи:",r"R=\frac{U}{I}")
        doc_handler.append_content(r"где $R$ - сопротивление, $U$ - напряжение, $I$ -  сила ток\\")
        doc_handler.add_text_with_math(r"Cопротивление полупроводника:", r"R_{\text{п}} = R_m\exp{(\frac{E_g}{2kT})}")
        doc_handler.append_content(r"где $kT$ - средняя энергия теплового движения, $R_m$ - предел к которому стремится значение сопротивления полупроводника при повышении температуры\\")
        doc_handler.add_text_with_math(r"Формула для расчета ширины запрещенной зоны:", r"E_g=2k \cdot \frac{\Delta \ln({R_{\text{п}}})}{\Delta (1/T)}")
        doc_handler.append_content(r"где $k$ - постоянная Больцмана, $k=1,38*10^{-23} \textit{ Дж/К}=8,62*10^{-5}\textit{ эВ/К}$)\\")
        doc_handler.add_text_with_math(r"Зависимость сопротивления от температуры для металла при небольших диапазонах температур:", r"R_{\text{м}} = R_0(1+\alpha T)")
        doc_handler.append_content(r"где $R_0$ - сопротивление данного образца при температуре $0^\circ C$, $\alpha$ - температурный коэффициент сопротивления\\")
    # Process data
    data = {
        1: {'T': [305,315,325,335,345],'Im': [1159,1258,1337,1387,1424],'U': [0.360,0.256,0.184,0.128,0.091]},
        2: {'T': [345,335,325,315,305],'Im': [1098,1113,1125,1146,1164],'U': [1.496,1.468,1.439,1.412,1.384]}}
    data_processor.process_data(data, 1)

    # Add tables
    with doc_handler.doc.create(Section("Результаты эксперимента")):
        with doc_handler.doc.create(Table(position='h!')) as table:
            # Add centering command
            table.append(Command('centering'))
            # Create and populate the Tabular
            tabular = Tabular('|c|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$T$, К"),
                NoEscape(r"$I$, мкА"),
                NoEscape(r"$U$, В"),
                NoEscape(r"$R$, Ом"),
                NoEscape(r"$\ln{R}$"),
                NoEscape(r"$\frac{10^3}{T}$, 1/К")
            ])
            tabular.add_hline()
            for i in range(len(data_processor.T_values_1)):
                tabular.add_row([
                    f"{data_processor.T_values_1[i]}",
                    f"{data_processor.Im_values_1[i]}",
                    f"{data_processor.U_values_1[i]:.3f}",
                    f"{data_processor.R_values[i]:.3f}",
                    f"{data_processor.lnR_values[i]:.3f}",
                    f"{data_processor.x_values[i]:.3f}"
                ])
                tabular.add_hline()
            # Append the Tabular to the Table
            table.append(tabular)
            # Add caption with label
            table.add_caption('Полупроводниковый образец')

        data_processor.process_data(data, 2)
        with doc_handler.doc.create(Table(position='h!')) as table:
            # Add centering command
            table.append(Command('centering'))
            # Create and populate the Tabular
            tabular = Tabular('|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$T$, К"),
                NoEscape(r"$I$, мкА"),
                NoEscape(r"$U$, В"),
                NoEscape(r"$R$, кОм"),
                NoEscape(r"$t$, °C")
            ])
            tabular.add_hline()
            for i in range(len(data_processor.T_values_2)):
                tabular.add_row([
                    f"{data_processor.T_values_2[i]}",
                    f"{data_processor.Im_values_2[i]}",
                    f"{data_processor.U_values_2[i]:.3f}",
                    f"{data_processor.kR_values[i]:.3f}",
                    f"{data_processor.t_values[i]}°"
                ])
                tabular.add_hline()
            # Append the Tabular to the Table
            table.append(tabular)
            # Add caption with label
            table.add_caption('Металлический образец образец')

        # Calculate alpha
        alpha_list_2, par1_list_2, par2_list_2  = data_processor.calculate_alpha(data_processor.t_values,[x*10**(3) for x in data_processor.kR_values])
        with doc_handler.doc.create(Table(position='h!')) as table:
            # Add centering command
            table.append(Command('centering'))
            # Create and populate the Tabular
            tabular = Tabular('|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$i$"),
                NoEscape(r"$j$"),
                NoEscape(r"$\alpha_{ij}$, $10^{-3}$ $\text{К}^{-1}$")
            ])
            tabular.add_hline()
            for i in range(len(par1_list_2)):
                tabular.add_row([
                    f"{par1_list_2[i]}",
                    f"{par2_list_2[i]}",
                    f"{alpha_list_2[i]*10**3:.3f}"
                ])
                tabular.add_hline()
            tabular.add_row(
                [
                    MultiColumn(2, align="|c|", data=NoEscape(r"$\langle \alpha \rangle$")),
                    round(np.mean([x*10**(3) for x in alpha_list_2]),3)
                ]
            )
            tabular.add_hline()
            # Append the Tabular to the Table
            table.append(tabular)
            # Add caption with label
            table.add_caption('Температурный коэффициент сопротивления металла')
        E_list_1, par1_list_1, par2_list_1 = data_processor.calculate_E(data_processor.T_values_1,data_processor.R_values)
        with doc_handler.doc.create(Table(position='h!')) as table:
            # Add centering command
            table.append(Command('centering'))
            # Create and populate the Tabular
            tabular = Tabular('|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$i$"),
                NoEscape(r"$j$"),
                NoEscape(r"$E_{ij}$, Дж"),
                NoEscape(r"$E_{ij}$, эВ")
            ])
            tabular.add_hline()
            for i in range(len(par1_list_1)):
                tabular.add_row([
                    f"{par1_list_1[i]}",
                    f"{par2_list_1[i]}",
                    data_processor.represent_as_power_of_10(E_list_1[i]*(1.38*10**(-23))),
                    f"{E_list_1[i]*(8.61*10**(-5)):.3f}"
                ])
                tabular.add_hline()
            tabular.add_row(
                [
                    MultiColumn(2, align="|c|", data=NoEscape(r"$\langle E \rangle$")),
                    data_processor.represent_as_power_of_10(np.mean([x*(1.38*10**(-23)) for x in E_list_1])),round(np.mean([x*(8.61*10**(-5)) for x in E_list_1]),3)
                ]
            )
            tabular.add_hline()
            # Append the Tabular to the Table
            table.append(tabular)
            # Add caption with label
            table.add_caption('Ширина запрещенной зоны полупроводника')
        doc_handler.append_content(r"\newpage")

        plotter.plot_graph(data_processor.x_values, data_processor.lnR_values, 'LnR_vs_T.png',
                           r"$\frac{1}{T}$", r"$\ln{R}$  ", r"График зависимости $\ln{R}$ от $\frac{1}{T}$")
        # Add the caption to the figure
        doc_handler.add_graph('LnR_vs_T.png', r"График зависимости $\ln{R}$ от $\frac{1}{T}$")

        plotter.plot_graph(data_processor.t_values, data_processor.kR_values, 'kR_vs_t.png',
                           r"$t$", r"$R_\text{м}$ ", r"График зависимости $t$ от $R_\text{м}$")
        # Add the caption to the figure
        doc_handler.add_graph('kR_vs_t.png', r"График зависимости $t$ от $R_\text{м}$")
    # Error calculation
    with doc_handler.doc.create(Section("Оценка погрешностей")):
        doc_handler.append_content(r'Для погрешности температурного коэффициента сопротивления'
                                   r' используем формулу для стандартного отклонения среднего '
                                   r'значения: \[ \Delta \alpha = t_{\alpha, n}'
                                   r' \sqrt{\frac{\sum_{i=1}^n (\alpha_{ij} -'
                                   r' \langle \alpha \rangle)^2}{n(n-1)}}\]')

        # Инициализация и расчет
        t_alpha_n_example = 2.131  # Коэффициент Стьюдента для n = 5, доверие 95%
        calculator_1 = ErrorCalculator(t_alpha_n=t_alpha_n_example, measurements=[x*10**(3) for x in alpha_list_2])
        mean_value_1 = calculator_1.mean_value()
        error_1 = calculator_1.calculate_error()

        doc_handler.append_content(r"\\ Вычисленная погрешность ($\Delta \alpha$): "+f"{data_processor.represent_as_power_of_10(error_1*10**(-3))} " + r"$\frac{1}{^\circ C}$")
        doc_handler.append_content(r"\\Для погрешности ширины запрещенной зоны используем формулу для стандартного отклонения"
                                   r" среднего значения:\[ \Delta E_g = t_{\alpha, n} \sqrt{\frac{\sum_{i=1}^n (E_{gij} "
                                   r"- \langle E_g \rangle)^2}{n(n-1)}}\]")
        calculator_2 = ErrorCalculator(t_alpha_n=t_alpha_n_example, measurements= [x*(1.38*10**(-23)) for x in E_list_1])
        mean_value_2 = calculator_1.mean_value()
        error_2 = calculator_2.calculate_error()
        doc_handler.append_content(
            r"\\ Вычисленная абсолютная погрешность ($\Delta E_g$): " + f"{data_processor.represent_as_power_of_10(error_2)} " + r"Дж")
        calculator_3 = ErrorCalculator(t_alpha_n=t_alpha_n_example,
                                       measurements=[x*(8.61*10**(-5)) for x in E_list_1])
        mean_value_3 = calculator_1.mean_value()
        error_3 = calculator_3.calculate_error()
        doc_handler.append_content(
            r"\\ Вычисленная абсолютная погрешность ($\Delta E_g$): " + f"{np.round(error_3,3)} " + r"эВ")
        # Run calculations and display results data_processor.calculate_alpha(data_processor.t_values,



    # Summary

    with doc_handler.doc.create(Section("Выводы")):
        doc_handler.append_content(r'Табличное значение коэффициента сопротивления меди: $4,28 \cdot 10^{-3} \text{ К}^{-1}$\\')
        doc_handler.append_content(r'Табличное значение ширины запрещенной зоны германия: $0,72 \text{ эВ}$\\')
        doc_handler.append_content(r'Построенные графики имеют линейный вид, что согласуется с теоретическим поведением '
                                   r'полупроводника и металла при нагревании. Исходя из полученного среднего значения '
                                   r' коэффициента сопротивления металла и абсолютной погрешности, можно сделать вывод,'
                                   r'что наиболее вероятным металлическим образцом в данной лабораторной работе '
                                   r'является медь. Аналогично, исходя из полученного среднего значения '
                                   r' ширины запрещенной зоны полупроводника и абсолютной погрешности, можно сделать вывод,'
                                   r'что наиболее вероятным полупроводниковым образцом образцом в данной лабораторной работе '
                                   r'является германий. Также в данной лабораторной работе подтвердились следующие '
                                   r'теоретические выкладки – сопротивление металла при нагревании увеличивается, '
                                   r'а у полупроводника наоборот – уменьшается.')

    # Generate PDF
    doc_handler.generate_pdf("lab_report_3.05")