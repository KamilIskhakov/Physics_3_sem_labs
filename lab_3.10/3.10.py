import numpy as np
import matplotlib.pyplot as plt
from pylatex import Document, NoEscape, Figure, Tabular, Section, MultiColumn, Package, Table, Command
import math
from typing import List

from abc import ABC, abstractmethod

from scipy.optimize import fsolve


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
    def calculate_delta_theor(L, C, R):
        """
        Calculates the theoretical phase shift delta for an RLC circuit.

        Parameters:
        L (float): Inductance in henrys
        C (float): Capacitance in farads
        R (float): Resistance in ohms

        Returns:
        tuple: (delta_radians, delta_degrees) representing the phase shift in radians and degrees.
        """

        omega_0 = 1 / math.sqrt(L * C)

        beta = R / (2 * L)

        omega_sq_minus_beta_sq = omega_0 ** 2 - beta ** 2

        if omega_sq_minus_beta_sq < 0:
            raise ValueError("корень берется из отрицательного числа.")

        sqrt_omega_sq_minus_beta_sq = math.sqrt(omega_sq_minus_beta_sq)

        y = -sqrt_omega_sq_minus_beta_sq

        x = beta

        delta_theor = math.atan(y/ x)

        if delta_theor < 0:
            delta_theor += math.pi

        return delta_theor
    @staticmethod
    def represent_as_power_of_10(number):
        if number == 0:
            return NoEscape("$0$")

        exponent = int(math.log10(abs(number)))
        coefficient = number / (10 ** exponent)
        coefficient_rounded = round(coefficient, 3)

        return NoEscape(f"${coefficient_rounded} \\cdot 10^{{{exponent}}}$")

    def approximate_and_find_x(self, x_data, y_data, y_target, filename='plot.png'):
        coefficients = np.polyfit(x_data, y_data, 1)
        polynomial = np.poly1d(coefficients)

        def equation(x):
            return polynomial(x) - y_target

        x_initial_guess = np.mean(x_data)
        x_solution = fsolve(equation, x_initial_guess)

        plt.scatter(x_data, y_data, label='Data Points')

        x_curve = np.linspace(min(x_data), max(x_data), 100)
        y_curve = polynomial(x_curve)
        plt.plot(x_curve, y_curve, label=f'Fitted Degree-{1} Polynomial')

        plt.xlabel(NoEscape(r"$R$"))
        plt.ylabel(NoEscape(r"$\mu$"))
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return x_solution

    def calculate_Q(self):
        return (1/self.values_R_s[0])*math.sqrt(self.values_L[0]/0.000000022)

    def process_data(self, data, num_key):
        for key, values in data.items():
            if key == num_key:
                if num_key == 1:
                    self.values_R_1 = values['R']
                    self.values_T_1 = values['T']
                    self.values_2U_i = values['2U_i']
                    self.values_2U_i_n = values['2U_i_n']
                    self.values_n = values['n']
                    self.values_lambda = [1/x * np.log(y/z) for x, y, z in zip(self.values_n, self.values_2U_i,self.values_2U_i_n)]
                    self.values_Q = [(2 * math.pi)/(1 - np.exp(-2 *x))  for x in self.values_lambda]
                    R_0 = -1*self.approximate_and_find_x(self.values_R_1[:-1],self.values_lambda[:-1],0,'mu_vs_R.png')[0]
                    self.values_R_s = [R_0+x  for x in self.values_R_1]
                    self.values_L = [1000*((math.pi*x)**2*0.000000022)/y**2  for x, y in zip(self.values_R_s[:-1], self.values_lambda[:-1])]

                elif num_key == 2:
                    self.values_T_2 = values['C']
                    self.values_C = [0.022,0.033,0.047,0.47]
                    np.mean_L = np.mean(data_processor.values_L)*10**(-3)
                    self.values_T_th = [
                        (10 ** 6 * 2 * math.pi) / math.sqrt(abs((1 / (np.mean_L * x*10**(-6))) - (77.54 ** 2 / (4 * np.mean_L ** 2))))
                        for x in self.values_C]
                    self.values_delta_T = [(abs(x-y)/y) * 100 for x, y in zip(self.values_T_2, self.values_T_th)]
                    self.values_T_tomp = [(10 **6 *2*math.pi*math.sqrt(np.mean_L*x*10**(-6))) for x in self.values_C]
                    self.values_T_th_T_exp = [x / y for x, y in zip(self.values_T_th, self.values_T_2)]
                elif num_key == 3:
                    self.values_tT = values['tT']
                    self.delta_ph_exp = 2*math.pi*(self.values_tT[0]/86)
                    self.delta_ph_th = self.calculate_delta_theor(np.mean(data_processor.values_L)*10**(-3),0.000000022,40)

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
    def plot_graph_2(self, x_data, y_1_data,y_2_data, filename, xlabel, ylabel, title, legend_labels=None):
        plt.figure()
        plt.plot(x_data, y_1_data, 'o-', label=legend_labels[0] if legend_labels else None)
        plt.plot(x_data, y_2_data, 'o-', label=legend_labels[1] if legend_labels else None)
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
        r'{\LARGE\textbf{Лабораторная работа № 3.10: \\ Изучение свободных затухающих электромагнитных колебаний}}\\[1cm]')
    doc_handler.append_content(r'{\Large Исхаков Камиль Фархатович}\\[1cm]')
    doc_handler.append_content(r'{\Large \today}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(r'\end{center}')
    doc_handler.append_content(r'\newpage')

    with doc_handler.doc.create(Section('Основные формулы')):
        doc_handler.append_content(r"\begin{itemize}    \item $\lambda$ --- логарифмический декремент затухания;    \item $U_i$ --- амплитуда напряжения на конденсаторе в момент времени $i$;    \item $U_{i+n}$ --- амплитуда напряжения на конденсаторе в момент времени $i+n$;    \item $n$ --- число полных периодов, между моментами времени $i$ и $i+n$;    \item $\beta$ --- коэффициент затухания;    \item $T$ --- период колебаний;    \item $R$ --- полное сопротивление контура;    \item $L$ --- индуктивность катушки;    \item $C$ --- емкость конденсатора;    \item $Q$ --- добротность контура;    \item $R_M$ --- добавочное сопротивление магазина;    \item $R_0$ --- собственное сопротивление контура;    \item $R_{\text{крит}}$ --- критическое сопротивление контура.\end{itemize}\begin{equation*}    \lambda = \frac{1}{n}\ln{\frac{U_i}{U_{i+n}}}\end{equation*} \begin{equation*}    \lambda = \beta T = \frac{R}{L}\frac{\pi}{\sqrt{\frac{1}{LC} - \frac{R^2}{4L^2}}}\end{equation*} \begin{equation*}    R = R_M + R_0\end{equation*} \begin{equation*}    R_0 = -R_M|_{\lambda = 0}\end{equation*} \begin{equation*}    Q = \frac{2\pi}{1 - e^{-2\lambda}}\end{equation*} \begin{equation*}    R_{\text{крит}} = 2\sqrt{\frac{L}{C}}\end{equation*} \begin{equation*}    T = 2\pi\sqrt{LC}\end{equation*}")
        doc_handler.add_graph('lab_3_10_p1.png', r"Принципиальная электрическая схема установки")

    # Ваши данные
    data = {
        1: {'R': [0,20,40,60,80,100,300],'T': [86.0,86.0,86.0,86.0,86.0,86.0,86.0],'2U_i': [2*3.04,2*2.68,2*2.46,2*2.2,2*1.92,2*1.76,2*0.66],'2U_i_n': [2*1.14,2*0.76,2*0.54,2*0.36,2*0.26,2*0.18,2*0.06],'n': [3,3,3,3,3,3,2]}, # мВ
        2: {'C': [86.0,110,132,418]},
        3: {'tT': [22.5,134]},}
    data_processor.process_data(data, 1)

    with doc_handler.doc.create(Section("Результаты эксперимента")):
        with doc_handler.doc.create(Table(position='h!')) as table:
            table.append(Command('centering'))
            tabular = Tabular('|c|c|c|c|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$R_M$, Ом"),
                NoEscape(r"$T_M$, мс"),
                NoEscape(r"$2 U_i$, дел"),
                NoEscape(r"$2 U_{i+n}$, дел"),
                NoEscape(r"$n$"),
                NoEscape(r"$\lambda$"),
                NoEscape(r"$Q$"),
                NoEscape(r"$R$"),
                NoEscape(r"$L$")
            ])
            tabular.add_hline()

            for i in range(len(data_processor.values_R_1)-1):
                tabular.add_row([
                    f"{data_processor.values_R_1[i]:.2f}",
                    f"{data_processor.values_T_1[i]:.2f}",
                    f"{data_processor.values_2U_i[i]:.2f}",
                    f"{data_processor.values_2U_i_n[i]:.2f}",
                    f"{data_processor.values_n[i]}",
                    f"{data_processor.values_lambda[i]:.2f}",
                    f"{data_processor.values_Q[i]:.2f}",
                    f"{data_processor.values_R_s[i]:.2f}",
                    f"{data_processor.values_L[i]:.2f}"
                ])
                tabular.add_hline()
            tabular.add_row([
                f"{data_processor.values_R_1[-1]:.2f}",
                f"{data_processor.values_T_1[-1]:.2f}",
                f"{data_processor.values_2U_i[-1]:.2f}",
                f"{data_processor.values_2U_i_n[-1]:.2f}",
                f"{data_processor.values_n[-1]}",
                f"{data_processor.values_lambda[-1]:.2f}",
                f"{data_processor.values_Q[-1]:.2f}",
                f"{data_processor.values_R_s[-1]:.2f}",
                f" – "
            ])
            tabular.add_hline()
            table.append(tabular)
            table.add_caption('Измерения 1')

        doc_handler.append_content(r'Среднее значение индуктивности: ' + f"{np.mean(data_processor.values_L):.2f}" +r" мГн\\")
        doc_handler.append_content(r'Погрешность среднего значения индуктивности: ' + r"$0.16$" +r" мГн\\" )
        doc_handler.append_content(r'Для $R = $' + f"{data_processor.values_R_s[0]:.2f} " + r"посчитаем добротность $Q = \frac{1}{R} \cdot \sqrt{\frac{L}{C}}$= " + f"{13.18} " + r"\\")
        doc_handler.append_content(
            r'Критическое сопротивление $R = 1087$ Ом \\')
        doc_handler.append_content(
            r'$R_0 = $' +f' {data_processor.values_R_s[0]:.2f}' +r' Ом \\')
        doc_handler.append_content(
            r'Теоретическое значение критического сопротивления $R$ при $L = 11$ мГн: ' +f"{2*(math.sqrt((11*10**(-3))/(0.000000022))):.2f}" + r" Ом\\")
        data_processor.process_data(data, 3)
        doc_handler.append_content(
            r'Эксперементальное значение сдвига фаз: $\delta_{\text{эксп}} = $' + f" {data_processor.delta_ph_exp:.2f}" + r"\\")
        doc_handler.append_content(
            r'Теоретическое значение сдвига фаз: $\delta_{\text{теор}} = $' + f" {data_processor.delta_ph_th:.2f}" + r"\\")
        data_processor.process_data(data, 2)
        with doc_handler.doc.create(Table(position='h!')) as table:
            table.append(Command('centering'))
            tabular = Tabular('|c|c|c|c|c|c|')
            tabular.add_hline()
            tabular.add_row([
                NoEscape(r"$C$, мкФ"),
                NoEscape(r"$T_{\text{эксп}}$, мс"),
                NoEscape(r"$T_{\text{теор}}$, мс"),
                NoEscape(r"$\delta T, \%$"),
                NoEscape(r"$T_{\text{томп}}$, мс"),
                NoEscape(r"$T_{\text{теор}}/T_{\text{эксп}}$")
            ])
            tabular.add_hline()

            for i in range(len(data_processor.values_T_2)):
                tabular.add_row([
                    f"{data_processor.values_C[i]:}",
                    f"{data_processor.values_T_2[i]:.2f}",
                    f"{data_processor.values_T_th[i]:.2f}",
                    f"{data_processor.values_delta_T[i]:.2f}",
                    f"{data_processor.values_T_tomp[i]:.2f}",
                    f"{data_processor.values_T_th_T_exp[i]:.2f}"
                ])
                tabular.add_hline()

            table.append(tabular)

            table.add_caption('Измерения 2')

        plotter.plot_graph(data_processor.values_R_s, data_processor.values_Q, 'Q_vs_R.png',
                           r"$R$", r"$Q$  ", r"График зависимости $Q$ от $R$")

        doc_handler.add_graph('Q_vs_R.png', r"График зависимости $Q$ от $R$")
        plotter.plot_graph(data_processor.values_R_1[:-1], data_processor.values_lambda[:-1], 'Lambda_vs_R.png',
                           r"$R$", r"$\lambda$  ", r"График зависимости $\lambda$ от $R$")

        doc_handler.add_graph('Lambda_vs_R.png', r"График зависимости $\lambda$ от $R$")
        plotter.plot_graph_2(data_processor.values_C, data_processor.values_T_2,data_processor.values_T_th, 'T_vs_C.png',
                           r"$C$", r"$T$  ", r"График зависимости $T$ от $C$",[r"$T_{\text{эксп}}$",r"$T_{\text{теор}}$"])

        doc_handler.add_graph('T_vs_C.png', r"График зависимости $T$ от $C$")
        plotter.plot_graph_2(data_processor.values_C, [x**2 for x in data_processor.values_T_2], [x**2 for x in data_processor.values_T_th],
                             'T^2_vs_C.png',
                             r"$C$", r"$T^2$  ", r"График зависимости $T^2$ от $C$",
                             [r"$T_{\text{эксп}}^2$", r"$T_{\text{теор}}^2$"])

        doc_handler.add_graph('T^2_vs_C.png', r"График зависимости $T^2$ от $C$")
        doc_handler.add_graph('lab_3_10_petly.jpg', r"Фазовая кривая $I(U)$")
    # Итоги
    doc_handler.append_content(r"\newpage")
    with doc_handler.doc.create(Section("Выводы")):
        doc_handler.append_content(r'Расчеты показали, что полученные '
                                   r'экспериментальные значения периода, логарифмического декремента затухания,'
                                   r' добротности контура и фазового сдвига согласуются с теоретическими предсказаниями, хотя '
                                   r'наблюдаются некоторые расхождения, которые могут быть обусловлены неидеальностями экспериментальной установки.')

    doc_handler.generate_pdf("lab_report_3.10")