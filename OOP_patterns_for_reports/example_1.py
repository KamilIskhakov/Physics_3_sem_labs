
import numpy as np
import matplotlib.pyplot as plt
from pylatex import Document, NoEscape, Figure, Tabular, Section, MultiColumn, Package
import math

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

    def add_graph(self, image_filename, caption):
        """
        Функция для добавления графика в центр.
        :param image_filename: Название файла нужного графика.
        :param caption: Описание графика.
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_filename, width='400px')
            fig.add_caption(NoEscape(caption))


    def generate_pdf(self, filename):
        self.doc.generate_pdf(filename, clean_tex=False)

class DataProcessor(IDataProcessor):
    def __init__(self, params):
        self.params = params
        self.Bc_values = []
        self.e_m_values = []
        self.U_values = []
        self.critical_points = []
        self.delta_U_values = []
        self.delta_B_c_values = []
        self.delta_r_a_values = []

    def process_data(self, data, params):
        for U, values in data.items():
            IL = values['IL']
            Ia = values['Ia']

            diff = np.diff(Ia)
            diff = list(map(abs, diff))
            idx = np.argmax(diff)

            Bc = self.calculate_Bc(IL[idx], params) * 10**3
            e_m = self.calculate_em(U, Bc * 10**-3, params['ra'])

            self.Bc_values.append(Bc)
            self.e_m_values.append(e_m)
            self.U_values.append(U)
            self.critical_points.append([U, round(IL[idx], 3), round(Bc, 3), self.represent_as_power_of_10(e_m)])
            self.delta_U_values.append(0.1)
            self.delta_B_c_values.append(0.001)
            self.delta_r_a_values.append(0.0001)

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

# Main Execution

if __name__ == "__main__":
    # Initialize components
    doc_handler = DocumentHandler("lab_report")
    data_processor = DataProcessor(params={'N': 1500, 'l': 0.036, 'd': 0.037, 'ra': 0.003})
    plotter = Plotter()

    # Create title page
    doc_handler.append_content(r'\begin{center}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(
        r'{\LARGE\textbf{Лабораторная работа № 3.03: \\ Определение удельного заряда электрона}}\\[1cm]')
    doc_handler.append_content(r'{\Large Исхаков Камиль Фархатович}\\[1cm]')
    doc_handler.append_content(r'{\Large \today}')
    doc_handler.append_content(r'\vspace*{\fill}')  # Add vertical space
    doc_handler.append_content(r'\end{center}')
    doc_handler.append_content(r'\newpage')


    # Add data and formulas
    with doc_handler.doc.create(Section('Основные формулы')):
        doc_handler.append_content(r"\begin{displaymath} B_c = \mu_0 I_c N \frac{1}{\sqrt{(l^2+d^2)}} \end{displaymath}")
        doc_handler.append_content(r"где $\mu_0 = 4 \pi 10^{-7}$ Гн/м – магнитная постоянная, $N$ – число витков соленоида, $l$ – его длина, $d$ – его диаметр.")
        doc_handler.append_content(r"\begin{displaymath} \frac{e}{m} = \frac{8 U}{B_c^2 r_a^2} \end{displaymath}")
        doc_handler.append_content(r"где $e$ – заряд электрона, $m$ – его масса, $r_a$ – радиус анода, $U$ – анодное напряжение.")

    # Process data
    data = {
        12: {'IL': [0.00000001, 0.039, 0.063, 0.094, 0.124, 0.150, 0.177, 0.218, 0.250, 0.280, 0.313, 0.339, 0.370, 0.404, 0.431, 0.470, 0.505, 0.536, 0.570, 0.6], 'Ia': [0.279, 0.278, 0.277, 0.278, 0.279, 0.280, 0.278, 0.263, 0.207, 0.159, 0.122, 0.103, 0.082, 0.067, 0.059, 0.049, 0.043, 0.039, 0.036, 0.034]},
        14: {'IL': [0.00000001, 0.030, 0.049, 0.086, 0.109, 0.144, 0.172, 0.200, 0.225, 0.251, 0.275, 0.303, 0.330, 0.359, 0.393, 0.425, 0.448, 0.477, 0.500, 0.523], 'Ia': [0.353, 0.352, 0.351, 0.351, 0.352, 0.353, 0.352, 0.348, 0.336, 0.272, 0.220, 0.181, 0.153, 0.128, 0.101, 0.087, 0.078, 0.069, 0.064, 0.059]},
        15: {'IL': [0.00000001, 0.033, 0.057, 0.082, 0.106, 0.136, 0.156, 0.185, 0.208, 0.236, 0.267, 0.282, 0.303, 0.330, 0.358, 0.385, 0.412, 0.441, 0.469, 0.5], 'Ia': [0.389, 0.388, 0.389, 0.388, 0.389, 0.392, 0.392, 0.389, 0.383, 0.365, 0.266, 0.239, 0.207, 0.174, 0.149, 0.123, 0.106, 0.093, 0.082, 0.074]}
    }
    data_processor.process_data(data, data_processor.params)

    # Add tables
    with doc_handler.doc.create(Section("Результаты эксперимента")):
        with doc_handler.doc.create(Tabular('|c|c|c|c|')) as table:
            table.add_hline()
            table.add_row([NoEscape(r"$U$, В"), NoEscape(r"$I_{L_c}$, мА"), NoEscape(r"$B_c$, мТл"), NoEscape(r"$e/m$, Кл/кг")])
            table.add_hline()
            for row in data_processor.critical_points:
                table.add_row(row)
                table.add_hline()

    # Plot and add graphs
    plotter.plot_graph(data_processor.U_values, [ (x*10**-3)**2 for x in data_processor.Bc_values], 'Bc2_vs_U.png', "Анодное напряжение, В", r"$B_c^2$, $\text{Тл}^2$", r"График зависимости $B_c^2$ от $U$")
    # Add the caption to the figure
    doc_handler.add_graph('Bc2_vs_U.png',r"График зависимости $B_c^2$ от анодного напряжения $U$")

    # Error calculation
    with doc_handler.doc.create(Section("Оценка погрешностей")):
        doc_handler.append_content(r'Оценим погрешность удельного заряда электрона:')
        doc_handler.append_content(r'Вычислим для каждого из значений анодного напряжения и критического тока:')
        doc_handler.append_content(r'$$\Delta \left( \frac{e}{m} \right) = \sqrt{ \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial U} \Delta U \right)^2 + \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial B_c} \Delta B_c \right)^2 + \left( \frac{\partial \left( \frac{e}{m} \right)}{\partial r_a} \Delta r_a \right)^2 }$$')
        doc_handler.append_content(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial U} = \frac{8}{B_c^2 r_a^2}$$')
        doc_handler.append_content(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial B_c} = -\frac{16U}{B_c^3 r_a^2}$$')
        doc_handler.append_content(r'$$\frac{\partial \left( \frac{e}{m} \right)}{\partial r_a} = -\frac{16U}{B_c^2 r_a^3}$$')

        mean_em = np.mean(data_processor.e_m_values)
        for error_data in data_processor.calculate_errors(data_processor.U_values, data_processor.delta_U_values, data_processor.Bc_values, data_processor.delta_B_c_values, data_processor.params['ra'], data_processor.delta_r_a_values):
            U, delta_em, relative_error = error_data
            doc_handler.append_content(f'Оценим погрешность для U = {U} В: ')
            doc_handler.append_content(r'$$\Delta \left( \frac{e}{m} \right) = \sqrt{ \left( \frac{8 \Delta U}{B_c^2 r_a^2} \right)^2 + \left( \frac{16U \Delta B_c}{B_c^3 r_a^2} \right)^2 + \left( \frac{16U \Delta r_a}{B_c^2 r_a^3} \right)^2 } = \text{' + data_processor.represent_as_power_of_10(delta_em) + r'} \text{Кл/кг}$$')
            doc_handler.append_content(f'Относительная погрешность: {relative_error:.2f}'+ r"\%")
            doc_handler.append_content(r'\newline')

    # Summary
    with doc_handler.doc.create(Section("Выводы")):
        doc_handler.append_content(r'Табличное значение удельного заряда электрона: $\frac{e}{m} = 1,76\cdot10^{11}$ Кл/кг')
        doc_handler.append_content(r'В ходе работы был определен удельный заряд электрона. Табличное и экспериментальное значение удельного заряда электрона в нашем случае получилось почти идентичными. Во время выполнения всей лабораторной работы значения получались вполне реалистичными, кроме задания с построением зависимости $B^2_c$ от $U$, поскольку значение удельного заряда, вычисленный через коэффициент наклона прямой оказалось в полтора раза больше табличного значения. Расхождение теоретического значения удельного заряда от экспериментального может быть из-за влияния облака заряда, накапливающегося в диоде.')

    # Generate PDF
    doc_handler.generate_pdf("lab_report")