import numpy as np
import skfuzzy
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
person = ctrl.Antecedent(np.arange(0, 11, 1), 'person')
ac_temperature = ctrl.Consequent(np.arange(16, 31, 1), 'ac_temperature')

# Define membership functions
temperature['cold'] = mf.trapmf(temperature.universe, [0, 0, 10, 20])
temperature['medium'] = mf.trapmf(temperature.universe, [10, 20, 22, 30])
temperature['hot'] = mf.trapmf(temperature.universe, [22, 30, 40, 40])

person['empty'] = mf.trapmf(person.universe, [0, 0, 2, 3])
person['medium'] = mf.trapmf(person.universe, [2, 3, 5, 6])
person['crowded'] = mf.trapmf(person.universe, [5, 6, 10, 10])

ac_temperature['output_low'] = mf.trimf(ac_temperature.universe, [16, 18, 22])
ac_temperature['output_medium'] = mf.trimf(ac_temperature.universe, [20, 22, 24])
ac_temperature['output_high'] = mf.trimf(ac_temperature.universe, [22, 26, 30])

def fuzzy_inference(temperature, person, temperature_value, person_value):
    # Menghitung derajat keanggotaan
    temperature_degree = dict()

    for term, membership in temperature.terms.items():
        temperature_degree[term] = skfuzzy.interp_membership(
            temperature.universe, membership.mf, temperature_value)

    person_degree = dict()

    for term, membership in person.terms.items():
        person_degree[term] = skfuzzy.interp_membership(
            person.universe, membership.mf, person_value)

    # Menghitung Firing Strength untuk setiap anggota
    firing_strength = dict()
    for t in temperature.terms.keys():
        for p in person.terms.keys():
            firing_strength[(t, p)] = min(
                temperature_degree[t], person_degree[p])

    # Menghitung weighted average pada Temperatur AC
    numerator = 0
    denominator = 0

    for (t, p), strength in firing_strength.items():
        ac_membership_high = mf.trimf(ac_temperature.universe, [16, 20, 24])
        ac_membership_medium = mf.trimf(ac_temperature.universe, [22, 24, 26])
        ac_membership_low = mf.trimf(ac_temperature.universe, [24, 28, 30])

        ac_membership = (
            (ac_membership_high if (t == 'hot') and (p == 'crowded') else 0) +
            (ac_membership_medium if (t == 'medium') and (p == 'medium') else 0) +
            (ac_membership_low if (t == 'cold') and (p == 'empty') else 0)
        )

        numerator += strength * np.sum(ac_temperature.universe * ac_membership)
        denominator += strength * np.sum(ac_membership)

    if denominator != 0:
        ac_temperature_output = numerator / denominator
    else:
        ac_temperature_output = numerator

    return ac_temperature_output

class FuzzyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Inference System")

        ttk.Label(self.root, text="Temperature").grid(row=0, column=0, padx=10, pady=10)
        self.temp_entry = ttk.Entry(self.root)
        self.temp_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(self.root, text="Number of Persons").grid(row=1, column=0, padx=10, pady=10)
        self.person_entry = ttk.Entry(self.root)
        self.person_entry.grid(row=1, column=1, padx=10, pady=10)

        ttk.Button(self.root, text="Calculate", command=self.calculate_output).grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(self.root, text="Show Membership Functions", command=self.plot_membership_functions).grid(row=3, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(self.root, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def calculate_output(self):
        temperature_value = float(self.temp_entry.get())
        person_value = float(self.person_entry.get())
        output = fuzzy_inference(temperature, person, temperature_value, person_value)
        self.result_label.config(text=f"AC Temperature Output: {output:.2f}")

    def plot_membership_functions(self):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

        # Plot temperature membership functions
        for term, membership in temperature.terms.items():
            axes[0].plot(temperature.universe, membership.mf, label=term)

        axes[0].set_title('Temperature Membership Functions')
        axes[0].legend()
        axes[0].axvline(float(self.temp_entry.get()), color='r', linestyle='--', label='Input Value')

        # Plot person membership functions
        for term, membership in person.terms.items():
            axes[1].plot(person.universe, membership.mf, label=term)

        axes[1].set_title('Person Membership Functions')
        axes[1].legend()
        axes[1].axvline(float(self.person_entry.get()), color='r', linestyle='--', label='Input Value')

        # Plot AC temperature membership functions
        for term, membership in ac_temperature.terms.items():
            axes[2].plot(ac_temperature.universe, membership.mf, label=term)

        axes[2].set_title('AC Temperature Membership Functions')
        axes[2].legend()
        axes[2].axvline(float(self.temp_entry.get()), color='r', linestyle='--', label='Input Value')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyGUI(root)
    root.mainloop()
s