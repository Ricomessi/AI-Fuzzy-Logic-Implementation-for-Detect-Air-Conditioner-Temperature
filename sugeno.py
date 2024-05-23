import numpy as np
import pandas as pd
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Define fuzzy logic system
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

ac_temperature['output_low'] = mf.trimf(ac_temperature.universe, [16, 20, 24])
ac_temperature['output_medium'] = mf.trimf(ac_temperature.universe, [22, 24, 26])
ac_temperature['output_high'] = mf.trimf(ac_temperature.universe, [24, 28, 30])

# Define rules
rule1 = ctrl.Rule(temperature['cold'] & person['empty'], ac_temperature['output_high'])
rule2 = ctrl.Rule(temperature['cold'] & person['medium'], ac_temperature['output_medium'])
rule3 = ctrl.Rule(temperature['cold'] & person['crowded'], ac_temperature['output_medium'])

rule4 = ctrl.Rule(temperature['medium'] & person['crowded'], ac_temperature['output_medium'])
rule5 = ctrl.Rule(temperature['medium'] & person['empty'], ac_temperature['output_high'])
rule6 = ctrl.Rule(temperature['medium'] & person['medium'], ac_temperature['output_medium'])

rule7 = ctrl.Rule(temperature['hot'] & person['empty'], ac_temperature['output_medium'])
rule8 = ctrl.Rule(temperature['hot'] & person['medium'], ac_temperature['output_low'])
rule9 = ctrl.Rule(temperature['hot'] & person['crowded'], ac_temperature['output_low'])

ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
ac = ctrl.ControlSystemSimulation(ac_ctrl)

# Create GUI
class FuzzyLogicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Logic System")

        # Temperature input
        ttk.Label(root, text="Temperature:").grid(column=0, row=0)
        self.temperature_var = tk.DoubleVar()
        self.temperature_entry = ttk.Entry(root, textvariable=self.temperature_var)
        self.temperature_entry.grid(column=1, row=0)

        # Person input
        ttk.Label(root, text="Number of Persons:").grid(column=0, row=1)
        self.person_var = tk.IntVar()
        self.person_entry = ttk.Entry(root, textvariable=self.person_var)
        self.person_entry.grid(column=1, row=1)

        # Output label
        self.output_label = ttk.Label(root, text="Output AC Temperature:")
        self.output_label.grid(column=0, row=2, columnspan=2)

        # Calculate button
        ttk.Button(root, text="Calculate", command=self.calculate_output).grid(column=0, row=3, columnspan=2)

        # Show Membership Functions button
        ttk.Button(root, text="Show Membership Functions", command=self.plot_membership_functions).grid(column=0, row=4, columnspan=2)

    def calculate_output(self):
        # Update fuzzy logic system inputs
        ac.input['temperature'] = self.temperature_var.get()
        ac.input['person'] = self.person_var.get()

        # Compute the result
        ac.compute()

        # Update the output label
        self.output_label["text"] = f"Output AC Temperature: {ac.output['ac_temperature']:.2f}"

    def plot_membership_functions(self):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

        # Plot temperature membership functions
        ax[0].plot(temperature.universe, mf.trapmf(temperature.universe, [0, 0, 10, 20]), label='Cold')
        ax[0].plot(temperature.universe, mf.trapmf(temperature.universe, [10, 20, 22, 30]), label='Medium')
        ax[0].plot(temperature.universe, mf.trapmf(temperature.universe, [22, 30, 40, 40]), label='Hot')
        ax[0].set_title('Temperature Membership Functions')
        ax[0].legend()
        ax[0].axvline(self.temperature_var.get(), color='r', linestyle='--', label='Input Value')
        ax[0].legend()

        # Plot person membership functions
        ax[1].plot(person.universe, mf.trapmf(person.universe, [0, 0, 2, 3]), label='Empty')
        ax[1].plot(person.universe, mf.trapmf(person.universe, [2, 3, 5, 6]), label='Medium')
        ax[1].plot(person.universe, mf.trapmf(person.universe, [5, 6, 10, 10]), label='Crowded')
        ax[1].set_title('Person Membership Functions')
        ax[1].legend()
        ax[1].axvline(self.person_var.get(), color='r', linestyle='--', label='Input Value')
        ax[1].legend()

        # Plot AC temperature membership functions
        ax[2].plot(ac_temperature.universe, mf.trimf(ac_temperature.universe, [16, 20, 24]), label='Low')
        ax[2].plot(ac_temperature.universe, mf.trimf(ac_temperature.universe, [22, 24, 26]), label='Medium')
        ax[2].plot(ac_temperature.universe, mf.trimf(ac_temperature.universe, [24, 28, 30]), label='High')
        ax[2].set_title('AC Temperature Membership Functions')
        ax[2].legend()
        ax[2].axvline(ac.output['ac_temperature'], color='r', linestyle='--', label='Output Value')
        ax[2].legend()

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    root = tk.Tk()
    fuzzy_logic_gui = FuzzyLogicGUI(root)
    root.mainloop()
