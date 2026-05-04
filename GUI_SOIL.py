#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
import sys
import platform
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


#def resource_path(relative_path):
    #if hasattr(sys, '_MEIPASS'):
    #   return os.path.join(sys._MEIPASS, relative_path)
    #return os.path.join(os.path.abspath("."), relative_path)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Check Windows version
version = platform.version()
release = platform.release()

if release not in ["10", "11"]:
    root = tk.Tk()
    root.withdraw()  
    messagebox.showerror(
        "Unsupported Operating System",
        f"This app requires Windows 10 or newer.\n\nDetected OS: Windows {release}\nPlease update your system to run RocPF."
    )
    sys.exit()

# -----------------------------------------------------------
# 3. LAZY MODEL LOADING
# -----------------------------------------------------------
modelfs = None
modelss = None

scaler_XF = None
scaler_XS = None
scaler_FS = None
scaler_SS = None

def get_models_and_scalers():
    """Load model and scalers only when needed (Predict button)."""
    global modelfs, modelss, scaler_XF, scaler_XS, scaler_FS, scaler_SS

    if modelfs is None and modelss is None:
        
        import tensorflow as tf
        import joblib

        modelfs = tf.keras.models.load_model(resource_path("dnn_fos.h5"))
        modelss = tf.keras.models.load_model(resource_path("dnn_ss.h5"))

        scaler_XF = joblib.load(resource_path("scaler_XF.pkl"))
        scaler_XS = joblib.load(resource_path("scaler_XS.pkl"))
        scaler_FS = joblib.load(resource_path("scaler_FS.pkl"))
        scaler_SS = joblib.load(resource_path("scaler_SS.pkl"))
        
    return modelfs, modelss, scaler_XF, scaler_XS, scaler_FS, scaler_SS


# In[29]:


# Input ranges for validation

modelfs, modelss, scaler_XF, scaler_XS, scaler_FS , scaler_SS = get_models_and_scalers()

input_ranges = [
    (4, 200),   # Slope height
    (15, 75),   # Slope angle  
    (10, 30),   # Unit weight
    (0, 150),   # Cohesion
    (0, 50),     # Friction angle
    (0, 1),      # Pore pressure ratio
    (0, 0.6)    # Horizontal seismic coefficient
]

# FoS predict function
def predict_fos():
    
    try:
        
        # Get user inputs
        inputs = [
            float(entry_slope_height.get()),
            float(entry_slope_angle.get()),
            float(entry_unit_weight.get()),
            float(entry_cohesion.get()),
            float(entry_friction.get()),
            float(entry_pore.get()),
            float(entry_hseismic.get())
        ]

        # Validate input ranges
        for i, (val, (low, high)) in enumerate(zip(inputs, input_ranges)):
            if not (low <= val <= high):
                fos_value.set("—")
                verdict.set("—")
                label_name = labels[i][0].split('\t')[0]
                messagebox.showwarning("Input Out of Range", f"Input {label_name} should be in the range {low} - {high}.")
                return
            
        input_ranges_fos = [
            (4, 200),   # Slope height
            (15, 75),   # Slope angle  
            (0, 150),   # Cohesion
            (0, 50),     # Friction angle
            (0, 1),      # Pore pressure ratio
            (0, 0.6)    # Horizontal seismic coefficient
        ]

        # Get user inputs
        inputs_fos = [
            float(entry_slope_height.get()),
            float(entry_slope_angle.get()),
            float(entry_cohesion.get()),
            float(entry_friction.get()),
            float(entry_pore.get()),
            float(entry_hseismic.get())
        ]
        
        # Preprocess
        X_scaled_fos = scaler_XF.transform([inputs_fos])
        pred_scaled_fos = modelfs.predict(X_scaled_fos, verbose=0)[0][0]
        pred_fos = round(scaler_FS.inverse_transform([[pred_scaled_fos]])[0][0], 3)

        fos_temp.set(f"{pred_fos:.3f}")
        
        if 0 < pred_fos < 0.1:
            messagebox.showwarning(
            "Failure Condition",
            "Predicted FoS is extremely low.\n"
            "Results may be unstable."
            )
            
        if pred_fos <= 0:
            fos_value.set("—")
            verdict.set("—")
            messagebox.showwarning(
            "Failure Condition",
            "Predicted FoS ≤ 0.\n"
            "The slope is in a failure or post-failure state.\n"
            "Stability assessment is not applicable."
            )
        
        else:

            fos_value.set(f"{pred_fos:.3f}")

            kh = float(entry_hseismic.get())
            fos = float(fos_value.get())

            if kh == 0.0:  # Static condition
                if fos < 1.500:
                    verdict.set("Unstable")
                else:
                    verdict.set("Stable")
                    
            else:          # Pseudo-static condition
                if fos < 1.200:
                    verdict.set("Unstable")
                else:
                    verdict.set("Stable")
            
            

    except ValueError:
        fos_value.set("—")
        verdict.set("—")
        messagebox.showerror("Input Error", "Please enter valid numeric values.")


#input_ranges_ss = [
  #  (4, 200),    # Slope height
   # (15, 75),    # Slope angle
   # (10, 30),    # Unit weight
  #  (0, 150),    # Cohesion
   # (0, 50),     # Friction angle
   # (0, 1),      # Pore pressure ratio
   # (0, 0.6)     # Horizontal seismic coefficient
#]


# FoS predict function
def predict_ss():
    
    try:

        # Get user inputs
        inputs = [
            float(entry_slope_height.get()),
            float(entry_slope_angle.get()),
            float(entry_unit_weight.get()),
            float(entry_cohesion.get()),
            float(entry_friction.get()),
            float(entry_pore.get()),
            float(entry_hseismic.get())
        ]
        
        # Validate input ranges
        for i, (val, (low, high)) in enumerate(zip(inputs, input_ranges)):
            if not (low <= val <= high):
                ss_value.set("—")
                label_name = labels[i][0].split('\t')[0]
                messagebox.showwarning("Input Out of Range", f"Input {label_name} should be in the range {low} - {high}.")
                return

        # If FoS not predicted yet
        if fos_temp.get().strip() == "":
            ss_value.set("—")
            messagebox.showwarning(
            "Missing Step",
            "Please predict FoS first.\n"
            "Follow the button order after any input change."
            )

        else:

            if float(fos_temp.get()) <= 0:
                ss_value.set("—")
                messagebox.showwarning(
                "Failure Condition",
                "Predicted FoS ≤ 0.\n"
                "The slope is in a failure or post-failure state.\n"
                "Stability assessment is not applicable."
                )

            else:
                # Preprocess
                X_scaled_ss = scaler_XS.transform([inputs])
                pred_scaled_ss = modelss.predict(X_scaled_ss, verbose=0)[0][0]
                pred_ss = round(scaler_SS.inverse_transform([[pred_scaled_ss]])[0][0], 3)
            
                if pred_ss <= 0:
                    ss_value.set("—")
                    messagebox.showwarning(
                    "Invalid Shear Stress",
                    "Predicted shear stress ≤ 0."
                    )

                else:
                    ss_value.set(f"{pred_ss:.3f}")
            

    except ValueError:
        ss_value.set("—")
        messagebox.showerror("Input Error", "Please enter valid numeric values.")


# In[30]:


# ===============================
# MAIN WINDOW
# ===============================

root = tk.Tk()
root.title("SoilCF – Circular Failure Analysis Tool (DNN-assisted)")
#root.geometry("1190x650")
#root.state("zoomed")

root.update_idletasks()
width = 1210
height = 710
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")
root.resizable(False, False)

root.configure(bg="#FFD966")
#print(root.attributes())
#root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

root.iconbitmap(resource_path("soilcf_icon.ico"))

# Fonts and styles
label_font = ("Arial", 12, "bold")
entry_font = ("Arial", 12)

# Heading
tk.Label(root, text="Stability Calculator for Circular Failure in Soil Slopes", 
         bg="#2176c7", fg="white", font=("Arial", 24, "bold"), pady=10).pack(fill=tk.X)


# ===============================
# STRING VARIABLES (OUTPUTS)
# ===============================
fos_temp = tk.StringVar()
fos_value = tk.StringVar()
verdict = tk.StringVar()
ss_value = tk.StringVar()
strength_value = tk.StringVar()
normal_value = tk.StringVar()

def reset_outputs(*args):
    # Invalidate all outputs when any input changes
    fos_temp.set("")
    
    fos_value.set("—")
    verdict.set("—")
    
    ss_value.set("—")
    strength_value.set("—")
    normal_value.set("—")

            
# ===============================
# INPUT ENTRIES
# ===============================

entry_slope_height = tk.Entry(root)
entry_slope_angle = tk.Entry(root)
entry_unit_weight = tk.Entry(root)
entry_cohesion = tk.Entry(root)
entry_friction = tk.Entry(root)
entry_pore = tk.Entry(root)
entry_hseismic = tk.Entry(root)

entries = [
    entry_slope_height,
    entry_slope_angle,
    entry_unit_weight,
    entry_cohesion,
    entry_friction,
    entry_pore,
    entry_hseismic
]

labels = [
    ("Slope height (m)\t\t [4 - 200] :", entry_slope_height),
    ("Slope angle (°)\t\t [15 - 75] :", entry_slope_angle),
    ("Unit weight (kN/m³)\t\t [10 - 30] :", entry_unit_weight),
    ("Cohesion (kPa)\t\t [0 - 150] :", entry_cohesion),
    ("Friction angle (°)\t\t   [0 - 50] :", entry_friction),
    ("Pore pressure ratio\t\t     [0 - 1] :", entry_pore),
    ("Horizontal seismic coefficient\t  [0 - 0.6] :", entry_hseismic)
]


tk.Label(
    root,
    text="Enter Inputs:",
    bg="#FFD966",
    font=("Arial", 15, "bold") #, pady=-600
).place(x=60, y=110)


# ===============================
# PLACE INPUTS
# ===============================
x0, y0 = 60, 170
for i, (txt, ent) in enumerate(labels):
    tk.Label(root, text=txt, bg="#FFD966",
             font=("Arial", 12)).place(x=x0, y=y0 + i*50)
    ent.place(x=x0+330, y=y0 + i*50, width=100)


# ===============================
# OUTPUT FIELDS
# ===============================

tk.Entry(root, textvariable=fos_value, justify="center", state="readonly").place(x=730, y=190, width=90)
tk.Entry(root, textvariable=verdict, justify="center", state="readonly").place(x=1000, y=190, width=140)

tk.Entry(root, textvariable=ss_value, justify="center", state="readonly").place(x=1000, y=320, width=140)
tk.Entry(root, textvariable=strength_value, justify="center", state="readonly").place(x=1000, y=450, width=140)
tk.Entry(root, textvariable=normal_value, justify="center", state="readonly").place(x=1000, y=580, width=140)

# ===============================
# CALCULATION FUNCTIONS
# ===============================

def calculate_shear_strength():
    try:
        fos = float(fos_value.get())
        tau = float(ss_value.get())


        if fos <= 0.000:
            strength_value.set("—")
            messagebox.showwarning(
            "Failure Condition",
            "Predicted FoS ≤ 0.\n"
            "The slope is in a failure or post-failure state.\n"
            "Stability assessment is not applicable."
            )
            
        else:
            strength = fos * tau
            strength_value.set(f"{strength:.3f}")

    except:
        strength_value.set("—")
        messagebox.showerror("Error", "Predict FoS and shear stress first.")


def calculate_normal_stress():
    try:
        fos = float(fos_value.get())
        strength = float(strength_value.get())
        cohesion = float(entry_cohesion.get())
        phi = float(entry_friction.get())

        if fos <= 0.000:
            normal_value.set("—")
            messagebox.showwarning(
            "Failure Condition",
            "Predicted FoS ≤ 0.\n"
            "The slope is in a failure or post-failure state.\n"
            "Stability assessment is not applicable."
            )
            
        else:

            if strength <= cohesion:
                normal_value.set("—")
                messagebox.showwarning(
                "Information",
                "Base normal stress: Not Applicable.\n"
                "Shear strength is less than cohesion.\n"
                )

            else:

                if phi <= 0:
                    normal_value.set("—")
                    messagebox.showwarning(
                        "Invalid Input",
                        "Average base normal stress cannot be computed for friction angle ≤ 0°."
                    )

                else:
                    sigma = (strength - cohesion) / np.tan(np.radians(phi))
                    normal_value.set(f"{sigma:.3f}")

    except:
        normal_value.set("—")
        messagebox.showerror("Error", "Calculate shear strength first.")

# ===============================
# BUTTON STYLE
# ===============================
#style = tk.Style()
#style.configure("Blue.TButton", background="#1F4E79", foreground="white", font=("Arial", 10, "bold"), padding=6)

# ===============================
# SECTION HEADERS
# ===============================

tk.Label(root, text="FoS:", bg="#FFD966", font=("Arial", 12)).place(x=680, y=190)
tk.Label(root, text="Stability status:", bg="#FFD966", font=("Arial", 12)).place(x=870, y=190)

tk.Label(root, text="Average shear stress (kPa):", bg="#FFD966", font=("Arial", 12)).place(x=680, y=320)
tk.Label(root, text="Average shear strength (kPa):", bg="#FFD966", font=("Arial", 12)).place(x=680, y=450)
tk.Label(root, text="Average base normal stress (kPa):", bg="#FFD966", font=("Arial", 12)).place(x=680, y=580)


# ===============================
# BUTTONS
# ===============================

tk.Button(
    root,
    text="1. Predict FoS against Circular Failure",
    font=("Arial", 14, "bold"), bg="#8B0000", fg="white",
    command=predict_fos        # ✅ NO parentheses
).place(x=640, y=120, width=380)

tk.Button(
    root,
    text="2. Predict average shear stress along CSS",
    font=("Arial", 14, "bold"), bg="#8B0000", fg="white",   #"#275d8f",
    command=predict_ss         # ✅ NO parentheses
).place(x=640, y=250, width=420)

tk.Button(
    root,
    text="3. Calculate average shear strength along CSS",
    font=("Arial", 14, "bold"), bg="#275d8f", fg="white",
    command=calculate_shear_strength
).place(x=640, y=380, width=460)

tk.Button(
    root,
    text="4. Calculate average base normal stress along CSS",
    font=("Arial", 14, "bold"), bg="#275d8f", fg="white",
    command=calculate_normal_stress
).place(x=640, y=510, width=500)


tk.Label(
    root,
    text=(
        "FoS: Factor of Safety\nCSS: Critical Slip Surface\n\n\n"
        "*IMPORTANT:\n• Follow the button order (1 → 4).\n• For any change in input, predictions and calculations must be repeated in order.\n"
        "• This tool assumes user familiarity with slope stability principles."
    ),
    bg= "#FFD966",
    font=("Arial", 10),
    justify="left",
    anchor="w"
).place(x=60, y=530)


# Footer Frame
footer_frame = tk.Frame(root, bg="#ccc")
footer_frame.pack(side="bottom", fill="x", pady=0, padx=0)


# Left Footer Label

tk.Label(
    footer_frame,
    text="Factor of safety and average shear stress predicted using Deep Neural Networks",
    bg="#ccc",
    font=("Times New Roman", 12, "italic", "bold")
).pack(side="left", padx=10)


# Right Footer Label
tk.Label(
    footer_frame,
    text="Stability status based on IS 14243 (Part 2): 1995",
    bg="#ccc",
    font=("Times New Roman", 12, "italic", "bold")
).pack(side="right", padx=10)

root.mainloop()


# In[ ]:




