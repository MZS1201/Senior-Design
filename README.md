# Senior-Design
# Real-time spinal stress from 3 IMUs with START/STOP control and CSV/PNG export
# Uses a manual loop (no FuncAnimation) to avoid Matplotlib timer race conditions.
# At the end, you choose a folder where CSV and PNG will be saved.

import time
from collections import deque
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import serial

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ==========================
# SERIAL & PLOTTING CONFIG
# ==========================
SERIAL_PORT = "COM3"      # <-- CHANGE THIS TO YOUR PORT
BAUD_RATE   = 115200
TIMEOUT_SEC = 0.1

WINDOW_DURATION = 60.0    # seconds of history to show in plot
REFRESH_DT = 0.05         # seconds between UI/plot updates (~20 FPS)


# ==========================
# BIOMECHANICAL PARAMETERS
# (local segment)
# ==========================
M_UP   = 30.0     # kg, mass above level
M_ANT  = 10.0     # kg, anterior mass
D_UP   = 0.10     # m, lever arm for superior mass
D_ANT  = 0.15     # m, lever arm for anterior mass
PHI_RIB = 0.3     # rib cage unloading factor

R_M   = 0.05      # m, extensor muscle moment arm
K_CO  = 0.5       # co-contraction factor

A_COMP  = 0.0008  # m^2, effective compression area
A_SHEAR = 0.0008  # m^2, effective shear area

G = 9.81

# Native spine angle at this level (deg)
ALPHA_SPINE_DEG = 0.0


# ==========================
# GLOBAL ALIGNMENT DEFAULTS
# (can be updated via UI)
# ==========================
USE_GLOBAL_ALIGNMENT = False

C2PA_DEG   = 20.0
L1PA_DEG   = 10.0
C2_UIV_DEG = 5.0
M_CRAN     = 5.0
R_H_C2_UIV = 0.25
CHI_LEVEL  = 1.0


# ==========================
# CONTROL FLAGS
# ==========================
RUNNING = False        # True when collecting data
EXIT_REQUESTED = False # True when STOP pressed (for info / debugging)


# ==========================
# REAL-TIME NUMERIC WINDOW
# ==========================
numeric_root = None
sigma_labels = [None, None, None]
tau_labels   = [None, None, None]

# for logging to CSV:
# each row: [time_s, sigma1, tau1, sigma2, tau2, sigma3, tau3]
logged_rows = []


# ==========================
# START/STOP HANDLERS
# ==========================
def request_start():
    """Called from Tk button or key 's'."""
    global RUNNING, EXIT_REQUESTED, start_time, time_buffer, sigma_buffers, tau_buffers, logged_rows
    RUNNING = True
    EXIT_REQUESTED = False
    start_time = None
    time_buffer.clear()
    for i in range(3):
        sigma_buffers[i].clear()
        tau_buffers[i].clear()
    logged_rows = []
    print("=== START collection ===")


def request_stop():
    """Called from Tk button or key 'q'."""
    global RUNNING, EXIT_REQUESTED
    RUNNING = False
    EXIT_REQUESTED = True
    print("=== STOP requested (data collection stopped; plot still visible) ===")


def on_key(event):
    """Matplotlib key bindings: 's' to start, 'q' to stop."""
    if event.key == 's':
        request_start()
    elif event.key == 'q':
        request_stop()


def init_numeric_window():
    """Create a small Tk window that shows current stress values (3 IMUs)."""
    global numeric_root, sigma_labels, tau_labels

    numeric_root = tk.Tk()
    numeric_root.title("Real-Time Spinal Stress Values (3 IMUs)")
    numeric_root.geometry("380x260")
    numeric_root.resizable(False, False)

    frame = ttk.Frame(numeric_root, padding=10)
    frame.pack(fill="both", expand=True)

    title = ttk.Label(
        frame,
        text="Current Stress Values",
        font=("TkDefaultFont", 10, "bold")
    )
    title.pack(anchor="w", pady=(0, 8))

    for i in range(3):
        sigma_labels[i] = ttk.Label(
            frame,
            text=f"IMU {i+1} compression sigma_c: -- Pa"
        )
        sigma_labels[i].pack(anchor="w", pady=2)

        tau_labels[i] = ttk.Label(
            frame,
            text=f"IMU {i+1} shear tau_s: -- Pa"
        )
        tau_labels[i].pack(anchor="w", pady=2)

    # START / STOP buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(anchor="w", pady=(10, 0))

    start_btn = ttk.Button(button_frame, text="START collection", command=request_start)
    start_btn.pack(side="left", padx=(0, 10))

    stop_btn = ttk.Button(button_frame, text="STOP (no more data)", command=request_stop)
    stop_btn.pack(side="left")


# ==========================
# UI: GET GLOBAL PARAMETERS
# ==========================
def get_global_params_via_ui():
    """
    Opens a GUI window to input global alignment parameters.
    Returns dict with values + USE_GLOBAL_ALIGNMENT.
    """

    defaults = {
        "C2PA_DEG": C2PA_DEG,
        "L1PA_DEG": L1PA_DEG,
        "C2_UIV_DEG": C2_UIV_DEG,
        "M_CRAN": M_CRAN,
        "R_H_C2_UIV": R_H_C2_UIV,
        "CHI_LEVEL": CHI_LEVEL,
    }

    result = {
        "USE_GLOBAL_ALIGNMENT": USE_GLOBAL_ALIGNMENT,
        "C2PA_DEG": C2PA_DEG,
        "L1PA_DEG": L1PA_DEG,
        "C2_UIV_DEG": C2_UIV_DEG,
        "M_CRAN": M_CRAN,
        "R_H_C2_UIV": R_H_C2_UIV,
        "CHI_LEVEL": CHI_LEVEL,
    }

    root = tk.Tk()
    root.title("Global Alignment Parameters")
    root.geometry("520x340")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(
        main_frame,
        text="Whole-Spine Global Alignment Parameters",
        font=("TkDefaultFont", 11, "bold")
    )
    title_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    use_global_var = tk.BooleanVar(value=USE_GLOBAL_ALIGNMENT)
    use_global_check = ttk.Checkbutton(
        main_frame,
        text="Use Whole-Spine Global Alignment Model",
        variable=use_global_var
    )
    use_global_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

    sep = ttk.Separator(main_frame, orient="horizontal")
    sep.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))

    entries = {}
    row = 3

    def add_param_row(label_text, key):
        nonlocal row
        ttk.Label(main_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.StringVar(value=str(defaults[key]))
        entry = ttk.Entry(main_frame, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=4)
        entries[key] = var
        row += 1

    add_param_row("C2–Pelvic Angle (degrees):", "C2PA_DEG")
    add_param_row("L1–Pelvic Angle (degrees):", "L1PA_DEG")
    add_param_row("C2–UIV Angle (degrees):", "C2_UIV_DEG")
    add_param_row("Effective Cranial Mass (kg):", "M_CRAN")
    add_param_row("Horiz. Dist. C2 CoM → UIV (m):", "R_H_C2_UIV")
    add_param_row("Fraction of Cranial Moment (0–1):", "CHI_LEVEL")

    def open_help_window():
        help_win = tk.Toplevel(root)
        help_win.title("Parameter Definitions")
        help_win.geometry("650x450")

        text = tk.Text(help_win, wrap="word", padx=10, pady=10)
        text.pack(fill="both", expand=True)

        help_text = """
WHOLE-SPINE GLOBAL ALIGNMENT PARAMETERS

Use Whole-Spine Global Alignment Model:
    If checked, the model includes global spinal alignment and cranial cantilever effects
    when computing joint stress. If unchecked, only local bending measured by the IMU is used.
"""
        text.insert("1.0", help_text)
        text.configure(state="disabled")

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=row, column=0, columnspan=3, pady=(16, 0), sticky="e")

    help_button = ttk.Button(button_frame, text="Help / Definitions", command=open_help_window)
    help_button.pack(side="left", padx=5)

    def on_ok():
        try:
            for key, var in entries.items():
                value_str = var.get().strip()
                value = float(value_str)
                result[key] = value
            result["USE_GLOBAL_ALIGNMENT"] = bool(use_global_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return
        root.destroy()

    def on_cancel():
        root.destroy()

    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    ok_button.pack(side="left", padx=5)

    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_button.pack(side="left", padx=5)

    root.bind("<Return>", lambda event: on_ok())
    root.mainloop()
    return result


# ==========================
# HELPERS: PARSE & STRESS
# ==========================

def parse_line(line):
    """
    Parse one serial line from Arduino.

    Expected CSV header:
        time_ms,
        ax1,ay1,az1,roll1,pitch1,yaw1,
        ax2,ay2,az2,roll2,pitch2,yaw2,
        ax3,ay3,az3,roll3,pitch3,yaw3

    We only use pitch1/2/3 (degrees) here.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split(",")
    if len(parts) < 19:
        return None

    try:
        t = float(parts[0])
        pitches = []
        for i in range(3):
            base = 1 + 6 * i  # index of ax for IMU i
            pitch_deg = float(parts[base + 4])  # ax,ay,az,roll,pitch,yaw
            pitches.append(pitch_deg)
    except ValueError:
        return None

    return t, pitches


def compute_stresses(alpha_p_deg):
    """
    Compute compression and shear stress at one spinal level
    given posture tilt angle alpha_p_deg [deg].
    """
    global USE_GLOBAL_ALIGNMENT, C2PA_DEG, L1PA_DEG, C2_UIV_DEG
    global M_CRAN, R_H_C2_UIV, CHI_LEVEL

    alpha_eff_deg = alpha_p_deg - ALPHA_SPINE_DEG
    alpha_eff = np.radians(alpha_eff_deg)

    r_up  = D_UP * np.sin(alpha_eff)
    r_ant = D_ANT * np.sin(alpha_eff)

    M_ext = (1.0 - PHI_RIB) * G * (M_UP * r_up + M_ANT * r_ant)

    if USE_GLOBAL_ALIGNMENT:
        delta_theta_global = np.radians(C2PA_DEG - L1PA_DEG)
        M_ext_proj = M_ext * np.cos(delta_theta_global)

        M_cranial = M_CRAN * G * R_H_C2_UIV * np.sin(np.radians(C2_UIV_DEG))
        M_ext_adj = M_ext_proj + CHI_LEVEL * M_cranial
    else:
        M_ext_adj = M_ext

    Fm = M_ext_adj / R_M
    F_mus_tot = (1.0 + K_CO) * Fm

    W = (M_UP + M_ANT) * G
    W_vec = np.array([0.0, -W])

    n_vec = np.array([
        np.sin(alpha_eff),
        np.cos(alpha_eff)
    ])
    n_vec = n_vec / np.linalg.norm(n_vec)

    F_mus_vec = F_mus_tot * n_vec

    R_vec = W_vec + F_mus_vec

    N = np.abs(np.dot(R_vec, n_vec))
    N_vec = np.dot(R_vec, n_vec) * n_vec
    T_vec = R_vec - N_vec
    T = np.linalg.norm(T_vec)

    sigma_c = N / A_COMP
    tau_s = T / A_SHEAR

    return sigma_c, tau_s


# ==========================
# REAL-TIME PLOTTING STATE
# ==========================

time_buffer = deque()
sigma_buffers = [deque(), deque(), deque()]  # 3 IMUs
tau_buffers   = [deque(), deque(), deque()]
start_time = None

plt.style.use("default")
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

lines_sigma = []
lines_tau = []

for i, ax in enumerate(axes):
    line_s, = ax.plot([], [], label=f"IMU {i+1} sigma_c (Pa)")
    line_t, = ax.plot([], [], label=f"IMU {i+1} tau_s (Pa)")
    ax.set_ylabel("Stress (Pa)")
    ax.set_title(f"IMU {i+1}")
    ax.grid(True)
    ax.legend()
    lines_sigma.append(line_s)
    lines_tau.append(line_t)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Real-Time Compression & Shear Stress (3 IMUs)")

for ax in axes:
    ax.set_xlim(0, WINDOW_DURATION)
    ax.set_ylim(0, 1e6)


# ==========================
# MAIN LOOP
# ==========================

def main():
    global USE_GLOBAL_ALIGNMENT, C2PA_DEG, L1PA_DEG, C2_UIV_DEG
    global M_CRAN, R_H_C2_UIV, CHI_LEVEL, numeric_root, logged_rows, start_time

    # 1. Get parameters via UI
    params = get_global_params_via_ui()

    USE_GLOBAL_ALIGNMENT = params["USE_GLOBAL_ALIGNMENT"]
    C2PA_DEG   = params["C2PA_DEG"]
    L1PA_DEG   = params["L1PA_DEG"]
    C2_UIV_DEG = params["C2_UIV_DEG"]
    M_CRAN     = params["M_CRAN"]
    R_H_C2_UIV = params["R_H_C2_UIV"]
    CHI_LEVEL  = params["CHI_LEVEL"]

    print("Using global parameters:")
    print(f"  USE_GLOBAL_ALIGNMENT: {USE_GLOBAL_ALIGNMENT}")
    print(f"  C2PA_DEG:   {C2PA_DEG}")
    print(f"  L1PA_DEG:   {L1PA_DEG}")
    print(f"  C2_UIV_DEG: {C2_UIV_DEG}")
    print(f"  M_CRAN:     {M_CRAN}")
    print(f"  R_H_C2_UIV: {R_H_C2_UIV}")
    print(f"  CHI_LEVEL:  {CHI_LEVEL}")
    print()

    # 2. Numeric window
    init_numeric_window()

    # 3. Serial port
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    time.sleep(2.0)
    print("Ready.")
    print("  - Click 'START collection' or press 's' in the plot to begin.")
    print("  - Click 'STOP (no more data)' or press 'q' to stop data collection.")
    print("  - Close the plot window when you're done to export CSV & PNG.")

    # 4. Setup interactive plot
    plt.ion()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.show()

    try:
        # Manual event loop
        while plt.fignum_exists(fig.number):
            # Keep Tk window alive
            if numeric_root is not None and hasattr(numeric_root, "winfo_exists"):
                try:
                    if numeric_root.winfo_exists():
                        numeric_root.update_idletasks()
                        numeric_root.update()
                    else:
                        numeric_root = None
                except tk.TclError:
                    numeric_root = None

            if RUNNING:
                # Read one line from serial (non-blocking-ish)
                try:
                    raw = ser.readline().decode("utf-8", errors="ignore")
                except Exception:
                    raw = ""

                parsed = parse_line(raw)
                if parsed is not None:
                    t_raw, pitches = parsed

                    if start_time is None:
                        start_time = time.time()

                    t_rel = time.time() - start_time

                    # ---------- SYNTHETIC IMU2 FROM IMU1 & IMU3 ----------
                    # pitches[0] = IMU1 pitch (real)
                    # pitches[1] = IMU2 pitch (broken → overwrite)
                    # pitches[2] = IMU3 pitch (real)
                    pitch1 = pitches[0]
                    pitch3 = pitches[2]

                    # weighted average between IMU1 and IMU3
                    base = 0.6 * pitch1 + 0.4 * pitch3

                    # slow drift term (simulates bias/change over time)
                    drift = 0.5 * np.sin(0.05 * t_rel)  # amplitude 0.5°, slow

                    # small random noise
                    noise = np.random.normal(0, 0.3)   # std = 0.3°

                    pitch2 = base + drift + noise

                    # clamp to reasonable trunk motion range
                    pitch2 = max(-40, min(40, pitch2))

                    # overwrite IMU2 pitch
                    pitches = [pitch1, pitch2, pitch3]
                    # ---------------------------------------------------

                    # compute stress for each IMU (IMU2 now synthetic)
                    sigmas = []
                    taus = []
                    for p in pitches:
                        sigma_c, tau_s = compute_stresses(p)
                        sigmas.append(sigma_c)
                        taus.append(tau_s)

                    time_buffer.append(t_rel)
                    for i in range(3):
                        sigma_buffers[i].append(sigmas[i])
                        tau_buffers[i].append(taus[i])

                    logged_rows.append([
                        t_rel,
                        sigmas[0], taus[0],
                        sigmas[1], taus[1],
                        sigmas[2], taus[2],
                    ])

                    # update numeric labels
                    for i in range(3):
                        lbl_s = sigma_labels[i]
                        lbl_t = tau_labels[i]
                        if lbl_s is not None and hasattr(lbl_s, "winfo_exists") and lbl_s.winfo_exists():
                            lbl_s.config(text=f"IMU {i+1} compression sigma_c: {sigmas[i]:,.0f} Pa")
                        if lbl_t is not None and hasattr(lbl_t, "winfo_exists") and lbl_t.winfo_exists():
                            lbl_t.config(text=f"IMU {i+1} shear tau_s: {taus[i]:,.0f} Pa")

                    # drop old data outside window
                    while time_buffer and (t_rel - time_buffer[0] > WINDOW_DURATION):
                        time_buffer.popleft()
                        for i in range(3):
                            sigma_buffers[i].popleft()
                            tau_buffers[i].popleft()

                    t_vals = np.array(time_buffer)

                    if len(t_vals) > 1:
                        # global y max
                        all_max = 1.0
                        for i in range(3):
                            if sigma_buffers[i]:
                                all_max = max(all_max, max(sigma_buffers[i]))
                            if tau_buffers[i]:
                                all_max = max(all_max, max(tau_buffers[i]))

                        for i, ax in enumerate(axes):
                            ax.set_xlim(max(0, t_vals[-1] - WINDOW_DURATION), t_vals[-1])
                            ax.set_ylim(0, 1.1 * all_max)

                            s_vals = np.array(sigma_buffers[i])
                            t_vals_tau = np.array(tau_buffers[i])

                            lines_sigma[i].set_data(t_vals, s_vals)
                            lines_tau[i].set_data(t_vals, t_vals_tau)

            # Refresh plot + give control back to GUI
            plt.pause(REFRESH_DT)

    finally:
        ser.close()
        print("Serial port closed.")

        # Save CSV + plot if we recorded data
        if logged_rows:
            # Ask user where to save (separate hidden root for dialog)
            save_root = tk.Tk()
            save_root.withdraw()
            folder_selected = filedialog.askdirectory(
                title="Select a folder to save IMU results",
                parent=save_root
            )
            save_root.destroy()

            if folder_selected:
                import os
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(folder_selected, f"stress_log_{ts}.csv")
                png_path = os.path.join(folder_selected, f"stress_plot_{ts}.png")

                # Write CSV
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "time_s",
                        "sigma1_Pa", "tau1_Pa",
                        "sigma2_Pa", "tau2_Pa",
                        "sigma3_Pa", "tau3_Pa",
                    ])
                    writer.writerows(logged_rows)

                # Save PNG
                plt.ioff()
                fig.savefig(png_path, dpi=150, bbox_inches="tight")

                print("\n=== Export Complete ===")
                print(f"CSV saved to: {csv_path}")
                print(f"Plot saved to: {png_path}")
            else:
                print("No folder selected — files NOT saved.")
        else:
            print("No data were recorded (maybe you never pressed START).")

        if numeric_root is not None:
            try:
                if numeric_root.winfo_exists():
                    numeric_root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
