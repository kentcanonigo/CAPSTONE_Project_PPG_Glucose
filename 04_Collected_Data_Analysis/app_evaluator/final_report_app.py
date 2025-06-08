import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys
import threading
from datetime import datetime

class FinalReportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Final Comparative Analysis Report")
        self.root.geometry("650x600")

        # --- Define Paths ---
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.results_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")
            self.log_file_path = os.path.join(self.results_dir, "fine_tuned_model_evaluation_log.csv")
        except Exception as e:
            messagebox.showerror("Path Error", f"Could not determine project paths: {e}")
            self.root.destroy()

        self.processing_in_progress = False

        self._setup_gui()
        self._check_log_file()

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        # Info Frame
        info_frame = ttk.LabelFrame(main_frame, text="Report Configuration", padding="10")
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, text="Input Data File:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_frame, text="fine_tuned_model_evaluation_log.csv", font="TkDefaultFont 9 italic").grid(row=0, column=1, sticky="w")
        
        # Action Button
        self.process_button = ttk.Button(main_frame, text="Generate Final Comparison Report", command=self._start_report_thread)
        self.process_button.grid(row=1, column=0, pady=10, ipady=5)

        # Results Table
        results_frame = ttk.LabelFrame(main_frame, text="Table 4.4: Comparative Performance of Single vs. Fused Approaches", padding="10")
        results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_cols = ["Approach", "Avg. mARD (%)", "Avg. RMSE (mg/dL)", "Avg. MAE (mg/dL)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=4)
        
        col_widths = {"Approach": 200, "Avg. mARD (%)": 120, "Avg. RMSE (mg/dL)": 120, "Avg. MAE (mg/dL)": 120}
        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, anchor='center', width=col_widths.get(col, 120), minwidth=100)
            
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        # Log Viewer
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding="5")
        log_frame.grid(row=3, column=0, sticky="ew", pady=(10,0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="ew")
        
        main_frame.rowconfigure(2, weight=1)

    def _log_message(self, message):
        self.root.after(0, lambda: self._update_log_text(message))

    def _update_log_text(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _check_log_file(self):
        if not os.path.exists(self.log_file_path):
            self.process_button.config(state=tk.DISABLED)
            msg = f"Input file not found. Please run the Fine-Tuner App first to generate:\n\n{self.log_file_path}"
            self._log_message(f"ERROR: {msg}")
            messagebox.showerror("File Not Found", msg)
        else:
             self._log_message("Ready to generate report. Found the required input file.")

    def _start_report_thread(self):
        if self.processing_in_progress: return
        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        self._log_message("--- Generating Final Report ---")
        threading.Thread(target=self._run_report_generation, daemon=True).start()

    def _run_report_generation(self):
        try:
            # 1. Load the detailed evaluation data
            self._log_message(f"Reading data from {self.log_file_path}")
            results_df = pd.read_csv(self.log_file_path)
            
            # 2. Isolate results for each category
            single_finger_approaches = ["Index Finger", "Middle Finger", "Ring Finger"]
            fusion_approaches = ["SNR-Weighted Fusion", "SQI-Selected Fusion"]
            
            single_finger_df = results_df[results_df['Approach'].isin(single_finger_approaches)]
            fusion_df = results_df[results_df['Approach'].isin(fusion_approaches)]

            if single_finger_df.empty or fusion_df.empty:
                raise ValueError("Log file is missing results for single-finger or fusion methods.")

            # 3. Calculate average performance for single fingers
            # This complex groupby first averages segments per participant, then averages participants.
            avg_single_finger_perf = single_finger_df.groupby(['ParticipantID', 'Approach']).mean(numeric_only=True).groupby(level='Approach').mean()
            # Then we average the 3 finger results together
            final_avg_single_finger = avg_single_finger_perf.mean()
            
            # 4. Calculate average performance for fusion methods
            avg_fusion_perf = fusion_df.groupby('Approach').mean(numeric_only=True)

            # 5. Prepare data for display
            display_data = []
            
            # Add average single finger row
            display_data.append([
                "Average Single-Finger",
                f"{final_avg_single_finger['mARD(%)']:.2f}",
                f"{final_avg_single_finger['RMSE(mg/dL)']:.2f}",
                f"{final_avg_single_finger['MAE(mg/dL)']:.2f}"
            ])
            
            # Add fusion rows
            for approach in fusion_approaches:
                if approach in avg_fusion_perf.index:
                    row = avg_fusion_perf.loc[approach]
                    display_data.append([
                        approach,
                        f"{row['mARD(%)']:.2f}",
                        f"{row['RMSE(mg/dL)']:.2f}",
                        f"{row['MAE(mg/dL)']:.2f}"
                    ])

            self.root.after(0, self._update_results_display, display_data)
            
            # 6. Log the discussion point
            sqi_mard = avg_fusion_perf.loc['SQI-Selected Fusion']['mARD(%)']
            avg_single_mard = final_avg_single_finger['mARD(%)']
            improvement = ((avg_single_mard - sqi_mard) / avg_single_mard) * 100
            self._log_message(f"Discussion Point: SQI Fusion resulted in a {improvement:.2f}% mARD reduction vs. the single-finger average.")

            self._log_message("\n--- Report Generation Complete ---")

        except Exception as e:
            import traceback
            error_msg = f"ERROR during report generation: {e}\n{traceback.format_exc()}"
            self._log_message(error_msg)
            messagebox.showerror("Report Error", error_msg)
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def _update_results_display(self, results_list):
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        for row in sorted(results_list, key=lambda x: x[0]):
            self.results_tree.insert("", tk.END, values=row)

if __name__ == '__main__':
    root = ThemedTk(theme="arc")
    app = FinalReportApp(root)
    root.mainloop()