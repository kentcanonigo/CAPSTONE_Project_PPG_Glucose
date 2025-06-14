import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
from datetime import datetime
import threading

class FinalReportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Final Comparative Analysis Report")
        self.root.withdraw() 
        self.root.geometry("650x600")

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.results_dir = os.path.join(project_root, "04_Collected_Data_Analysis", "evaluation_results")
            
            # This is the correct file that the fine_tuner_app now saves
            self.log_file_path = os.path.join(self.results_dir, "fine_tuned_aggregated_results.csv")

        except Exception as e:
            messagebox.showerror("Path Error", f"Could not determine project paths: {e}")
            self.root.destroy()
            return

        self.processing_in_progress = False
        self._setup_gui()
        self._check_log_file()
        self._center_window()
        self.root.deiconify()

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        info_frame = ttk.LabelFrame(main_frame, text="Report Configuration", padding="10")
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, text="Input Data File:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_frame, text=os.path.basename(self.log_file_path), font="TkDefaultFont 9 italic").grid(row=0, column=1, sticky="w")
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=10)
        control_frame.columnconfigure(0, weight=1) 
        self.process_button = ttk.Button(control_frame, text="Generate Final Comparison Report", command=self._start_report_thread)
        self.process_button.pack(pady=5, ipady=5) 

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
            msg = f"Input file not found. Please run the Fine-Tuner App first to generate the log file:\n\n{os.path.basename(self.log_file_path)}"
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
            self._log_message(f"Reading data from {os.path.basename(self.log_file_path)}")
            results_df = pd.read_csv(self.log_file_path)
            
            single_finger_approaches = ["Index Finger", "Middle Finger", "Ring Finger"]
            single_finger_df = results_df[results_df['Approach'].isin(single_finger_approaches)]

            if single_finger_df.empty:
                raise ValueError("Log file is missing results for individual finger methods.")

            avg_single_finger_metrics = single_finger_df[self.results_cols[1:]].mean(numeric_only=True)
            
            display_data = []
            display_data.append([
                "Average Single-Finger",
                f"{avg_single_finger_metrics['Avg. mARD (%)']:.2f}",
                f"{avg_single_finger_metrics['Avg. RMSE (mg/dL)']:.2f}",
                f"{avg_single_finger_metrics['Avg. MAE (mg/dL)']:.2f}"
            ])
            self._log_message(f"Calculated Average Single-Finger Performance: mARD {avg_single_finger_metrics['Avg. mARD (%)']:.2f}%")
            
            fusion_approaches = ["SNR-Weighted Fusion", "SQI-Selected Fusion"]
            for approach in fusion_approaches:
                approach_row = results_df[results_df['Approach'] == approach]
                if not approach_row.empty:
                    # Convert the row to a list to be displayed in the treeview
                    row_values = list(approach_row.iloc[0])
                    display_data.append(row_values)
                    self._log_message(f"Extracted {approach} Performance: mARD {row_values[1]}%")

            self.root.after(0, self._update_results_display, display_data)
            
            fusion_df = results_df[results_df['Approach'].isin(fusion_approaches)]
            if not fusion_df.empty:
                # Find the row with the minimum mARD among the fusion methods
                best_fusion_row = fusion_df.loc[fusion_df['Avg. mARD (%)'].idxmin()]
                
                best_fusion_name = best_fusion_row['Approach']
                best_fusion_mard = best_fusion_row['Avg. mARD (%)']
                avg_single_mard = avg_single_finger_metrics['Avg. mARD (%)']

                if avg_single_mard > 0:
                     improvement = ((avg_single_mard - best_fusion_mard) / avg_single_mard) * 100
                     # Create the correct log message using the dynamic winner
                     self._log_message(f"Discussion Point: The best fusion method ({best_fusion_name}) shows a {improvement:.2f}% improvement in mARD over the single-finger average.")

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
        
        display_order = {"Average Single-Finger": 0, "SNR-Weighted Fusion": 1, "SQI-Selected Fusion": 2}
        sorted_results = sorted(results_list, key=lambda r: display_order.get(r[0], 99))

        for row in sorted_results:
            self.results_tree.insert("", tk.END, values=row)

if __name__ == '__main__':
    # FIX: Removed the unnecessary check for SCRIPTS_LOADED_SUCCESSFULLY
    root = ThemedTk(theme="arc")
    app = FinalReportApp(root)
    root.mainloop()