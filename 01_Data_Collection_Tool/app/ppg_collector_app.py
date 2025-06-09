import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk # For applying themes
import serial
import serial.tools.list_ports
import threading
import time
import csv
import os
from datetime import datetime
import pandas as pd
import random
import math
from playsound import playsound

class PPGDataCollectorApp:
    def __init__(self, root_window, initial_geometry="650x900"):
        self.root = root_window
        self.root.title("PPG Data Collector")
        self.root.withdraw() # Hide window initially

        self.root.geometry(initial_geometry) # Set initial size

        self.serial_connection = None
        self.is_collecting = False
        self.collected_ppg_data = []
        
        self.simulation_mode = tk.BooleanVar(value=False) 

        try:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            data_collection_tool_dir = os.path.dirname(app_dir)
            project_root_dir = os.path.dirname(data_collection_tool_dir)
            self.base_save_path = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data")
        except NameError: 
            self.base_save_path = os.path.join(os.getcwd(), "..", "..", "04_Collected_Data_Analysis", "Collected_Data")
            self.base_save_path = os.path.abspath(self.base_save_path)

        self.labels_dir = os.path.join(self.base_save_path, "Labels")
        self.raw_data_dir = os.path.join(self.base_save_path, "RawData")
        self.labels_file_path = os.path.join(self.labels_dir, "collected_labels.csv")
        self.labels_df = None 

        self._create_directories()
        self._setup_gui() 
        self._load_history_data() 

        self.app_dir = os.path.dirname(os.path.abspath(__file__))

        # Center the window just before making it visible
        self._center_window() 
        self.root.deiconify() # Make window visible now that it's centered
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

    def _center_window(self): # Removed width and height_str arguments
            """Centers the Tkinter window on the screen."""
            self.root.update_idletasks() # Process all pending Tkinter events to get accurate window size
            
            # Get current window width and height
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            x_coordinate = int((screen_width / 2) - (width / 2))
            y_coordinate = int((screen_height / 2) - (height / 2))
            
            self.root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

    def _play_alert_sound(self):

        sound_file_path = os.path.join(self.app_dir, "./assets/collection_done.wav")

        try:
            if os.path.exists(sound_file_path):
                # Run playsound in a separate thread to avoid blocking the GUI
                sound_thread = threading.Thread(target=playsound, args=(sound_file_path,), daemon=True)
                sound_thread.start()
                self._log_message("Played custom alert sound.")
            else:
                self._log_message(f"Alert sound file '{sound_file_name}' not found. Playing system bell.")
                self.root.bell()
        except Exception as e:
            self._log_message(f"Error playing sound with playsound: {e}. Playing system bell.")
            self.root.bell()

    def _create_directories(self):
        try:
            os.makedirs(self.labels_dir, exist_ok=True)
            os.makedirs(self.raw_data_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Directory Error", f"Could not create data directories in {self.base_save_path}: {e}")
            self.base_save_path = "Collected_Data_Fallback"
            self.labels_dir = os.path.join(self.base_save_path, "Labels")
            self.raw_data_dir = os.path.join(self.base_save_path, "RawData")
            self.labels_file_path = os.path.join(self.labels_dir, "collected_labels.csv")
            os.makedirs(self.labels_dir, exist_ok=True)
            os.makedirs(self.raw_data_dir, exist_ok=True)
            print(f"Warning: Using fallback save path: {self.base_save_path}")
        # Log message will be called after log_text is initialized in _setup_gui

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10 10 10 10") 
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Create Log Frame and self.log_text first for availability ---
        # We will .grid() this frame later in its final visual position
        self._log_frame_for_setup = ttk.LabelFrame(main_frame, text="Log", padding="5")
        self._log_frame_for_setup.columnconfigure(0, weight=1)
        self._log_frame_for_setup.rowconfigure(0, weight=1)
        self.log_text = tk.Text(self._log_frame_for_setup, height=10, width=60, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1) # Increased height
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(self._log_frame_for_setup, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        # Initial log message now that self.log_text exists
        self._log_message(f"Data will be saved in: {self.base_save_path}") 

        # --- Participant Info Section --- (Row 0 in main_frame)
        participant_frame = ttk.LabelFrame(main_frame, text="Participant Information", padding="10")
        participant_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0,5), padx=5) # Reduced bottom pady
        participant_frame.columnconfigure(1, weight=1); participant_frame.columnconfigure(3, weight=1)
        # (All participant entry fields and clear button go here as before)
        ttk.Label(participant_frame, text="Subj. ID:").grid(row=0, column=0, sticky=tk.W, padx=(0,5), pady=3)
        self.subject_id_entry = ttk.Entry(participant_frame, width=18)
        self.subject_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(participant_frame, text="Sample #:").grid(row=0, column=2, sticky=tk.W, padx=(10,5), pady=3)
        self.sample_num_entry = ttk.Entry(participant_frame, width=18)
        self.sample_num_entry.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=3)
        self.sample_num_entry.insert(0, "1")
        ttk.Label(participant_frame, text="Gender:").grid(row=1, column=0, sticky=tk.W, padx=(0,5), pady=3)
        self.gender_var = tk.StringVar()
        self.gender_combobox = ttk.Combobox(participant_frame, textvariable=self.gender_var, values=["Male", "Female", "Other"], state="readonly", width=15)
        self.gender_combobox.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=3); self.gender_combobox.current(0)
        ttk.Label(participant_frame, text="Age:").grid(row=1, column=2, sticky=tk.W, padx=(10,5), pady=3)
        self.age_entry = ttk.Entry(participant_frame, width=18)
        self.age_entry.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(participant_frame, text="Height (cm):").grid(row=2, column=0, sticky=tk.W, padx=(0,5), pady=3)
        self.height_entry = ttk.Entry(participant_frame, width=18)
        self.height_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(participant_frame, text="Weight (kg):").grid(row=2, column=2, sticky=tk.W, padx=(10,5), pady=3)
        self.weight_entry = ttk.Entry(participant_frame, width=18)
        self.weight_entry.grid(row=2, column=3, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(participant_frame, text="Glucose (mg/dL):").grid(row=3, column=0, sticky=tk.W, padx=(0,5), pady=3)
        self.glucose_entry = ttk.Entry(participant_frame, width=18)
        self.glucose_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=3, columnspan=3)
        self.clear_fields_button = ttk.Button(participant_frame, text="Clear Input Fields", command=self._clear_input_fields)
        self.clear_fields_button.grid(row=4, column=0, columnspan=4, pady=(10,5))


        # --- Serial Connection Section --- (Row 1 in main_frame)
        serial_frame = ttk.LabelFrame(main_frame, text="Device Connection", padding="10")
        serial_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5, padx=5) # Reduced bottom pady
        serial_frame.columnconfigure(1, weight=1)
        # (All serial connection widgets go here as before)
        ttk.Label(serial_frame, text="COM Port:").grid(row=0, column=0, sticky=tk.W, padx=(0,5), pady=3)
        self.com_port_combobox = ttk.Combobox(serial_frame, state="readonly", width=15)
        self.com_port_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=3)
        self.refresh_button = ttk.Button(serial_frame, text="Refresh Ports", command=self.refresh_com_ports)
        self.refresh_button.grid(row=0, column=2, pady=3, padx=(5,0))
        self.sim_mode_checkbox = ttk.Checkbutton(serial_frame, text="Simulation Mode", variable=self.simulation_mode, command=self._toggle_simulation_mode)
        self.sim_mode_checkbox.grid(row=0, column=3, sticky=tk.E, padx=(10,0), pady=3)
        self.connect_button = ttk.Button(serial_frame, text="Connect", command=self.toggle_serial_connection)
        self.connect_button.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5,3), padx=(0,5))
        self.connection_status_label = ttk.Label(serial_frame, text="Status: Disconnected", foreground="red", anchor=tk.E)
        self.connection_status_label.grid(row=1, column=2, columnspan=2, sticky=(tk.W,tk.E), pady=(5,3), padx=(5,0))
        self.refresh_com_ports()


        # --- Data Collection Section --- (Row 2 in main_frame)
        collection_frame = ttk.LabelFrame(main_frame, text="Data Collection", padding="10")
        collection_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5, padx=5) # Reduced bottom pady
        collection_frame.columnconfigure(1, weight=1) 
        # (Duration entry and Start button go here as before)
        ttk.Label(collection_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W, padx=(0,5), pady=3) 
        self.duration_entry = ttk.Entry(collection_frame, width=18) 
        self.duration_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=3, columnspan=3) 
        self.duration_entry.insert(0, "20")
        self.start_collection_button = ttk.Button(collection_frame, text="Start Collection & Save New Entry", command=self.start_data_collection_and_save)
        self.start_collection_button.grid(row=1, column=0, columnspan=4, pady=(10,5), ipady=5) 
        self.start_collection_button.config(state=tk.DISABLED)

        # --- History Section --- (Row 3 in main_frame)
        history_frame = ttk.LabelFrame(main_frame, text="Collected Labels History (View Only)", padding="10")
        history_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5) # Reduced bottom pady
        history_frame.columnconfigure(0, weight=1); history_frame.rowconfigure(0, weight=1)
        # (Treeview and its scrollbars, and Refresh History button go here as before)
        self.history_cols_csv = ['Timestamp', 'ID', 'Sample_Num', 'Gender', 'Age', 'Height_cm', 'Weight_kg', 'Glucose_mgdL']
        self.history_cols_display = ['Timestamp', 'ID', 'Sample', 'Gender', 'Age', 'Height', 'Weight', 'Glucose']
        self.history_tree = ttk.Treeview(history_frame, columns=self.history_cols_display, show='headings', height=10) # Height already increased
        for i, display_col_name in enumerate(self.history_cols_display):
            self.history_tree.heading(display_col_name, text=display_col_name)
            if display_col_name == 'Timestamp': self.history_tree.column(display_col_name, width=130, anchor='w', minwidth=100)
            elif display_col_name in ['ID', 'Age', 'Gender']: self.history_tree.column(display_col_name, width=50, anchor='center', minwidth=40)
            elif display_col_name == 'Sample': self.history_tree.column(display_col_name, width=60, anchor='center', minwidth=50)
            else: self.history_tree.column(display_col_name, width=80, anchor='e', minwidth=60)
        hist_scrollbar_y = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        hist_scrollbar_x = ttk.Scrollbar(history_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=hist_scrollbar_y.set, xscrollcommand=hist_scrollbar_x.set)
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        hist_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hist_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E)) 
        self.load_history_button = ttk.Button(history_frame, text="Refresh History", command=self._load_history_data)
        self.load_history_button.grid(row=2, column=0, columnspan=2, pady=(5,0), sticky=tk.E) 

        # --- Grid the Log Frame (created earlier) into its final visual position --- (Row 4 in main_frame)
        self._log_frame_for_setup.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10,5), padx=5)

        # Configure main_frame row weights for expansion
        main_frame.columnconfigure(0, weight=1) 
        main_frame.rowconfigure(3, weight=1) # Allow history_frame (at row 3) to expand
        main_frame.rowconfigure(4, weight=1) # Allow log_frame (at row 4) to expand

    def _log_message(self, message):
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            if self.root and self.root.winfo_exists(): 
                 self.root.update_idletasks()
        else:
            print(f"LOG (log_text not ready or root destroyed): {message}")

    def refresh_com_ports(self):
        try:
            ports = [port.device for port in serial.tools.list_ports.comports()]
            self.com_port_combobox['values'] = ports
            if ports:
                self.com_port_combobox.current(0)
            else:
                self.com_port_combobox.set('')
            self._log_message("COM ports refreshed.")
        except Exception as e:
            self._log_message(f"Error refreshing COM ports: {e}")

    def _clear_input_fields(self, clear_id_too=True):
        if clear_id_too:
            self.subject_id_entry.delete(0, tk.END)
            self.sample_num_entry.delete(0, tk.END)
            self.sample_num_entry.insert(0, "1")
        self.gender_combobox.current(0) 
        self.age_entry.delete(0, tk.END)
        self.height_entry.delete(0, tk.END)
        self.weight_entry.delete(0, tk.END)
        self.glucose_entry.delete(0, tk.END)
        self._log_message("Input fields cleared.")

    def _toggle_simulation_mode(self):
        if self.simulation_mode.get():
            self._log_message("Simulation Mode ENABLED.")
            self.com_port_combobox.config(state=tk.DISABLED)
            self.refresh_button.config(state=tk.DISABLED)
            if self.serial_connection and self.serial_connection.is_open:
                self.toggle_serial_connection() 
            self.connection_status_label.config(text="Status: Simulation Mode", foreground="blue")
            self.connect_button.config(text="Simulate Connect", state=tk.NORMAL)
            self.start_collection_button.config(state=tk.DISABLED) 
        else: 
            self._log_message("Simulation Mode DISABLED.")
            self.com_port_combobox.config(state="readonly")
            self.refresh_button.config(state=tk.NORMAL)
            self.connect_button.config(text="Connect", state=tk.NORMAL) 
            if "SIM" in self.connection_status_label.cget("text").upper():
                self.connection_status_label.config(text="Status: Disconnected", foreground="red")
            self.start_collection_button.config(state=tk.DISABLED)

    def toggle_serial_connection(self):
        if self.simulation_mode.get():
            current_button_text = self.connect_button.cget("text")
            if "Simulate Connect" == current_button_text : 
                self.connection_status_label.config(text="Status: SIM Connected", foreground="blue")
                self.connect_button.config(text="Simulate Disconnect")
                self.start_collection_button.config(state=tk.NORMAL)
                self._log_message("Simulated connection ESTABLISHED.")
            else: 
                self.connection_status_label.config(text="Status: Simulation Mode", foreground="blue")
                self.connect_button.config(text="Simulate Connect")
                self.start_collection_button.config(state=tk.DISABLED)
                self._log_message("Simulated connection CLOSED.")
            return

        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.connection_status_label.config(text="Status: Disconnected", foreground="red")
            self.connect_button.config(text="Connect")
            self.start_collection_button.config(state=tk.DISABLED)
            self._log_message("Disconnected from device.")
        else:
            selected_port = self.com_port_combobox.get()
            if not selected_port:
                messagebox.showerror("Error", "No COM port selected.")
                return
            try:
                self.serial_connection = serial.Serial(selected_port, 115200, timeout=1)
                self.connection_status_label.config(text=f"Status: Connected to {selected_port}", foreground="green")
                self.connect_button.config(text="Disconnect")
                self.start_collection_button.config(state=tk.NORMAL)
                self._log_message(f"Connected to device on {selected_port}.")
            except serial.SerialException as e:
                messagebox.showerror("Connection Error", f"Failed to connect: {e}")
                self._log_message(f"Error connecting: {e}")
    
    def _validate_inputs(self):
        self.subject_id = self.subject_id_entry.get().strip()
        self.sample_num = self.sample_num_entry.get().strip()
        self.gender = self.gender_combobox.get()
        self.age = self.age_entry.get().strip()
        self.height = self.height_entry.get().strip()
        self.weight = self.weight_entry.get().strip()
        self.glucose = self.glucose_entry.get().strip()
        self.duration_str = self.duration_entry.get().strip()

        if not all([self.subject_id, self.sample_num, self.gender, self.age, self.height, self.weight, self.glucose, self.duration_str]):
            messagebox.showerror("Input Error", "All participant information fields and duration must be filled.")
            return False
        try:
            self.age = int(self.age) 
            self.height = float(self.height)
            self.weight = float(self.weight)
            self.glucose = float(self.glucose) 
            self.duration = int(self.duration_str)
            if self.duration <= 0:
                raise ValueError("Duration must be positive.")
            int(self.sample_num) 
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric input for a field or duration: {e}")
            return False
        return True

    def start_data_collection_and_save(self):
        if not self._validate_inputs():
            return
        
        is_sim_ready_to_collect = self.simulation_mode.get() and "Simulate Disconnect" == self.connect_button.cget("text")
        is_real_ready_to_collect = (not self.simulation_mode.get()) and (self.serial_connection and self.serial_connection.is_open)

        if not (is_sim_ready_to_collect or is_real_ready_to_collect):
            messagebox.showerror("Error", "Device not connected (or simulation not 'connected'). Please connect first.")
            return

        if self.is_collecting:
            messagebox.showwarning("Busy", "Data collection already in progress.")
            return

        self.is_collecting = True
        self.start_collection_button.config(state=tk.DISABLED)
        self.clear_fields_button.config(state=tk.DISABLED) 
        self.load_history_button.config(state=tk.DISABLED)

        self.collected_ppg_data = []
        self._log_message(f"Starting data collection for Subject ID: {self.subject_id}, Sample: {self.sample_num} for {self.duration} seconds.")
        
        collection_thread = threading.Thread(target=self._collect_ppg_data_thread, daemon=True) 
        collection_thread.start()

    def _collect_ppg_data_thread(self):
        if self.simulation_mode.get():
            self._log_message("Starting SIMULATED data generation...")
            self.collected_ppg_data = [] 
            self.root.after(0, lambda: self._log_message("Device acknowledged start. (SIMULATED)"))
            
            sim_start_time = time.time()
            target_sample_interval = 1.0 / 100 
            esp_timestamp_ms_sim = int(time.time() * 1000) 

            for i in range(self.duration * 100): 
                if not self.is_collecting: break 
                
                t_sim = i * target_sample_interval
                ppg1 = int(2048 + 500 * math.sin(2 * math.pi * 1.2 * t_sim + 0) + random.randint(-50, 50))
                ppg2 = int(2048 + 450 * math.sin(2 * math.pi * 1.1 * t_sim + 0.5) + random.randint(-50, 50))
                ppg3 = int(2048 + 550 * math.sin(2 * math.pi * 1.3 * t_sim + 1.0) + random.randint(-50, 50))
                
                ppg1 = max(0, min(4095, ppg1)); ppg2 = max(0, min(4095, ppg2)); ppg3 = max(0, min(4095, ppg3))

                esp_timestamp_ms_sim += int(target_sample_interval * 1000)
                self.collected_ppg_data.append([str(esp_timestamp_ms_sim), str(ppg1), str(ppg2), str(ppg3)])
                
                loop_run_time = time.time() - sim_start_time
                time_to_sleep = ((i + 1) * target_sample_interval) - loop_run_time
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

                if (i+1) % 100 == 0 :
                     self.root.after(0, lambda i_copy=i: self._log_message(f"Simulated {i_copy+1} data points..."))

            self._log_message(f"Finished SIMULATING {len(self.collected_ppg_data)} PPG data points.")
            self.root.after(0, lambda: self._log_message("Device signaled completion. (SIMULATED)"))
        else: 
            try:
                start_command = f"S,{self.duration}\n"
                if self.serial_connection and self.serial_connection.is_open:
                    self.serial_connection.write(start_command.encode())
                    self._log_message(f"Sent start command: {start_command.strip()}")
                else:
                    self._log_message("ERROR: Serial not connected.")
                    self.is_collecting = False 
                    self.root.after(0, self._on_collection_finish)
                    return

                collection_deadline = time.time() + self.duration + 3 
                
                while time.time() < collection_deadline and self.is_collecting:
                    if self.serial_connection.in_waiting > 0:
                        try:
                            line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                            if line == "ACK_START":
                                self.root.after(0, lambda: self._log_message("Device acknowledged start."))
                                continue
                            if line == "ACK_DONE":
                                self.root.after(0, lambda: self._log_message("Device signaled completion."))
                                self.is_collecting = False 
                                break 
                            if line.startswith("ERR_"):
                                self.root.after(0, lambda lc=line: self._log_message(f"Device Error: {lc}"))
                                continue
                            if line: 
                                parts = line.split(',')
                                if len(parts) == 4: 
                                    self.collected_ppg_data.append(parts)
                        except Exception as e_read:
                            self.root.after(0, lambda ec=e_read: self._log_message(f"Error reading line: {ec}"))
                    time.sleep(0.001) 
                
                if self.is_collecting: 
                    self._log_message("Collection timed out from Python side.")
                    self.is_collecting = False 
                self._log_message(f"Finished collecting {len(self.collected_ppg_data)} PPG data points.")
            except serial.SerialException as e_serial:
                self._log_message(f"Serial error: {e_serial}")
                self.root.after(0, lambda es=e_serial: messagebox.showerror("Serial Error", f"{es}"))
            except Exception as e_generic:
                self._log_message(f"Collection error: {e_generic}")
                self.root.after(0, lambda eg=e_generic: messagebox.showerror("Error", f"{eg}"))
        
        self.is_collecting = False 
        self.root.after(0, self._on_collection_finish)

    def _on_collection_finish(self):

        self._play_alert_sound()

        is_sim_mode = self.simulation_mode.get()
        is_sim_effectively_connected = is_sim_mode and "Simulate Disconnect" == self.connect_button.cget("text")
        is_real_connected = (not is_sim_mode) and (self.serial_connection and self.serial_connection.is_open)

        self.start_collection_button.config(state=tk.NORMAL if (is_sim_effectively_connected or is_real_connected) else tk.DISABLED)
        self.clear_fields_button.config(state=tk.NORMAL)
        self.load_history_button.config(state=tk.NORMAL)
        
        if hasattr(self, 'collected_ppg_data') and self.collected_ppg_data: 
            self._save_new_entry_data() 
        elif not self.is_collecting : 
            self._log_message("No new PPG data collected/simulated to save for a new entry.")

    def _save_new_entry_data(self):
        """Saves a NEW participant info entry and NEW collected PPG data."""
        
        # Ensure instance variables from _validate_inputs are up-to-date
        # self.subject_id, self.sample_num, self.gender, self.age, 
        # self.height, self.weight, self.glucose should have been set in _validate_inputs

        label_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ID': self.subject_id, 
            'Sample_Num': self.sample_num,    # Matches self.history_cols_csv
            'Gender': self.gender, 
            'Age': self.age,
            'Height_cm': self.height,       # Matches self.history_cols_csv
            'Weight_kg': self.weight,       # Matches self.history_cols_csv
            'Glucose_mgdL': self.glucose    # Matches self.history_cols_csv
        }
        
        # Use self.history_cols_csv for the header, as this defines the CSV structure
        header = self.history_cols_csv 
        
        file_already_exists = os.path.isfile(self.labels_file_path)
        # Check if file is empty only if it exists
        is_empty = os.path.getsize(self.labels_file_path) == 0 if file_already_exists else True
        
        try:
            # Open in append mode ('a'). If file doesn't exist, 'a' creates it.
            # If new file or empty file, write header. Otherwise, just append row.
            with open(self.labels_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not file_already_exists or is_empty: 
                    writer.writeheader()
                writer.writerow(label_entry)
            self._log_message(f"New participant info saved to {self.labels_file_path}")
        except IOError as e:
            self._log_message(f"Error saving labels: {e}")
            messagebox.showerror("File Error", f"Could not save labels: {e}")
            return # Stop if labels can't be saved

        # Save Raw PPG Data (this part seems fine)
        ppg_filename = f"{self.subject_id}_{self.sample_num}_ppg.csv"
        ppg_filepath = os.path.join(self.raw_data_dir, ppg_filename)
        try:
            with open(ppg_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Header for the PPG data file
                writer.writerow(['timestamp_ms_device', 'ppg_finger1', 'ppg_finger2', 'ppg_finger3']) 
                writer.writerows(self.collected_ppg_data)
            self._log_message(f"PPG data saved to {ppg_filepath}")
            messagebox.showinfo("Success", f"Data saved for Subject {self.subject_id}, Sample {self.sample_num}.")
        except IOError as e:
            self._log_message(f"Error saving PPG data: {e}")
            messagebox.showerror("File Error", f"Could not save PPG data: {e}")
            # Note: Label data was already saved at this point.
        
        self._load_history_data() # Refresh history view to show the new entry
        
        # Auto-increment sample number for the next potential entry for the same subject
        try: 
            current_sample_num = int(self.sample_num_entry.get())
            self.sample_num_entry.delete(0, tk.END)
            self.sample_num_entry.insert(0, str(current_sample_num + 1))
        except ValueError: 
            # If sample_num_entry wasn't a valid int, just leave it.
            # Or set it to a default like "1" if preferred.
            pass 
        
        # Clear input fields for the next entry, but keep Subject ID if doing multiple samples
        self._clear_input_fields(clear_id_too=False)


    # Method for loading history data (kept from previous version with edit/delete functionality)
    def _load_history_data(self):
        """Loads and displays data from collected_labels.csv into the Treeview."""
        for i in self.history_tree.get_children():
            self.history_tree.delete(i)
        
        if not os.path.isfile(self.labels_file_path) or os.path.getsize(self.labels_file_path) == 0:
            self._log_message("Labels history file is empty or not found.")
            self.labels_df = pd.DataFrame(columns=self.history_cols_csv) # Use CSV cols for DataFrame
            return

        try:
            self.labels_df = pd.read_csv(self.labels_file_path)
            # Ensure all expected CSV columns exist in DataFrame, add if missing
            for col in self.history_cols_csv:
                if col not in self.labels_df.columns:
                    self.labels_df[col] = pd.NA 
            
            self.labels_df.reset_index(drop=True, inplace=True)

            # When inserting into treeview, select data using CSV column names,
            # but the treeview itself is configured with display column names.
            # The `values` should correspond to the order of `self.history_cols_display`.
            for index, row in self.labels_df.iterrows():
                # Create a list of values from the row, in the order of self.history_cols_csv
                # This ensures data maps correctly to the display columns defined for the tree.
                display_values = [row.get(csv_col, '') for csv_col in self.history_cols_csv]
                self.history_tree.insert("", tk.END, iid=index, values=display_values)

            self._log_message("Labels history loaded/refreshed.")
        except pd.errors.EmptyDataError:
            self._log_message("Labels history file is empty (pd.errors.EmptyDataError).")
            self.labels_df = pd.DataFrame(columns=self.history_cols_csv)
        except Exception as e:
            self._log_message(f"Error loading history: {e}")
            messagebox.showerror("History Error", f"Could not load labels history: {e}")
            self.labels_df = pd.DataFrame(columns=self.history_cols_csv)


if __name__ == '__main__':
    # 1. Create the ThemedTk instance FIRST and make it the root.
    #    This will be the *only* Tk root window.
    main_window = ThemedTk(theme="arc")  # Apply your desired theme here
                                         # This 'main_window' IS the root.

    # 2. Initialize your application class, passing this root window.
    app = PPGDataCollectorApp(main_window, initial_geometry="650x900") # Pass geometry

    # 3. Set up the closing protocol on this same root window.
    def on_closing():
        if hasattr(app, 'serial_connection') and app.serial_connection and app.serial_connection.is_open:
            app.serial_connection.close()
            # Use print if log_text might be gone or if app instance is not reliably accessible
            print("Serial port closed on application exit.")
        main_window.destroy()

    main_window.protocol("WM_DELETE_WINDOW", on_closing)

    # 4. Start the Tkinter main loop on this root window.
    main_window.mainloop()