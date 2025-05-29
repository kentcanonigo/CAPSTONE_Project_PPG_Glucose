import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import time
import csv
import os
from datetime import datetime

class PPGDataCollectorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("PPG Data Collector")
        # Try a more compact initial size
        self.root.geometry("500x630") # ADJUSTED from 600x750

        # ... (rest of __init__ remains the same) ...
        self.serial_connection = None
        self.is_collecting = False
        self.collected_ppg_data = []

        self.base_save_path = "Collected_Data"
        self.labels_dir = os.path.join(self.base_save_path, "Labels")
        self.raw_data_dir = os.path.join(self.base_save_path, "RawData")
        self.labels_file_path = os.path.join(self.labels_dir, "collected_labels.csv")

        self._create_directories()
        self._setup_gui()

    def _create_directories(self):
        """Creates necessary directories for saving data if they don't exist."""
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def _setup_gui(self):
        """Sets up the main GUI layout (more compact version)."""
        main_frame = ttk.Frame(self.root, padding="5 5 5 5") # Reduced padding
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Log Section (MOVED EARLIER for self.log_text availability) ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5") # Reduced padding
        log_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10,5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=8, width=60, state=tk.DISABLED) # Reduced height
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        # End of Log Section creation

        # --- Participant Info Section (2 columns of fields) ---
        participant_frame = ttk.LabelFrame(main_frame, text="Participant Information", padding="5") # Reduced padding
        participant_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        # Configure 4 columns for label-entry pairs
        participant_frame.columnconfigure(1, weight=1)
        participant_frame.columnconfigure(3, weight=1)


        ttk.Label(participant_frame, text="Subj. ID:").grid(row=0, column=0, sticky=tk.W, padx=(0,2), pady=2)
        self.subject_id_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.subject_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(participant_frame, text="Sample #:").grid(row=0, column=2, sticky=tk.W, padx=(5,2), pady=2)
        self.sample_num_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.sample_num_entry.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=2)
        self.sample_num_entry.insert(0, "1")

        ttk.Label(participant_frame, text="Gender:").grid(row=1, column=0, sticky=tk.W, padx=(0,2), pady=2)
        self.gender_combobox = ttk.Combobox(participant_frame, values=["Male", "Female", "Other"], state="readonly", width=12) # Reduced width
        self.gender_combobox.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.gender_combobox.current(0)

        ttk.Label(participant_frame, text="Age:").grid(row=1, column=2, sticky=tk.W, padx=(5,2), pady=2)
        self.age_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.age_entry.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(participant_frame, text="Height (cm):").grid(row=2, column=0, sticky=tk.W, padx=(0,2), pady=2)
        self.height_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.height_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(participant_frame, text="Weight (kg):").grid(row=2, column=2, sticky=tk.W, padx=(5,2), pady=2)
        self.weight_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.weight_entry.grid(row=2, column=3, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(participant_frame, text="Glucose (mg/dL):").grid(row=3, column=0, sticky=tk.W, padx=(0,2), pady=2)
        self.glucose_entry = ttk.Entry(participant_frame, width=15) # Reduced width
        self.glucose_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)


        # --- Serial Connection Section ---
        serial_frame = ttk.LabelFrame(main_frame, text="Serial Connection (ESP32)", padding="5") # Reduced padding
        serial_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        serial_frame.columnconfigure(1, weight=1) # Make combobox expand
        # serial_frame.columnconfigure(2, weight=0)
        # serial_frame.columnconfigure(3, weight=0)


        ttk.Label(serial_frame, text="COM Port:").grid(row=0, column=0, sticky=tk.W, padx=(0,2), pady=2)
        self.com_port_combobox = ttk.Combobox(serial_frame, state="readonly", width=12) # Reduced width
        self.com_port_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        self.refresh_button = ttk.Button(serial_frame, text="Refresh", command=self.refresh_com_ports) # Shortened text
        self.refresh_button.grid(row=0, column=2, sticky=tk.E, pady=2, padx=(5,0))
        self.refresh_com_ports()

        self.connect_button = ttk.Button(serial_frame, text="Connect", command=self.toggle_serial_connection, width=10) # Fixed width
        self.connect_button.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
        
        self.connection_status_label = ttk.Label(serial_frame, text="Status: Disconnected", foreground="red", width=25, anchor=tk.E) # Fixed width and anchor
        self.connection_status_label.grid(row=1, column=2, columnspan=2, sticky=tk.E, pady=(5,2), padx=(5,0))


        # --- Data Collection Section ---
        collection_frame = ttk.LabelFrame(main_frame, text="Data Collection", padding="5") # Reduced padding
        collection_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        collection_frame.columnconfigure(1, weight=1) # Make entry expand

        ttk.Label(collection_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W, padx=(0,2), pady=2) # Shortened text
        self.duration_entry = ttk.Entry(collection_frame, width=15) # Reduced width
        self.duration_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, columnspan=3) # Span to use available space
        self.duration_entry.insert(0, "10")

        self.start_collection_button = ttk.Button(collection_frame, text="Start Collection & Save", command=self.start_data_collection_and_save)
        self.start_collection_button.grid(row=1, column=0, columnspan=4, pady=(10,5))
        self.start_collection_button.config(state=tk.DISABLED)

        # Configure resizing behavior for main_frame content
        main_frame.columnconfigure(0, weight=1) # Allow main_frame to expand
        main_frame.rowconfigure(3, weight=1) # Allow log_frame to expand vertically

    def _log_message(self, message):
        """Appends a message to the log Text widget."""
        if hasattr(self, 'log_text') and self.log_text: # Check if log_text exists
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
            self.log_text.see(tk.END) # Scroll to the end
            self.log_text.config(state=tk.DISABLED)
            if self.root: # Check if root window still exists
                 self.root.update_idletasks() # Ensure GUI updates
        else:
            print(f"LOG (log_text not ready): {message}") # Fallback to console print

    def refresh_com_ports(self):
        """Refreshes the list of available COM ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.com_port_combobox['values'] = ports
        if ports:
            self.com_port_combobox.current(0)
        else:
            self.com_port_combobox.set('')
        self._log_message("COM ports refreshed.")

    def toggle_serial_connection(self):
        """Connects to or disconnects from the selected COM port."""
        if self.serial_connection and self.serial_connection.is_open:
            # Disconnect
            self.serial_connection.close()
            self.connection_status_label.config(text="Status: Disconnected", foreground="red")
            self.connect_button.config(text="Connect")
            self.start_collection_button.config(state=tk.DISABLED)
            self._log_message("Disconnected from ESP32.")
        else:
            # Connect
            selected_port = self.com_port_combobox.get()
            if not selected_port:
                messagebox.showerror("Error", "No COM port selected.")
                return
            try:
                # Common baud rate for ESP32, adjust if yours is different
                self.serial_connection = serial.Serial(selected_port, 115200, timeout=1)
                self.connection_status_label.config(text=f"Status: Connected to {selected_port}", foreground="green")
                self.connect_button.config(text="Disconnect")
                self.start_collection_button.config(state=tk.NORMAL)
                self._log_message(f"Connected to ESP32 on {selected_port}.")
            except serial.SerialException as e:
                messagebox.showerror("Connection Error", f"Failed to connect to {selected_port}:\n{e}")
                self._log_message(f"Error connecting to {selected_port}: {e}")


    def _validate_inputs(self):
        """Validates participant info and collection parameters."""
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
            self.glucose = float(self.glucose) # Or int if you prefer
            self.duration = int(self.duration_str)
            if self.duration <= 0:
                raise ValueError("Duration must be positive.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric input for Age, Height, Weight, Glucose, or Duration: {e}")
            return False
        return True

    def start_data_collection_and_save(self):
        """Handles the full data collection and saving process."""
        if not self._validate_inputs():
            return
        if not self.serial_connection or not self.serial_connection.is_open:
            messagebox.showerror("Error", "ESP32 not connected.")
            return
        if self.is_collecting:
            messagebox.showwarning("Busy", "Data collection already in progress.")
            return

        self.is_collecting = True
        self.start_collection_button.config(state=tk.DISABLED)
        self.collected_ppg_data = [] # Clear previous data
        self._log_message(f"Starting data collection for Subject ID: {self.subject_id}, Sample: {self.sample_num} for {self.duration} seconds.")

        # Run data collection in a separate thread to keep GUI responsive
        collection_thread = threading.Thread(target=self._collect_ppg_data_thread)
        collection_thread.start()

    def _collect_ppg_data_thread(self):
        """Thread function to collect PPG data from ESP32."""
        try:
            # Send start command to ESP32 (e.g., "S,duration_in_seconds\n")
            # Adjust this command based on your ESP32 firmware
            start_command = f"S,{self.duration}\n"
            self.serial_connection.write(start_command.encode())
            self._log_message(f"Sent start command to ESP32: {start_command.strip()}")
            
            start_time = time.time()
            while (time.time() - start_time) < self.duration:
                if self.serial_connection.in_waiting > 0:
                    try:
                        line = self.serial_connection.readline().decode('utf-8').strip()
                        if line:
                            parts = line.split(',')
                            if len(parts) == 4: 
                                self.collected_ppg_data.append(parts)
                            # Optional: Log lines that don't match expected format for debugging ESP32 output
                            # else:
                            #     self._log_message(f"Debug: Received non-data line: {line}")
                    except UnicodeDecodeError:
                        self._log_message("Warning: Unicode decode error on serial line.")
                    except Exception as e:
                        self._log_message(f"Error reading line: {e}")
                time.sleep(0.001) 

            self._log_message(f"Finished collecting {len(self.collected_ppg_data)} PPG data points.")

        except serial.SerialException as e:
            self._log_message(f"Serial error during collection: {e}")
            # Schedule messagebox to be shown in main thread
            self.root.after(0, lambda: messagebox.showerror("Serial Error", f"Serial communication error: {e}"))
        except Exception as e:
            self._log_message(f"Error during collection: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            self.is_collecting = False
            self.root.after(0, self._on_collection_finish)

    def _on_collection_finish(self):
        """Called after data collection thread finishes, to update GUI and save."""
        self.start_collection_button.config(state=tk.NORMAL if self.serial_connection and self.serial_connection.is_open else tk.DISABLED)
        if self.collected_ppg_data:
            self._save_data()
        else:
            self._log_message("No PPG data collected to save.")
            messagebox.showwarning("No Data", "No PPG data was collected.")

    def _save_data(self):
        """Saves participant info and collected PPG data to CSV files."""
        label_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ID': self.subject_id,
            'Sample_Num': self.sample_num,
            'Gender': self.gender,
            'Age': self.age,
            'Height_cm': self.height,
            'Weight_kg': self.weight,
            'Glucose_mgdL': self.glucose 
        }
        file_exists = os.path.isfile(self.labels_file_path)
        try:
            with open(self.labels_file_path, 'a', newline='') as csvfile:
                fieldnames = ['Timestamp', 'ID', 'Sample_Num', 'Gender', 'Age', 'Height_cm', 'Weight_kg', 'Glucose_mgdL']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(self.labels_file_path) == 0: # Check if file is new or empty
                    writer.writeheader()
                writer.writerow(label_entry)
            self._log_message(f"Participant info saved to {self.labels_file_path}")
        except IOError as e:
            self._log_message(f"Error saving labels: {e}")
            messagebox.showerror("File Error", f"Could not save labels data: {e}")
            return

        ppg_filename = f"{self.subject_id}_{self.sample_num}_ppg.csv"
        ppg_filepath = os.path.join(self.raw_data_dir, ppg_filename)
        try:
            with open(ppg_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp_ms_esp32', 'ppg_finger1', 'ppg_finger2', 'ppg_finger3'])
                writer.writerows(self.collected_ppg_data)
            self._log_message(f"PPG data saved to {ppg_filepath}")
            messagebox.showinfo("Success", f"Data collection complete and saved for Subject {self.subject_id}, Sample {self.sample_num}.")
        except IOError as e:
            self._log_message(f"Error saving PPG data: {e}")
            messagebox.showerror("File Error", f"Could not save PPG data: {e}")

        try:
            current_sample_num = int(self.sample_num_entry.get())
            self.sample_num_entry.delete(0, tk.END)
            self.sample_num_entry.insert(0, str(current_sample_num + 1))
        except ValueError:
            pass


if __name__ == '__main__':
    main_window = tk.Tk()
    app = PPGDataCollectorApp(main_window)
    main_window.mainloop()
