import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import serial
import serial.tools.list_ports
import threading
import time
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

class PPGLiveMonitorApp:
    """
    A standalone application for real-time monitoring of 3-channel PPG signals
    to check for signal quality before official data collection.
    """
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("PPG Live Signal Monitor")
        self.root.geometry("900x750")

        # --- State and Serial Attributes ---
        self.serial_connection = None
        self.is_monitoring = False

        # --- Plotting Attributes ---
        self.ani = None
        self.buffer_size = 1000  # ~10s @ 100Hz
        self.plot_data_1 = collections.deque(maxlen=self.buffer_size)
        self.plot_data_2 = collections.deque(maxlen=self.buffer_size)
        self.plot_data_3 = collections.deque(maxlen=self.buffer_size)
        # 
        self._setup_gui()
        self._center_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _center_window(self):
        """Centers the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_gui(self):
        """Creates and places all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Connection Control Frame ---
        conn_frame = ttk.LabelFrame(main_frame, text="Device Connection", padding="10")
        conn_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        conn_frame.columnconfigure(3, weight=1)

        ttk.Label(conn_frame, text="COM Port:").pack(side=tk.LEFT, padx=(0, 5))
        self.com_port_combobox = ttk.Combobox(conn_frame, state="readonly", width=15)
        self.com_port_combobox.pack(side=tk.LEFT, padx=5)
        
        self.refresh_button = ttk.Button(conn_frame, text="Refresh Ports", command=self.refresh_com_ports)
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.connect_button = ttk.Button(conn_frame, text="Connect & Start Monitor", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red", anchor=tk.E)
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        self.refresh_com_ports()

        # --- Plotting Frame ---
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=1, column=0, sticky="nsew")

        fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        fig.suptitle("Real-Time PPG Signals", fontsize=16)
        
        self.ax1.set_title("Finger 1 (Index)")
        self.ax1.set_ylabel("ADC Reading")
        self.line1, = self.ax1.plot([], [], color='crimson', lw=1.5)

        self.ax2.set_title("Finger 2 (Middle)")
        self.ax2.set_ylabel("ADC Reading")
        self.line2, = self.ax2.plot([], [], color='limegreen', lw=1.5)
        
        self.ax3.set_title("Finger 3 (Ring)")
        self.ax3.set_ylabel("ADC Reading")
        self.ax3.set_xlabel("Time (Samples)")
        self.line3, = self.ax3.plot([], [], color='royalblue', lw=1.5)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(True)
            ax.set_xlim(0, self.buffer_size) # Set fixed X-axis
            ax.set_ylim(0, 4096)             # Default Y-axis limit

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start animation, but it will only show data when is_monitoring is True
        # THE KEY FIX IS HERE: blit=False allows the axes to be rescaled dynamically.
        self.ani = FuncAnimation(fig, self._update_plot, blit=False, interval=50, cache_frame_data=False)
        self.canvas.draw()

    def refresh_com_ports(self):
        """Scans for available serial ports and updates the combobox."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.com_port_combobox['values'] = ports
        if ports:
            self.com_port_combobox.current(0)
        else:
            self.com_port_combobox.set('')

    def toggle_connection(self):
        """Connects or disconnects from the device and controls monitoring."""
        if self.is_monitoring:
            # --- Stop monitoring ---
            self.is_monitoring = False
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            
            self.status_label.config(text="Status: Disconnected", foreground="red")
            self.connect_button.config(text="Connect & Start Monitor")
            print("Monitoring stopped and disconnected.")
        else:
            # --- Start monitoring ---
            port = self.com_port_combobox.get()
            if not port:
                messagebox.showerror("Error", "No COM port selected.")
                return
            try:
                self.serial_connection = serial.Serial(port, 115200, timeout=1)
                time.sleep(3)  # Wait longer for ESP32 to fully initialize
                
                # Clear any pending data
                self.serial_connection.flushInput()
                self.serial_connection.flushOutput()
                
                # Try different command formats
                commands_to_try = [
                    "S,3600\n",   # 1 hour in seconds
                    "S,60\n",     # 1 minute
                    "S,10\n",     # 10 seconds
                    "S,-1\n",     # -1 might mean continuous
                    "S\n",        # Just S without duration
                ]
                
                for cmd in commands_to_try:
                    print(f"Trying command: {cmd.strip()}")
                    self.serial_connection.write(cmd.encode())
                    time.sleep(0.5)  # Wait for response
                    
                    # Check for any response
                    if self.serial_connection.in_waiting > 0:
                        response = self.serial_connection.readline().decode('utf-8').strip()
                        print(f"Response: {response}")
                        
                        if "ERR" not in response:
                            print(f"Success with command: {cmd.strip()}")
                            break
                    else:
                        print("No immediate response, continuing...")
                        break  # Assume success if no error
                
                self.is_monitoring = True
                
                # Start reading thread
                self.monitoring_thread = threading.Thread(target=self._read_serial_for_plot, daemon=True)
                self.monitoring_thread.start()

                self.status_label.config(text="Status: Monitoring...", foreground="green")
                self.connect_button.config(text="Stop & Disconnect")
                print(f"Connected to {port} and started monitoring.")

            except serial.SerialException as e:
                messagebox.showerror("Connection Error", f"Failed to connect: {e}")
                print(f"Error connecting: {e}")

    def _read_serial_for_plot(self):
        """Continuously reads serial data and appends it to the plot deques."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.flushInput()
        
        while self.is_monitoring:
            try:
                if not self.serial_connection.is_open:
                    break
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    print(f"Received: {line}")  # Debug: Print all received data
                    
                    # Skip initialization messages and info lines
                    if any(skip_word in line.lower() for skip_word in 
                          ['ets', 'rst:', 'configsip', 'clk_drv', 'mode:', 'load:', 'entry', 
                           'analog pulse', 'sensor pins', 'waiting', 'data format', 'err_']):
                        continue
                    
                    # Check if this looks like numeric data
                    if ',' in line and not any(char.isalpha() for char in line.replace(',', '')):
                        parts = line.split(',')
                        print(f"Parts: {parts}, Length: {len(parts)}")  # Debug: Print parsed parts
                        
                        # Look for numeric data only (timestamp,ppg1,ppg2,ppg3)
                        if len(parts) == 4:
                            try:
                                # Parse as timestamp,ppg1,ppg2,ppg3
                                timestamp = int(parts[0])
                                ppg1, ppg2, ppg3 = int(parts[1]), int(parts[2]), int(parts[3])
                                
                                self.plot_data_1.append(ppg1)
                                self.plot_data_2.append(ppg2)
                                self.plot_data_3.append(ppg3)
                                print(f"Added data: PPG1={ppg1}, PPG2={ppg2}, PPG3={ppg3}")  # Debug
                                
                            except ValueError as e:
                                print(f"Value error parsing: {line} - {e}")
                                continue
                        elif len(parts) == 3:
                            try:
                                # Parse as ppg1,ppg2,ppg3 (no timestamp)
                                ppg1, ppg2, ppg3 = int(parts[0]), int(parts[1]), int(parts[2])
                                
                                self.plot_data_1.append(ppg1)
                                self.plot_data_2.append(ppg2)
                                self.plot_data_3.append(ppg3)
                                print(f"Added data: PPG1={ppg1}, PPG2={ppg2}, PPG3={ppg3}")  # Debug
                                
                            except ValueError as e:
                                print(f"Value error parsing: {line} - {e}")
                                continue
                    else:
                        print(f"Non-numeric data: {line}")
            except (serial.SerialException, TypeError, UnicodeDecodeError) as e:
                print(f"Serial error: {e}")
                time.sleep(0.01)
                continue
        print("Serial reading thread finished.")

    def _update_plot(self, frame):
        """Updates plot data for the animation. Called by FuncAnimation."""
        # Add data count indicator
        data_count = len(self.plot_data_1)
        if data_count > 0:
            print(f"Plotting {data_count} data points")  # Debug: Show we have data
        
        self.line1.set_data(range(len(self.plot_data_1)), self.plot_data_1)
        self.line2.set_data(range(len(self.plot_data_2)), self.plot_data_2)
        self.line3.set_data(range(len(self.plot_data_3)), self.plot_data_3)

        # Auto-scale Y-axis
        for ax, data in zip([self.ax1, self.ax2, self.ax3], [self.plot_data_1, self.plot_data_2, self.plot_data_3]):
            if data:
                min_val, max_val = min(data), max(data)
                padding = (max_val - min_val) * 0.1
                padding = max(padding, 20) 
                ax.set_ylim(min_val - padding, max_val + padding)
        
        return self.line1, self.line2, self.line3

    def on_closing(self):
        """Handles the main window closing event cleanly."""
        self.is_monitoring = False # Stop the reading thread
        time.sleep(0.1) # Give the thread a moment to exit
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial port closed on exit.")
        self.root.destroy()

if __name__ == '__main__':
    root = ThemedTk(theme="arc")
    app = PPGLiveMonitorApp(root)
    root.mainloop()