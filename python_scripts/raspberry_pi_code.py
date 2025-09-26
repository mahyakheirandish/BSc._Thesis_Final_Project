"""
Robotic Object Detection Raspberry Pi Application with LCD Display
Author: Mahya Kheirandish
Institution: Urmia University - Electrical-Computer Engineering Department
Year: 2025

This application provides a Tkinter-based GUI optimized for Raspberry Pi with
hardware integration including I2C LCD display for detected object information.
Features multi-threaded video processing and object filtering capabilities.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import threading

# HARDWARE INTEGRATION: LCD Display Support
try:
    import RPi.GPIO as GPIO
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
    print("LCD libraries loaded successfully")
except ImportError:
    print("RPi.GPIO or RPLCD not available. Running without LCD.")
    LCD_AVAILABLE = False

# YOLO MODEL INITIALIZATION
# Using nano model for optimal performance on Raspberry Pi
model_detection = YOLO('yolov8n.pt')

# CAMERA SETUP: Optimized for Raspberry Pi performance
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend for better Pi compatibility
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Reduced resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)             # Lower FPS for stable processing
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Minimize buffer for real-time processing

# LCD INITIALIZATION: 16x2 I2C LCD Display
if LCD_AVAILABLE:
    try:
        # Initialize LCD with I2C address 0x27 (common default)
        lcd = CharLCD('PCF8574', 0x27, port=1, cols=16, rows=2, dotsize=8)
        lcd.clear()
        lcd.write_string("Starting..")
        time.sleep(2)
        lcd.clear()
        print("LCD initialized successfully")
    except Exception as e:
        print(f"LCD initialization failed: {e}")
        LCD_AVAILABLE = False

def display_on_lcd(objects, object_counts):
    """
    Display detected objects and their counts on the LCD screen.
    
    Args:
        objects (list): List of detected object names
        object_counts (dict): Dictionary mapping object names to their counts
    
    LCD Display Format:
        Line 1: First object with count
        Line 2: Second object with count OR total count
    """
    if not LCD_AVAILABLE:
        return
    
    try:
        lcd.clear()
        if not objects:
            # No objects detected
            lcd.write_string("No objects\ndetected")
        else:
            # Display up to 2 objects (16x2 LCD limitation)
            if len(objects) == 1:
                # Single object detected
                obj_name = objects[0]
                count = object_counts[obj_name]
                line1 = f"{obj_name}({count})"[:16]  # Truncate to LCD width
                line2 = ""
            else:
                # Multiple objects detected
                obj1 = objects[0]
                count1 = object_counts[obj1]
                line1 = f"{obj1}({count1})"[:16]
                
                if len(objects) > 1:
                    obj2 = objects[1]
                    count2 = object_counts[obj2]
                    line2 = f"{obj2}({count2})"[:16]
                else:
                    # Show total count if many objects
                    total_count = sum(object_counts.values())
                    line2 = f"Total: {total_count}"[:16]
            
            lcd.write_string(f"{line1}\n{line2}")
    except Exception as e:
        print(f"LCD display error: {e}")

# OBJECT FILTERING VARIABLES
target_objects = ["ALL"]     # Currently active object filter
pending_objects = ["ALL"]    # Objects selected but not yet confirmed

class DetectionApp:
    """
    Main application class handling GUI, video processing, and object detection.
    Optimized for Raspberry Pi hardware with threading for smooth performance.
    """
    
    def __init__(self, root):
        """Initialize the main application window and components."""
        self.root = root
        self.root.title("YOLOv8 Object Detection - Mahya Kheirandish")
        self.root.geometry("1200x800")
        
        # MAIN LAYOUT: Split into video display and control panel
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT SIDE: Video Display Area
        video_frame = tk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display label with black background
        self.image_label = tk.Label(video_frame, bg='black')
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT SIDE: Object Selection Controls
        selection_frame = tk.Frame(main_frame, width=250)
        selection_frame.pack(side=tk.RIGHT, fill=tk.Y)
        selection_frame.pack_propagate(False)  # Maintain fixed width
        
        # Control panel title
        tk.Label(selection_frame, text="Select Objects to Detect:", 
                font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # OBJECT SELECTION LIST with scrollbar
        list_frame = tk.Frame(selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox for object selection (multiple selection enabled)
        self.object_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, 
                                        yscrollcommand=scrollbar.set)
        self.object_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.object_listbox.yview)
        
        # Populate listbox with YOLO class names
        self.populate_object_list()
        
        # Bind selection events
        self.object_listbox.bind('<<ListboxSelect>>', self.update_pending_objects)
        
        # CONTROL BUTTONS
        # Confirm selection button (green highlight)
        self.confirm_btn = tk.Button(selection_frame, text="Confirm Selection", 
                                   command=self.confirm_selection, bg='#4CAF50', fg='white', 
                                   font=('Arial', 10, 'bold'))
        self.confirm_btn.pack(fill=tk.X, pady=2)
        
        # Quick selection buttons
        self.select_all_btn = tk.Button(selection_frame, text="Select All", 
                                      command=self.select_all_objects)
        self.select_all_btn.pack(fill=tk.X, pady=2)
        
        self.clear_all_btn = tk.Button(selection_frame, text="Clear All", 
                                     command=self.clear_all_objects)
        self.clear_all_btn.pack(fill=tk.X, pady=2)
        
        # STATUS DISPLAY: Shows currently active object filter
        self.status_label = tk.Label(selection_frame, text="Current: ALL objects", 
                                   fg='blue', font=('Arial', 10, 'bold'), wraplength=230)
        self.status_label.pack(pady=10)
        
        # PERFORMANCE OPTIMIZATION VARIABLES
        self.frame_count = 0          # Frame counter for processing control
        self.last_display_time = 0    # Timer for LCD updates
        self.display_interval = 2     # LCD update interval (seconds)
        self.running = True           # Main loop control flag
        
        # START VIDEO PROCESSING: Separate thread to prevent GUI freezing
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        # Handle window close event properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def populate_object_list(self):
        """
        Populate the object selection listbox with YOLO class names.
        Includes "ALL objects" option and alphabetically sorted individual classes.
        """
        # Add "ALL" option first and select it by default
        self.object_listbox.insert(0, "ALL objects")
        self.object_listbox.select_set(0)
        
        # Add individual YOLO object classes in alphabetical order
        class_names = model_detection.names
        sorted_classes = sorted(class_names.items(), key=lambda x: x[1])
        
        for class_id, class_name in sorted_classes:
            self.object_listbox.insert(tk.END, class_name.title())

    def update_pending_objects(self, event=None):
        """
        Update pending object selection based on user's listbox selections.
        Handles logic for "ALL objects" vs specific object selections.
        
        Args:
            event: Tkinter event object (unused but required for callback)
        """
        global pending_objects
        selected_indices = self.object_listbox.curselection()
        
        if not selected_indices:
            pending_objects = ["ALL"]
            return
        
        pending_objects = []
        all_selected = False
        specific_objects = []
        
        # Process each selected item
        for index in selected_indices:
            item_text = self.object_listbox.get(index)
            if item_text == "ALL objects":
                all_selected = True
            else:
                specific_objects.append(item_text.lower())
        
        # Handle selection logic: ALL vs specific objects
        if all_selected and specific_objects:
            # If both ALL and specific objects selected, prioritize specific objects
            self.object_listbox.selection_clear(0)
            pending_objects = specific_objects
        elif all_selected:
            # Clear other selections when ALL is selected
            self.object_listbox.selection_clear(0, tk.END)
            self.object_listbox.select_set(0)
            pending_objects = ["ALL"]
        elif specific_objects:
            pending_objects = specific_objects
        else:
            pending_objects = ["ALL"]

    def confirm_selection(self):
        """
        Confirm the selected objects for detection and update the display.
        Activates the object filter for the detection system.
        """
        global target_objects
        target_objects = pending_objects.copy()
        
        # Update status display
        if target_objects[0] == "ALL":
            status_text = "Current: ALL objects"
            print("Detection confirmed: ALL objects")
        else:
            # Truncate long lists for display
            status_text = f"Current: {', '.join(target_objects[:3])}"
            if len(target_objects) > 3:
                status_text += f" (+{len(target_objects)-3} more)"
            print(f"Detection confirmed: {', '.join(target_objects)}")
        
        self.status_label.config(text=status_text)

    def select_all_objects(self):
        """Select all specific objects (excluding the ALL objects option)."""
        # Deselect "ALL objects" option
        self.object_listbox.selection_clear(0)
        
        # Select all specific object classes
        for i in range(1, self.object_listbox.size()):
            self.object_listbox.select_set(i)

    def clear_all_objects(self):
        """Clear all selections and default to ALL objects."""
        self.object_listbox.selection_clear(0, tk.END)
        self.object_listbox.select_set(0)  # Select "ALL objects"

    def video_loop(self):
        """
        Main video processing loop running in separate thread.
        Handles camera capture, YOLO inference, and display updates.
        Optimized for Raspberry Pi performance with frame skipping.
        """
        while self.running:
            try:
                # Capture frame from camera
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # PERFORMANCE OPTIMIZATION: Skip frames to reduce processing load
                self.frame_count += 1
                if self.frame_count % 3 != 0:
                    continue  # Process every 3rd frame only
                
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (320, 240))
                
                # YOLO INFERENCE: Run object detection
                results = model_detection.predict(source=small_frame, conf=0.4, 
                                                imgsz=320, verbose=False)
                
                detected_objects = []
                
                # PROCESS DETECTION RESULTS
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls)
                            label = model_detection.names[cls_id]
                            
                            # OBJECT FILTERING: Apply confirmed target object filter
                            if target_objects[0] == "ALL" or label.lower() in target_objects:
                                detected_objects.append(label)
                                
                                # DRAW BOUNDING BOXES on original frame
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # Scale coordinates back to original frame size
                                scale_x = frame.shape[1] / 320
                                scale_y = frame.shape[0] / 240
                                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                                
                                # Draw green bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # CONSOLE AND LCD OUTPUT: Update every 2 seconds
                current_time = time.time()
                unique_objects = list(set(detected_objects))  # Remove duplicates
                
                # Count occurrences of each detected object
                object_counts = {}
                for obj in detected_objects:
                    object_counts[obj] = object_counts.get(obj, 0) + 1
                
                # Periodic updates to avoid overwhelming console and LCD
                if current_time - self.last_display_time >= self.display_interval:
                    if unique_objects:
                        # CONSOLE OUTPUT: Show detected objects with counts
                        console_output = []
                        for obj in unique_objects[:5]:  # Limit to 5 objects
                            count = object_counts[obj]
                            if count > 1:
                                console_output.append(f"{obj}({count})")
                            else:
                                console_output.append(obj)
                        print(f"Detected: {', '.join(console_output)}")
                    else:
                        print("No objects detected")
                    
                    # LCD UPDATE: Display objects on hardware LCD
                    display_on_lcd(unique_objects, object_counts)
                    self.last_display_time = current_time
                
                # GUI UPDATE: Convert frame for Tkinter display
                self.update_gui_frame(frame)
                
            except Exception as e:
                print(f"Video processing error: {e}")
                continue

    def update_gui_frame(self, frame):
        """
        Update the GUI with the processed video frame.
        Converts OpenCV BGR format to Tkinter-compatible format.
        
        Args:
            frame: OpenCV BGR format frame
        """
        try:
            # Convert BGR to RGB for proper color display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage for Tkinter
            pil_image = Image.fromarray(rgb_frame)
            # Resize to fit GUI label while maintaining aspect ratio
            pil_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update GUI (must be done in main thread)
            self.root.after(0, self._update_image_label, photo)
            
        except Exception as e:
            print(f"GUI update error: {e}")

    def _update_image_label(self, photo):
        """
        Helper method to update image label in main thread.
        
        Args:
            photo: PhotoImage object for Tkinter display
        """
        try:
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Image label update error: {e}")

    def on_closing(self):
        """
        Handle application shutdown gracefully.
        Stops video thread, releases camera, and cleans up LCD display.
        """
        print("Shutting down application...")
        self.running = False  # Stop video processing loop
        
        # Release camera resources
        cap.release()
        
        # Clear LCD display with shutdown message
        if LCD_AVAILABLE:
            try:
                lcd.clear()
                lcd.write_string("Detection\nStopped")
                time.sleep(1)
                lcd.clear()
            except:
                pass
        
        # Close application
        self.root.destroy()

if __name__ == "__main__":
    # Create main Tkinter window
    root = tk.Tk()
    
    # Initialize and start the detection application
    app = DetectionApp(root)
    
    # Start the GUI event loop
    root.mainloop()
    
    print("Application terminated successfully")
