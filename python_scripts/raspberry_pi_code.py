import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import threading

# Add LCD support
try:
    import RPi.GPIO as GPIO
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    print("RPi.GPIO or RPLCD not available. Running without LCD.")
    LCD_AVAILABLE = False

# Your existing model load and LCD initialization
model_detection = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Initialize LCD (adjust address and port as needed)
if LCD_AVAILABLE:
    try:
        lcd = CharLCD('PCF8574', 0x27, port=1, cols=16, rows=2, dotsize=8)
        lcd.clear()
        lcd.write_string("Starting..")
        time.sleep(2)
        lcd.clear()
    except Exception as e:
        print(f"LCD initialization failed: {e}")
        LCD_AVAILABLE = False

def display_on_lcd(objects, object_counts):
    """Display detected objects on LCD"""
    if not LCD_AVAILABLE:
        return
    
    try:
        lcd.clear()
        if not objects:
            lcd.write_string("No objects\ndetected")
        else:
            # Display up to 2 objects (one per line) with counts
            if len(objects) == 1:
                obj_name = objects[0]
                count = object_counts[obj_name]
                line1 = f"{obj_name}({count})"[:16]
                line2 = ""
            else:
                obj1 = objects[0]
                count1 = object_counts[obj1]
                line1 = f"{obj1}({count1})"[:16]
                
                if len(objects) > 1:
                    obj2 = objects[1]
                    count2 = object_counts[obj2]
                    line2 = f"{obj2}({count2})"[:16]
                else:
                    total_count = sum(object_counts.values())
                    line2 = f"Total: {total_count}"[:16]
            
            lcd.write_string(f"{line1}\n{line2}")
    except Exception as e:
        print(f"LCD display error: {e}")

# Initialize target object filter
target_objects = ["ALL"]  # Default to detect all objects
pending_objects = ["ALL"]  # Objects selected but not yet confirmed

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection - Mahya Kheirandish")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video display
        video_frame = tk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(video_frame, bg='black')
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Object selection
        selection_frame = tk.Frame(main_frame, width=250)
        selection_frame.pack(side=tk.RIGHT, fill=tk.Y)
        selection_frame.pack_propagate(False)
        
        tk.Label(selection_frame, text="Select Objects to Detect:", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Object list widget with scrollbar
        list_frame = tk.Frame(selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.object_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
        self.object_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.object_listbox.yview)
        
        # Populate with YOLO class names
        self.populate_object_list()
        
        # Bind selection change
        self.object_listbox.bind('<<ListboxSelect>>', self.update_pending_objects)
        
        # Buttons
        self.confirm_btn = tk.Button(selection_frame, text="Confirm Selection", 
                                   command=self.confirm_selection, bg='#4CAF50', fg='white', 
                                   font=('Arial', 10, 'bold'))
        self.confirm_btn.pack(fill=tk.X, pady=2)
        
        self.select_all_btn = tk.Button(selection_frame, text="Select All", 
                                      command=self.select_all_objects)
        self.select_all_btn.pack(fill=tk.X, pady=2)
        
        self.clear_all_btn = tk.Button(selection_frame, text="Clear All", 
                                     command=self.clear_all_objects)
        self.clear_all_btn.pack(fill=tk.X, pady=2)
        
        # Status label
        self.status_label = tk.Label(selection_frame, text="Current: ALL objects", 
                                   fg='blue', font=('Arial', 10, 'bold'), wraplength=230)
        self.status_label.pack(pady=10)
        
        self.frame_count = 0
        self.last_display_time = 0
        self.display_interval = 2  # seconds
        self.running = True
        
        # Start video processing in separate thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def populate_object_list(self):
        """Populate the object list with YOLO class names"""
        # Add "ALL" option first
        self.object_listbox.insert(0, "ALL objects")
        self.object_listbox.select_set(0)  # Select by default
        
        # Add individual object classes in alphabetical order
        class_names = model_detection.names
        sorted_classes = sorted(class_names.items(), key=lambda x: x[1])
        
        for class_id, class_name in sorted_classes:
            self.object_listbox.insert(tk.END, class_name.title())

    def update_pending_objects(self, event=None):
        """Update pending objects based on user selection (not yet confirmed)"""
        global pending_objects
        selected_indices = self.object_listbox.curselection()
        
        if not selected_indices:
            pending_objects = ["ALL"]
            return
            
        pending_objects = []
        all_selected = False
        specific_objects = []
        
        for index in selected_indices:
            item_text = self.object_listbox.get(index)
            if item_text == "ALL objects":
                all_selected = True
            else:
                specific_objects.append(item_text.lower())
        
        # If "ALL" is selected along with specific objects, prioritize specific objects
        if all_selected and specific_objects:
            # Deselect "ALL objects" item
            self.object_listbox.selection_clear(0)
            pending_objects = specific_objects
        elif all_selected:
            # Clear all other selections when ALL is selected
            self.object_listbox.selection_clear(0, tk.END)
            self.object_listbox.select_set(0)
            pending_objects = ["ALL"]
        elif specific_objects:
            pending_objects = specific_objects
        else:
            pending_objects = ["ALL"]

    def confirm_selection(self):
        """Confirm the selected objects for detection"""
        global target_objects
        target_objects = pending_objects.copy()
        
        if target_objects[0] == "ALL":
            status_text = "Current: ALL objects"
            print("Detection confirmed: ALL objects")
        else:
            status_text = f"Current: {', '.join(target_objects[:3])}"
            if len(target_objects) > 3:
                status_text += f" (+{len(target_objects)-3} more)"
            print(f
            if len(target_objects) > 3:
                status_text += f" (+{len(target_objects)-3} more)"
            print(f"Detection confirmed: {', '.join(target_objects)}")
        
        self.status_label.setText(status_text)

    def select_all_objects(self):
        """Select all specific objects (not including ALL objects option)"""
        # First, deselect "ALL objects"
        all_item = self.object_list.item(0)
        all_item.setSelected(False)
        
        # Then select all specific object classes
        for i in range(1, self.object_list.count()):
            self.object_list.item(i).setSelected(True)

    def clear_all_objects(self):
        """Clear all selections and select ALL by default"""
        self.object_list.clearSelection()
        self.object_list.item(0).setSelected(True)  # Select "ALL objects"

    def update_frame(self):
        ret, frame = cap.read()
        if not ret:
            return
        
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            # Skip frames to reduce load
            return
        
        # Resize frame for processing
        small_frame = cv2.resize(frame, (320, 240))
        
        results = model_detection.predict(source=small_frame, conf=0.4, imgsz=320, verbose=False)
        
        detected_objects = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    label = model_detection.names[cls_id]
                    
                    # Filter based on confirmed target objects
                    if target_objects[0] == "ALL" or label.lower() in target_objects:
                        detected_objects.append(label)
                        # Draw bounding box on original frame scaled appropriately
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Scale back to original frame size (if different)
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 240
                        x1 = int(x1 * scale_x)
                        x2 = int(x2 * scale_x)
                        y1 = int(y1 * scale_y)
                        y2 = int(y2 * scale_y)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,255,0), 1)

        # Show detected labels on console every 2 seconds
        current_time = time.time()
        unique_objects = list(set(detected_objects))
        
        # Count occurrences of each object
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        if current_time - self.last_display_time >= self.display_interval:
            if unique_objects:
                # Display with counts for console
                console_output = []
                for obj in unique_objects[:5]:
                    count = object_counts[obj]
                    if count > 1:
                        console_output.append(f"{obj}({count})")
                    else:
                        console_output.append(obj)
                print(f"Detected: {', '.join(console_output)}")
            else:
                print("No objects detected")
            
            # Update LCD display with counts
            display_on_lcd(unique_objects, object_counts)
            self.last_display_time = current_time
        
        # Convert BGR OpenCV image to RGB for Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit the full label size
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        cap.release()
        if LCD_AVAILABLE:
            try:
                lcd.clear()
                lcd.write_string("Detection\nStopped")
                time.sleep(1)
                lcd.clear()
            except:
                pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())
