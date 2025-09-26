"""
Robotic Object Detection and Segmentation Desktop Application
Author: Mahya Kheirandish
Institution: Urmia University - Electrical-Computer Engineering Department
Year: 2025

This application provides a PyQt5-based GUI for real-time object detection and segmentation
using YOLOv8 models. It supports both detection and segmentation modes with object filtering.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, \
    QListWidget, QScrollArea
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Initialize YOLOv8 models
# Detection model for object detection with bounding boxes
model_detection = YOLO('yolov8n.pt')  # YOLOv8 nano model for detection
# Segmentation model for pixel-level object segmentation
model_segmentation = YOLO('yolov8n-seg.pt')  # YOLOv8 nano segmentation model

# Initialize video capture (camera index 1, change to 0 if needed)
cap = cv2.VideoCapture(1)

# Global control variables
detection_enabled = False      # Flag to enable/disable detection mode
segmentation_enabled = False   # Flag to enable/disable segmentation mode
target_object = ""            # Target object filter (empty = detect all)


class DetectionThread(QThread):
    """
    Background thread for continuous video processing and object detection/segmentation.
    This prevents GUI freezing during intensive video processing operations.
    """
    # Signal to emit processed frames to the main GUI thread
    frame_updated = pyqtSignal(np.ndarray)

    def run(self):
        """
        Main thread execution loop for video processing.
        Handles both detection and segmentation modes with object filtering.
        """
        global detection_enabled, segmentation_enabled, target_object
        
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                break  # Exit if camera disconnected

            # DETECTION MODE: Draw bounding boxes around detected objects
            if detection_enabled and not segmentation_enabled:
                # Run YOLOv8 detection inference
                results = model_detection.predict(source=frame, conf=0.5, stream=True)

                detected_objects = []
                annotated_frame = frame.copy()

                for result in results:
                    if target_object:
                        # FILTERED DETECTION: Show only specified target object
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                object_name = model_detection.names[int(box.cls)]
                                # Check if detected object matches target filter
                                if target_object.lower() in object_name.lower():
                                    detected_objects.append(object_name)
                                    # Manually draw bounding box and label
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, object_name, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # SHOW ALL OBJECTS: Use built-in YOLO plotting
                        annotated_frame = result.plot()
                        if result.boxes is not None:
                            detected_objects.extend([model_detection.names[int(box.cls)] for box in result.boxes])

                # Display detected object names on frame
                y_offset = 20
                for obj in set(detected_objects):  # Remove duplicates
                    cv2.putText(annotated_frame, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 20

            # SEGMENTATION MODE: Draw pixel-level masks on detected objects
            elif segmentation_enabled and not detection_enabled:
                # Run YOLOv8 segmentation inference
                results = model_segmentation.predict(source=frame, conf=0.5, stream=True)

                annotated_frame = frame.copy()
                for result in results:
                    # Check if segmentation masks are available
                    if hasattr(result, 'masks') and result.masks is not None:
                        if target_object:
                            # FILTERED SEGMENTATION: Show only specified target object
                            if result.boxes is not None and len(result.boxes) > 0:
                                for i, box in enumerate(result.boxes):
                                    object_name = model_segmentation.names[int(box.cls)]
                                    if target_object.lower() in object_name.lower():
                                        # Draw filtered segmentation mask manually
                                        if i < len(result.masks.data):
                                            mask = result.masks.data[i].cpu().numpy()
                                            colored_mask = np.zeros_like(annotated_frame)
                                            colored_mask[mask > 0.5] = [0, 255, 0]  # Green overlay
                                            annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.3, 0)

                                        # Draw bounding box and label
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, object_name, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # SHOW ALL SEGMENTATIONS: Use built-in YOLO plotting
                            annotated_frame = result.plot()
                    else:
                        # No masks available, show original frame
                        annotated_frame = frame.copy()
            else:
                # IDLE MODE: Show original camera feed
                annotated_frame = frame.copy()

            # Emit processed frame to GUI thread
            self.frame_updated.emit(annotated_frame)

        # Cleanup when thread ends
        cap.release()
        cv2.destroyAllWindows()


class MainWindow(QWidget):
    """
    Main GUI window class handling user interface and interactions.
    Provides controls for detection/segmentation modes and object filtering.
    """
    
    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        self.setWindowTitle("Object Detection and Segmentation")
        self.resize(1200, 700)  # Set window size

        # CONTROL BUTTONS
        # Toggle button for detection mode
        self.toggle_detection_button = QPushButton("Enable Detection")
        self.toggle_detection_button.clicked.connect(self.toggle_detection)

        # Toggle button for segmentation mode
        self.toggle_segmentation_button = QPushButton("Enable Segmentation")
        self.toggle_segmentation_button.clicked.connect(self.toggle_segmentation)

        # OBJECT FILTERING CONTROLS
        # Text input for specifying target object
        self.object_input = QLineEdit()
        self.object_input.setPlaceholderText("Enter object name (e.g., person, car, chair)")
        # Confirm button to apply object filter
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_target_object)

        # VIDEO DISPLAY
        # Label to display video feed
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # UNIVERSITY BRANDING
        # Logo display
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(100, 80)
        try:
            # Load and scale university logo
            logo_pixmap = QPixmap("urmia_logo.png")
            scaled_logo = logo_pixmap.scaled(80, 80, aspectRatioMode=1)
            self.logo_label.setPixmap(scaled_logo)
        except:
            # Fallback if logo file not found
            self.logo_label.setText("Logo")
            self.logo_label.setStyleSheet("border: 1px solid gray; text-align: center;")

        # Creator information display
        self.creator_label = QLabel("Creator: Mahya Kheirandish")
        self.creator_label.setStyleSheet("font-size: 12px; color: black; text-align: left;")
        self.creator_label.setFixedSize(200, 20)

        # OBJECT SELECTION PANEL
        # List widget showing available YOLO object classes
        self.objects_list = QListWidget()
        self.objects_list.setMaximumWidth(200)
        self.objects_list.setMinimumWidth(200)

        # Populate list with all available YOLO object classes
        available_objects = list(model_detection.names.values())
        for obj in sorted(available_objects):
            self.objects_list.addItem(obj)

        # Connect list selection to input field
        self.objects_list.itemClicked.connect(self.on_object_selected)

        # LAYOUT ORGANIZATION
        # Input controls layout
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Target Object:"))
        input_layout.addWidget(self.object_input)
        input_layout.addWidget(self.confirm_button)

        # Control buttons layout
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.toggle_detection_button)
        button_layout.addWidget(self.toggle_segmentation_button)

        # Header with logo and creator info
        header_layout = QHBoxLayout()
        logo_info_layout = QVBoxLayout()
        logo_info_layout.addWidget(self.logo_label)
        logo_info_layout.addWidget(self.creator_label)
        header_layout.addLayout(logo_info_layout)
        header_layout.addStretch()

        # Centered video display
        video_container_layout = QHBoxLayout()
        video_container_layout.addStretch()
        video_container_layout.addWidget(self.video_label)
        video_container_layout.addStretch()

        # Main content area
        main_content_layout = QVBoxLayout()
        main_content_layout.addLayout(header_layout)
        main_content_layout.addSpacing(20)
        main_content_layout.addLayout(video_container_layout)
        main_content_layout.addLayout(input_layout)
        main_content_layout.addLayout(button_layout)

        # Side panel for object selection
        side_panel_layout = QVBoxLayout()
        side_panel_layout.addWidget(QLabel("Available Objects:"))
        side_panel_layout.addWidget(self.objects_list)

        # Combine main content and side panel
        main_layout = QHBoxLayout()
        main_layout.addLayout(main_content_layout)
        main_layout.addLayout(side_panel_layout)
        self.setLayout(main_layout)

        # START VIDEO PROCESSING THREAD
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_updated.connect(self.update_frame)
        self.detection_thread.start()

    def on_object_selected(self, item):
        """
        Handle object selection from the available objects list.
        Automatically fills the input field with selected object name.
        """
        self.object_input.setText(item.text())

    def confirm_target_object(self):
        """
        Apply the target object filter based on user input.
        Empty input means detect all objects.
        """
        global target_object
        target_object = self.object_input.text().strip()
        if target_object:
            print(f"Target object set to: {target_object}")
        else:
            print("Target object cleared - showing all objects")

    def toggle_detection(self):
        """
        Toggle detection mode on/off.
        Automatically disables segmentation mode when detection is enabled.
        """
        global detection_enabled, segmentation_enabled
        if not detection_enabled:
            # Enable detection mode
            detection_enabled = True
            segmentation_enabled = False  # Mutual exclusion
            self.toggle_detection_button.setText("Disable Detection")
            self.toggle_segmentation_button.setText("Enable Segmentation")
        else:
            # Disable detection mode
            detection_enabled = False
            self.toggle_detection_button.setText("Enable Detection")

    def toggle_segmentation(self):
        """
        Toggle segmentation mode on/off.
        Automatically disables detection mode when segmentation is enabled.
        """
        global detection_enabled, segmentation_enabled
        if not segmentation_enabled:
            # Enable segmentation mode
            segmentation_enabled = True
            detection_enabled = False  # Mutual exclusion
            self.toggle_segmentation_button.setText("Disable Segmentation")
            self.toggle_detection_button.setText("Enable Detection")
        else:
            # Disable segmentation mode
            segmentation_enabled = False
            self.toggle_segmentation_button.setText("Enable Segmentation")

    def update_frame(self, frame):
        """
        Update the video display with processed frame from detection thread.
        Converts OpenCV BGR format to Qt-compatible RGB format.
        """
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display converted image in GUI
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """
        Handle application shutdown.
        Properly release camera resources and terminate background thread.
        """
        cap.release()
        cv2.destroyAllWindows()
        self.detection_thread.terminate()
        event.accept()


if __name__ == "__main__":
    # Create and run the PyQt5 application
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()