from ultralytics import YOLO
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, \
    QListWidget, QScrollArea
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Load YOLOv8 pretrained model (coco dataset has chairs, tables, etc.)
model_detection = YOLO('yolov8n.pt')  # YOLOv8 model for detection
model_segmentation = YOLO('yolov8n-seg.pt')  # YOLOv8 model for segmentation

# Initialize variables
cap = cv2.VideoCapture(1)
detection_enabled = False
segmentation_enabled = False
target_object = ""  # Empty means detect all objects


class DetectionThread(QThread):
    frame_updated = pyqtSignal(np.ndarray)

    def run(self):
        global detection_enabled, segmentation_enabled, target_object
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if detection_enabled and not segmentation_enabled:
                # Perform object detection only
                results = model_detection.predict(source=frame, conf=0.5, stream=True)

                detected_objects = []
                annotated_frame = frame.copy()

                for result in results:
                    if target_object:
                        # Filter results for target object only
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                object_name = model_detection.names[int(box.cls)]
                                if target_object.lower() in object_name.lower():
                                    detected_objects.append(object_name)
                                    # Draw filtered detection manually
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, object_name, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Show all detected objects
                        annotated_frame = result.plot()
                        if result.boxes is not None:
                            detected_objects.extend([model_detection.names[int(box.cls)] for box in result.boxes])

                # Overlay detected object names on the OpenCV frame
                y_offset = 20
                for obj in set(detected_objects):  # Use `set` to ensure unique names
                    cv2.putText(annotated_frame, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 20

            elif segmentation_enabled and not detection_enabled:
                # Perform semantic segmentation only
                results = model_segmentation.predict(source=frame, conf=0.5, stream=True)

                annotated_frame = frame.copy()
                for result in results:
                    if hasattr(result, 'masks') and result.masks is not None:
                        if target_object:
                            # Filter segmentation results for target object only
                            if result.boxes is not None and len(result.boxes) > 0:
                                for i, box in enumerate(result.boxes):
                                    object_name = model_segmentation.names[int(box.cls)]
                                    if target_object.lower() in object_name.lower():
                                        # Draw filtered segmentation manually
                                        if i < len(result.masks.data):
                                            mask = result.masks.data[i].cpu().numpy()
                                            colored_mask = np.zeros_like(annotated_frame)
                                            colored_mask[mask > 0.5] = [0, 255, 0]
                                            annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.3, 0)

                                        # Draw bounding box and label
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, object_name, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            annotated_frame = result.plot()
                    else:
                        annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()

            self.frame_updated.emit(annotated_frame)

        cap.release()
        cv2.destroyAllWindows()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection and Segmentation")
        self.resize(1200, 700)  # Make window larger

        # Create buttons
        self.toggle_detection_button = QPushButton("Enable Detection")
        self.toggle_detection_button.clicked.connect(self.toggle_detection)

        self.toggle_segmentation_button = QPushButton("Enable Segmentation")
        self.toggle_segmentation_button.clicked.connect(self.toggle_segmentation)

        # Create text input and confirm button for object filtering
        self.object_input = QLineEdit()
        self.object_input.setPlaceholderText("Enter object name (e.g., person, car, chair)")
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_target_object)

        # Create labels for displaying video
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # Create logo label
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(100, 80)
        try:
            logo_pixmap = QPixmap("urmia_logo.png")
            scaled_logo = logo_pixmap.scaled(80, 80, aspectRatioMode=1)  # Keep aspect ratio
            self.logo_label.setPixmap(scaled_logo)
        except:
            self.logo_label.setText("Logo")
            self.logo_label.setStyleSheet("border: 1px solid gray; text-align: center;")

        # Create creator label
        self.creator_label = QLabel("Creator: Mahya Kheirandish")
        self.creator_label.setStyleSheet("font-size: 12px; color: black; text-align: left;")
        self.creator_label.setFixedSize(200, 20)

        # Create side panel with available objects list
        self.objects_list = QListWidget()
        self.objects_list.setMaximumWidth(200)
        self.objects_list.setMinimumWidth(200)

        # Add all available object classes to the list
        available_objects = list(model_detection.names.values())
        for obj in sorted(available_objects):
            self.objects_list.addItem(obj)

        # Connect list item click to input field
        self.objects_list.itemClicked.connect(self.on_object_selected)

        # Layout for main content
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Target Object:"))
        input_layout.addWidget(self.object_input)
        input_layout.addWidget(self.confirm_button)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.toggle_detection_button)
        button_layout.addWidget(self.toggle_segmentation_button)

        # Create top header layout with logo and creator at very top
        header_layout = QHBoxLayout()
        logo_info_layout = QVBoxLayout()
        logo_info_layout.addWidget(self.logo_label)
        logo_info_layout.addWidget(self.creator_label)
        header_layout.addLayout(logo_info_layout)
        header_layout.addStretch()  # Push everything else to the right

        # Video layout centered with proper spacing
        video_container_layout = QHBoxLayout()
        video_container_layout.addStretch()
        video_container_layout.addWidget(self.video_label)
        video_container_layout.addStretch()

        main_content_layout = QVBoxLayout()
        main_content_layout.addLayout(header_layout)  # Logo at top
        main_content_layout.addSpacing(20)  # Add space between logo and video
        main_content_layout.addLayout(video_container_layout)  # Video centered
        main_content_layout.addLayout(input_layout)
        main_content_layout.addLayout(button_layout)

        # Side panel layout
        side_panel_layout = QVBoxLayout()
        side_panel_layout.addWidget(QLabel("Available Objects:"))
        side_panel_layout.addWidget(self.objects_list)

        # Main horizontal layout combining content and side panel
        main_layout = QHBoxLayout()
        main_layout.addLayout(main_content_layout)
        main_layout.addLayout(side_panel_layout)
        self.setLayout(main_layout)

        # Detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_updated.connect(self.update_frame)
        self.detection_thread.start()

    def on_object_selected(self, item):
        """Fill the input field when an object is selected from the list"""
        self.object_input.setText(item.text())

    def confirm_target_object(self):
        global target_object
        target_object = self.object_input.text().strip()
        if target_object:
            print(f"Target object set to: {target_object}")
        else:
            print("Target object cleared - showing all objects")

    def toggle_detection(self):
        global detection_enabled, segmentation_enabled
        if not detection_enabled:  # Enable detection
            detection_enabled = True
            segmentation_enabled = False  # Ensure segmentation is disabled
            self.toggle_detection_button.setText("Disable Detection")
            self.toggle_segmentation_button.setText("Enable Segmentation")
        else:  # Disable detection
            detection_enabled = False
            self.toggle_detection_button.setText("Enable Detection")

    def toggle_segmentation(self):
        global detection_enabled, segmentation_enabled
        if not segmentation_enabled:  # Enable segmentation
            segmentation_enabled = True
            detection_enabled = False  # Ensure detection is disabled
            self.toggle_segmentation_button.setText("Disable Segmentation")
            self.toggle_detection_button.setText("Enable Detection")
        else:  # Disable segmentation
            segmentation_enabled = False
            self.toggle_segmentation_button.setText("Enable Segmentation")

    def update_frame(self, frame):
        # Convert OpenCV frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display QImage in QLabel
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        cap.release()
        cv2.destroyAllWindows()
        self.detection_thread.terminate()
        event.accept()


# Run the application
app = QApplication([])
window = MainWindow()
window.show()
app.exec_()