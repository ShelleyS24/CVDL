from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLayout
)
from PyQt5.QtCore import Qt

class MainWindowLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CVDL_HW2_VGG19")
        self.setGeometry(500, 500, 700, 500)  # 視窗大小

        # 建立主要版面配置
        self.main_layout = QHBoxLayout()

        # 左側按鈕佈局
        self.button_layout = QVBoxLayout()
        self.button_layout.setSizeConstraint(QLayout.SetFixedSize)

        # 載入圖片按鈕
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedSize(180, 40)  
        self.button_layout.addWidget(self.load_button)

        # 功能按鈕
        self.aug_button = QPushButton("1. Show Augmented Images")
        self.aug_button.setFixedSize(180, 40)  
        self.button_layout.addWidget(self.aug_button)

        self.model_button = QPushButton("2. Show Model Structure")
        self.model_button.setFixedSize(180, 40)  
        self.button_layout.addWidget(self.model_button)

        self.accuracy_button = QPushButton("3. Show Accuracy and Loss")
        self.accuracy_button.setFixedSize(180, 40)  
        self.button_layout.addWidget(self.accuracy_button)

        self.infer_button = QPushButton("4. Inference")
        self.infer_button.setFixedSize(180, 40)  
        self.button_layout.addWidget(self.infer_button)

        # 右側圖片和標籤佈局
        self.image_layout = QVBoxLayout()
        self.image_layout.setAlignment(Qt.AlignCenter)  # 設置佈局居中對齊

        # 圖片顯示區域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(128, 128)  # 固定圖片顯示大小
        self.image_layout.addWidget(self.image_label)

        # 預測結果標籤
        self.result_label = QLabel("Predicted: ", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px;")  # 設置字體大小
        self.image_layout.addWidget(self.result_label)

        # 將按鈕佈局和圖片佈局添加到主佈局
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.image_layout)

        # 設置中心窗口小部件
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)