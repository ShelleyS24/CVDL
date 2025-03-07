from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtCore import Qt

class DcGANLayout(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVDL_HW2_DC_GAN")
        self.setGeometry(500, 500, 500, 600)  
        self.init_ui()

    def init_ui(self):
        # 主 Widget 和 Layout
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # 建立一個 QGroupBox 來包住按鈕並顯示標題
        self.group_box = QGroupBox("Question2 DcGAN", self)
        self.group_box.setAlignment(Qt.AlignLeft)

        # 建立按鈕的 Layout
        button_layout = QVBoxLayout()
        self.btn_show_images = QPushButton("1. Show Training Images", self.group_box)
        self.btn_show_structure = QPushButton("2. Show Model Structure", self.group_box)
        self.btn_show_loss = QPushButton("3. Show Training Loss", self.group_box)
        self.btn_inference = QPushButton("4. Inference", self.group_box)

        # 設定按鈕的固定大小
        button_size = (240, 60)
        self.btn_show_images.setFixedSize(*button_size)
        self.btn_show_structure.setFixedSize(*button_size)
        self.btn_show_loss.setFixedSize(*button_size)
        self.btn_inference.setFixedSize(*button_size)

        # 使用 QHBoxLayout 來置中按鈕
        centered_layout = QHBoxLayout()
        centered_layout.addStretch(1)
        centered_layout.addLayout(button_layout)
        centered_layout.addStretch(1)

        # 加入按鈕到 Layout
        button_layout.addWidget(self.btn_show_images)
        button_layout.addWidget(self.btn_show_structure)
        button_layout.addWidget(self.btn_show_loss)
        button_layout.addWidget(self.btn_inference)

        # 將按鈕 Layout 加入到 group_box
        self.group_box.setLayout(centered_layout)

        # 將 group_box 加入到主 Layout
        main_layout.addWidget(self.group_box)
        self.setCentralWidget(main_widget)