import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from Q1_layout import MainWindowLayout
import Q1


class MainWindow(MainWindowLayout):
    def __init__(self):
        super().__init__()

        # 參數
        self.image_path_Q1_1 = "Q1_image\Q1_1"
        self.training_validation_metrics = "training_validation_metrics.png"
        self.image_path_Q1_4 = None

        # 綁定按鈕功能
        self.load_button.clicked.connect(self.load_image)
        self.aug_button.clicked.connect(self.show_augmented_images)
        self.model_button.clicked.connect(self.show_model_structure)
        self.accuracy_button.clicked.connect(self.show_accuracy_loss)
        self.infer_button.clicked.connect(self.inference)

    # 載入圖片功能
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.xpm *.jpg)", options=options
        )
        if file_name:
            self.image_path_Q1_4 = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(128, 128)
            self.image_label.setPixmap(pixmap)
            print("圖片已載入")

    # Q1-1
    def show_augmented_images(self):
        fig = Q1.load_image(self.image_path_Q1_1)
        fig.show()

    # Q1-2
    def show_model_structure(self):
        Q1.show_model_summary()

    # Q1-3
    def show_accuracy_loss(self):
        Q1.show_training_validation_metrics(self.training_validation_metrics)

    # Q1-4
    def inference(self):
        Q1.inference_and_show(self.image_path_Q1_4, self.result_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
