import sys
from PyQt5.QtWidgets import QApplication
from Q2_layout import DcGANLayout
import Q2


class DcGANMain(DcGANLayout):
    def __init__(self):
        super().__init__()

        # 參數
        self.dataroot = "./Q2_images/mnist/"
        self.loss_plot = "loss_plot.png"
        self.model_path = r"weights\netG_epoch_50.pth"

        # 綁定按鈕功能
        self.btn_show_images.clicked.connect(self.show_training_images)
        self.btn_show_structure.clicked.connect(self.show_model_structure)
        self.btn_show_loss.clicked.connect(self.show_training_loss)
        self.btn_inference.clicked.connect(self.inference)

    # Q2-1
    def show_training_images(self):
        fig = Q2.process_and_display_mnist(self.dataroot)
        fig.show()

    # Q2-2
    def show_model_structure(self):
        Q2.show_model_structure()

    # Q2-3
    def show_training_loss(self):
        Q2.show_loss(self.loss_plot)

    # Q2-4
    def inference(self):
        Q2.inference(self.model_path, self.dataroot)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DcGANMain()
    window.show()
    sys.exit(app.exec_())
