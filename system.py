from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QStackedWidget, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
import cv2
import sys
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import mediapipe as mp

mp_hands = mp.solutions.hands

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.showMaximized()

        # 設置堆疊窗口管理器
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 添加首頁介面
        self.home_widget = HomeWidget(self)
        self.stacked_widget.addWidget(self.home_widget)

        # 添加介面 A
        self.page_a_widget = PageAWidget()
        self.stacked_widget.addWidget(self.page_a_widget)

        # 信號與槽：首頁跳轉到介面 A
        self.home_widget.clicked.connect(self.show_page_a)

    def show_page_a(self):
        self.stacked_widget.setCurrentWidget(self.page_a_widget)

class HomeWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # 設定背景圖
        self.background_label = QLabel(self)
        pixmap = QPixmap("home.jpg")
        self.background_label.setPixmap(pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(self.rect())

        # 中心文字區域
        self.text_container = QWidget(self)
        self.text_container.setGeometry(self.rect())
        text_layout = QVBoxLayout(self.text_container)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_layout.setContentsMargins(0, 0, 0, 0)

        # 標題 "內關穴"
        self.title_label = QLabel("Neiguan acupoint Detect System", self.text_container)
        self.title_label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white;")

        # 說明文字
        self.description_label = QLabel("Touch the screen to start.", self.text_container)
        self.description_label.setFont(QFont("Arial", 18))
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setStyleSheet("color: white;")

        # 添加文字到布局
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.description_label)

    def resizeEvent(self, event):
        # 背景圖自適應窗口大小
        self.background_label.setGeometry(self.rect())

        # 文字區域自適應窗口大小
        self.text_container.setGeometry(self.rect())

    def mousePressEvent(self, event):
        # 發送點擊信號
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

# 定義模型
class ArmPointDetector(nn.Module):
    def __init__(self):
        super(ArmPointDetector, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        last_channel = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 輸出 (cx, cy)
        )

    def forward(self, x):
        return self.backbone(x)

class PageAWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 設置背景圖
        self.background_label = QLabel(self)
        pixmap = QPixmap("background.jpg")
        self.background_label.setPixmap(pixmap)
        self.background_label.setScaledContents(True)  # 啟用縮放
        self.background_label.setGeometry(self.rect())  # 初始全局大小

        # 影像轉換
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ArmPointDetector().to(self.device)
        self.model.load_state_dict(torch.load("final_model_noise300.pth", map_location=self.device))
        self.model.eval()

        # 主水平佈局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左邊：鏡頭畫面
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)

        # 右邊：內關穴標題、介紹與按鈕
        right_layout = QVBoxLayout()

        # 標題
        title_label = QLabel("Neiguan acupoint")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(title_label)

        # 介紹文字
        intro_label = QLabel(
            "The Neiguan acupoint is highly effective in relieving persistent hiccups, nausea, vomiting, motion sickness, dizziness, headaches, palpitations, chest tightness, insomnia, and bloating."
        )
        intro_label.setWordWrap(True)  # 自動換行
        intro_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        intro_label.setFont(QFont("Arial", 16))
        right_layout.addWidget(intro_label)

        # 開啟/關閉鏡頭按鈕
        self.toggle_button = QPushButton("Start detecting")
        self.toggle_button.setFont(QFont("Arial", 14))
        self.toggle_button.clicked.connect(self.toggle_camera)
        right_layout.addWidget(self.toggle_button)

        # 將右邊佈局添加到主佈局
        main_layout.addLayout(right_layout)

        # 設置背景圖為底層
        self.background_label.stackUnder(self.video_label)

        # 鏡頭相關屬性
        self.camera_on = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

    def resizeEvent(self, event):
        """根據視窗大小調整物件比例和背景圖"""
        window_width = self.width()
        window_height = self.height()

        # 更新背景圖大小
        self.background_label.setGeometry(self.rect())

        # 動態調整左邊鏡頭畫面大小 (占據 60% 高度，保持寬高比 4:3)
        video_width = int(window_width * 0.4)
        video_height = int(window_height * 0.6)
        self.video_label.setFixedSize(video_width, video_height)

        # 動態調整按鈕大小 (設定為窗口寬度的 20%，高度的 10%)
        button_width = int(window_width * 0.2)
        button_height = int(window_height * 0.1)
        self.toggle_button.setFixedSize(button_width, button_height)

        # 將按鈕放置在視窗下四分之一處，並加入偏移量往上移
        offset = int(window_height * 0.05)  # 偏移量（視窗高度的 5%）
        button_x = (window_width - button_width) // 2  # 水平居中
        button_y = window_height * 3 // 4 - button_height // 2 - offset  # 下四分之一處，往上移
        self.toggle_button.move(button_x, button_y)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.toggle_button.setText("開啟鏡頭")
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")  # 還原黑色背景
        else:
            self.camera_on = True
            self.toggle_button.setText("關閉鏡頭")
            self.cap = cv2.VideoCapture(0)  # 開啟鏡頭
            if not self.cap.isOpened():
                self.camera_on = False
                self.toggle_button.setText("開啟鏡頭")
                self.video_label.setText("無法開啟鏡頭")
                return
            self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 將 BGR 影像轉為 RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.flip(rgb_frame, 1)  # 水平翻轉
                h, w, _ = rgb_frame.shape

                # 手部偵測
                with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                    results = hands.process(rgb_frame)
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]

                        # 計算手指的 X 和 Y 範圍（將歸一化座標轉換為像素座標）
                        thumb_x = int(hand_landmarks.landmark[4].x * w)  # 大拇指
                        index_x = int(hand_landmarks.landmark[8].x * w)  # 第一指
                        pinky_x = int(hand_landmarks.landmark[20].x * w)  # 小指

                        # 計算左右邊界
                        left = max(0, min(thumb_x, index_x, pinky_x) - 50)
                        right = min(w, max(thumb_x, index_x, pinky_x) + 50)

                        # 中指的 Y 座標（轉換為像素座標）
                        middle_y = int(hand_landmarks.landmark[12].y * h)

                        # 計算上下邊界
                        top = max(0, middle_y - 30)
                        bottom = h

                        # 繪製紅框
                        rgb_frame = cv2.rectangle(
                            rgb_frame, (left, top), (right, bottom), (255, 0, 0), 2
                        )

                        # 裁剪紅框內的影像
                        cropped_frame = rgb_frame[top:bottom, left:right]
                        
                        # 檢查裁剪是否成功
                        if cropped_frame.size > 0:
                            # 應用模型進行預測
                            input_image = self.transform(cropped_frame).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                output = self.model(input_image)
                            cx, cy = output[0].cpu().numpy()
                            pred_cx, pred_cy = int(cx * cropped_frame.shape[1]), int(cy * cropped_frame.shape[0])

                            # 繪製預測點（在裁剪範圍內）
                            cropped_frame = cv2.circle(cropped_frame, (pred_cx, pred_cy), radius=5, color=(255, 0, 0), thickness=-1)

                            # 將裁剪後的預測結果繪製回原影像
                            rgb_frame[top:bottom, left:right] = cropped_frame

                # 顯示更新後的影像
                qt_image = QImage(
                    rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio
                )
                self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
