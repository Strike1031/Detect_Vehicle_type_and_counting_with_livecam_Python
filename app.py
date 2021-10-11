import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from gui import *
import copy
from counter import *
from utils.sort import *
from models import *
from utils.utils import *
from utils.datasets import *
from config import *
import xlsxwriter
from time import gmtime, strftime, sleep
import _thread
import threading

video_time = "00:00:00"

class App(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.setupUi(self)
        self.label_image_size = (self.label_image.geometry().width(), self.label_image.geometry().height())
        self.video = None
        self.exampleImage = None
        self.imgScale = None
        self.get_points_flag = 0
        self.countArea = []
        self.road_code = None
        self.time_code = None
        self.show_label = names

        # button function
        self.pushButton_selectArea.clicked.connect(self.select_area)
        self.pushButton_openRtsp.clicked.connect(self.open_rtsp)
        self.pushButton_openVideo.clicked.connect(self.open_video)
        self.pushButton_start.clicked.connect(self.start_count)
        self.pushButton_pause.clicked.connect(self.pause)
        self.label_image.mouseDoubleClickEvent = self.get_points

        self.pushButton_selectArea.setEnabled(False)
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setEnabled(False)

        # some flags
        self.running_flag = 0
        self.pause_flag = 0
        self.counter_thread_start_flag = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_config = "config/coco.data"
        weights_path = "weights/yolov3.weights"
        model_def = "config/yolov3.cfg"
        data_config = parse_data_config(data_config)
        self.yolo_class_names = load_classes(data_config["names"])

        # Initiate model
        print("Loading model ...")
        self.yolo_model = Darknet(model_def).to(self.device)
        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.yolo_model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.yolo_model.load_state_dict(torch.load(weights_path))

        # counter Thread
        self.counterThread = CounterThread(self.yolo_model, self.yolo_class_names, self.device)
        self.counterThread.sin_counterResult.connect(self.show_image_label)
        self.counterThread.sin_done.connect(self.done)
        self.counterThread.sin_counter_results.connect(self.update_counter_results)
        self.started_video = False
        _thread.start_new_thread(self.show_time, ())


    def open_rtsp(self):
        self.videoList = [vars(self)[f"text_rtsp"].text()]
        # print (vars(self)[f"text_rtsp"].text())
        # gst_str = ('rtspsrc location={} latency={} ! '
        #            'rtph264depay ! h264parse ! omxh264dec ! '
        #            'nvvidconv ! '
        #            'video/x-raw, width=(int){}, height=(int){}, '
        #            'format=(string)BGRx ! '
        #            'videoconvert ! appsink').format(self.videoList[0], 1, 640, 480)

        print(vars(self)[f"text_rtsp"].text())

        vid = cv2.VideoCapture(self.videoList[0])

        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break

        self.pushButton_selectArea.setEnabled(True)
        self.pushButton_start.setText("Start")
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setText("Pause")
        self.pushButton_pause.setEnabled(False)

        # clear counting results
        KalmanBoxTracker.count = 0
        self.label_sum.setText("0")
        self.label_sum.repaint()

    def open_video(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'Open video', '', 'Video files(*.avi , *.mp4)')
        self.videoList = [openfile_name[0]]

        # opendir_name = QFileDialog.getExistingDirectory(self, "Open dir", "./")
        # self.videoList = [os.path.join(opendir_name,item) for item in os.listdir(opendir_name)]
        # self.videoList = list(filter(lambda x: not os.path.isdir(x) , self.videoList))
        # self.videoList.sort()

        vid = cv2.VideoCapture(self.videoList[0])

        # self.videoWriter = cv2.VideoWriter(openfile_name[0].split("/")[-1], cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (1920, 1080))

        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break

        self.pushButton_selectArea.setEnabled(True)
        self.pushButton_start.setText("Start")
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setText("Pause")
        self.pushButton_pause.setEnabled(False)

        # clear counting results
        KalmanBoxTracker.count = 0
        self.label_sum.setText("0")
        self.label_sum.repaint()

    def get_points(self, event):
        if self.get_points_flag:
            x = event.x()
            y = event.y()
            self.countArea.append([int(x * self.imgScale[1]), int(y * self.imgScale[0])])
            exampleImageWithArea = copy.deepcopy(self.exampleImage)
            for point in self.countArea:
                exampleImageWithArea[point[1] - 10:point[1] + 10, point[0] - 10:point[0] + 10] = (0, 255, 255)
            cv2.fillConvexPoly(exampleImageWithArea, np.array(self.countArea), (0, 0, 255))
            self.show_image_label(exampleImageWithArea)
        print(self.countArea)

    def select_area(self):

        # change Area needs update exampleImage
        if self.counter_thread_start_flag:
            ret, frame = self.videoCapture.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)

        if not self.get_points_flag:
            self.pushButton_selectArea.setText("Submit Area")
            self.get_points_flag = 1
            self.countArea = []
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_start.setEnabled(False)

        else:
            self.pushButton_selectArea.setText("Select Area")
            self.get_points_flag = 0
            exampleImage = copy.deepcopy(self.exampleImage)
            # painting area
            for i in range(len(self.countArea)):
                cv2.line(exampleImage, tuple(self.countArea[i]), tuple(self.countArea[(i + 1) % (len(self.countArea))]),
                         (0, 0, 255), 2)
            self.show_image_label(exampleImage)

            # enable start button
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(True)

    def show_image_label(self, img_np):
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.label_image_size)
        frame = QImage(img_np, self.label_image_size[0], self.label_image_size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.label_image.setPixmap(pix)
        self.label_image.repaint()

    def start_count(self):
        begin_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        global sum_speed
        for i in range(0,9999):
            sum_speed[i] = 0

        if self.running_flag == 0:
            print("Start")
            self.started_video = True
            self.started_time = gmtime()

            # clear count and display
            KalmanBoxTracker.count = 0
            for item in self.show_label:
                vars(self)[f"label_{item}"].setText('0')
            vars(self)[f"label_sum"].setText('0')
            # clear result file
            with open("results/results.txt", "w") as f:
                pass

            # start
            self.running_flag = 1
            self.pause_flag = 0
            self.pushButton_start.setText("Stop")
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_selectArea.setEnabled(False)
            # emit new parameter to counter thread
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.counterThread.sin_countArea.emit(self.countArea)
            self.counterThread.sin_videoList.emit(self.videoList)
            # start counter thread
            self.counterThread.start()

            self.pushButton_pause.setEnabled(True)


        elif self.running_flag == 1:  # push pause button
            self.started_video = False
            started_time = gmtime()
            last_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            cur_date = strftime("%Y-%m-%d", gmtime())

            # stop system
            self.running_flag = 0
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_selectArea.setEnabled(True)
            self.pushButton_start.setText("Start")
            # export the result into xlsx file

            workbook = xlsxwriter.Workbook('results/`output.xlsx')
            worksheet = workbook.add_worksheet(cur_date)

            worksheet.write("A1", "Date/Time")
            worksheet.write("A2", begin_time + "-" + last_time)
            worksheet.write("A3", begin_time + "-" + last_time)
            worksheet.write("A4", begin_time + "-" + last_time)
            worksheet.write("A5", begin_time + "-" + last_time)
            worksheet.write("A6", begin_time + "-" + last_time)

            worksheet.write("B1", "Vehicle Type")
            worksheet.write("B2", "Car")
            worksheet.write("B3", "Bus")
            worksheet.write("B4", "Truck")
            worksheet.write("B5", "Motorbike")
            worksheet.write("B6", "Bicycle")

            worksheet.write("C1", "Count")
            label_var = vars(self)[f"label_car"]
            worksheet.write("C2", label_var.text())
            label_var = vars(self)[f"label_bus"]
            worksheet.write("C3", label_var.text())
            label_var = vars(self)[f"label_truck"]
            worksheet.write("C4", label_var.text())
            label_var = vars(self)[f"label_motorbike"]
            worksheet.write("C5", label_var.text())
            label_var = vars(self)[f"label_bicycle"]
            worksheet.write("C6", label_var.text())

            worksheet.write("D1", "Speed")
            label_var = vars(self)[f"speed_car"]
            worksheet.write("D2", label_var.text())
            label_var.setText("0")
            label_var = vars(self)[f"speed_bus"]
            worksheet.write("D3", label_var.text())
            label_var.setText("0")
            label_var = vars(self)[f"speed_truck"]
            worksheet.write("D4", label_var.text())
            label_var.setText("0")
            label_var = vars(self)[f"speed_motorbike"]
            worksheet.write("D5", label_var.text())
            label_var.setText("0")
            label_var = vars(self)[f"speed_bicycle"]
            worksheet.write("D6", label_var.text())
            label_var.setText("0")

            label_var = vars(self)[f"label_car"]
            label_var.setText("0")
            label_var = vars(self)[f"label_truck"]
            label_var.setText("0")
            label_var = vars(self)[f"label_bus"]
            label_var.setText("0")
            label_var = vars(self)[f"label_motorbike"]
            label_var.setText("0")
            label_var = vars(self)[f"label_bicycle"]
            label_var.setText("0")
            label_var = vars(self)[f"label_sum"]
            label_var.setTest("0")

            workbook.close()


    def done(self, sin):
        if sin == 1:
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(False)
            self.pushButton_start.setText("Start")




    def update_counter_results(self, counter_results):
        global sum_speed
        with open("results/results.txt", "a") as f:
            for i, result in enumerate(counter_results):
                label_var = vars(self)[f"label_{result[2]}"]
                label_var.setText(str(int(label_var.text()) + 1))

                speed_var = vars(self)[f"speed_{result[2]}"]
                distance_var = vars(self)[f"label_distance"].text()
                print(distance_var)

                if sum_speed[objectName2number(result[2])] / int(label_var.text()) < 0:
                    speed_var.setText(str(float(
                        "{:.1f}".format(-sum_speed[objectName2number(result[2])] / int(label_var.text())*float(distance_var)/10))) + " km/h")
                else:
                    speed_var.setText(str(float(
                        "{:.1f}".format(sum_speed[objectName2number(result[2])] / int(label_var.text())*float(distance_var)/10))) + " km/h")

                label_var.repaint()
                label_sum_var = vars(self)[f"label_sum"]
                label_sum_var.setText(str(int(label_sum_var.text()) + 1))
                label_sum_var.repaint()
                f.writelines(' '.join(map(lambda x: str(x), result)))
                f.write(("\n"))
        # print("************************************************",len(counter_results))

    def pause(self):
        started_time = gmtime()
        if self.pause_flag == 0:
            self.pause_flag = 1
            self.pushButton_pause.setText("Continue")
            self.pushButton_start.setEnabled(False)
        else:
            self.pause_flag = 0
            self.pushButton_pause.setText("Pause")
            self.pushButton_start.setEnabled(True)

        self.counterThread.sin_pauseFlag.emit(self.pause_flag)

    def show_time(self):
        while (True):
            if self.started_video == True:
                cur_time = gmtime()
                period = time.mktime(cur_time) - time.mktime(self.started_time)
                period_time = str(int(period/3600))+":"+str(int((period % 3600)/60))+":"+str(int((period % 60)))
                video_time = period_time
            else:
                video_time = "00:00:00"
            vars(self)[f"label_8"].setText(video_time)

            sleep(1)

def objectName2number(objectName):
    if(objectName == 'car'):
        return 0
    if(objectName == 'truck'):
        return 1
    if(objectName == 'bus'):
        return 2
    if(objectName == 'motorbike'):
        return 3
    if(objectName == 'bicycle'):
        return 4



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())