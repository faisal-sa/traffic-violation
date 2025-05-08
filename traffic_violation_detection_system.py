import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel,
    QRadioButton, QButtonGroup, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage,QMouseEvent,QKeyEvent
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject,QRect
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo 
import threading
try:
    from picamera2 import Picamera2
except ImportError:
    pass 
import cv2
import os
import time
import pickle

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""


polygons = []




class Polygons:
    def __init__(self, name: str, poly_type: str):
        self.name = name
        self.poly_type = poly_type
        self.points = np.empty((0, 2), dtype=np.int32)
    def add_point(self, point):
        if len(point) != 2:
            raise ValueError("Point must be a tuple or list with 2 elements (x, y).")
        self.points = np.vstack([self.points, point])
    def delete_point(self, index):
        if index < 0 or index >= len(self.points):
            raise IndexError("Point index out of range.")
        self.points = np.delete(self.points, index, axis=0)
    def render(self, image, color=(0, 255, 0), thickness=2, closed=True):
        if self.points.shape[0] >= 1:
            pts = self.points.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=closed, color=color, thickness=thickness)
        return image
    def __repr__(self):
        return f"Polygons(name={self.name}, type={self.poly_type}, points=\n{self.points})"
    
    def contains_point(self, point):

        if len(self.points) < 3:
            return False  
            
        x, y = point
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0]
        for i in range(n + 1):
            p2x, p2y = self.points[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    # Save list of polygons to a file using pickle
    @staticmethod
    def save_polygons(polygons, filename="polygons.pkl"):
        with open(filename, 'wb') as file:
            pickle.dump(polygons, file)
    
    # Load list of polygons from a file using pickle
    @staticmethod
    def load_polygons(filename="polygons.pkl"):
        with open(filename, 'rb') as file:
            return pickle.load(file)




class ClickableLabel(QLabel):
    
    def mousePressEvent(self, ev: QMouseEvent):
        if self.pixmap() is None:
            return
        if polygons:
            xw, yw = ev.x(), ev.y()

            content = self.contentsRect()
            pm_size = self.pixmap().size()
            px = content.x() + (content.width() - pm_size.width())//2
            py = content.y() + (content.height() - pm_size.height())//2
            pixmap_rect = QRect(px, py, pm_size.width(), pm_size.height())

            if not pixmap_rect.contains(xw, yw):
                return

            # click relative to pixmap
            xr = xw - pixmap_rect.x()
            yr = yw - pixmap_rect.y()

            # map back to original image size
            orig_w = self.pixmap().width()
            orig_h = self.pixmap().height()
            scale_x = orig_w / pixmap_rect.width()
            scale_y = orig_h / pixmap_rect.height()

            xf = int(xr * scale_x)
            yf = int(yr * scale_y)
            poly=polygons[len(polygons)-1]
            if ev.button() == Qt.LeftButton:
                poly.add_point([xf, yf])

            if ev.button() == Qt.RightButton:
                if len(poly.points)>0:
                    poly.delete_point(len(poly.points)-1)
                else:
                    polygons.pop()

                











M_w, M_h = 640, 640

W, H = 1280, 720

s = min(M_w / W, M_h / H) 

# Calculate padding
p_x = (M_w - s * W) / 2
p_y = (M_h - s * H)/ 2




def get_pipeline(type,path=None):
    if type=="rpi":
        pipeline = "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! videoflip name=videoflip video-direction=horiz ! video/x-raw, format=RGB, width=1280, height=720 !  queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720  ! queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/faisalpi/hailo-apps-infra/hailo_apps_infra/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! hailofilter name=inference_hailofilter so-path=/home/faisalpi/hailo-apps-infra/hailo_apps_infra/./libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false !  queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=False qos=False ! queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! queue name=t->o_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback !queue name=usercallback->sink_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !fakesink sync=true"
    elif type=="file":
        pipeline = f"filesrc location={path} name='source' ! queue name=source_queue_decode leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! decodebin name=source_decodebin !  queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720  ! queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/faisalpi/hailo-apps-infra/hailo_apps_infra/yolov11n.hef batch-size=2 vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.3 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/faisalpi/hailo-apps-infra/hailo_apps_infra/./libyolo_hailortpp_postprocess.so function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=False qos=False ! queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! queue name=t->o_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback !queue name=usercallback->sink_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !fakesink sync=true"


    return pipeline

 
class VideoSignal(QObject):
	frame_ready = pyqtSignal(tuple)


class VideoProcessor:
    def __init__(self, pipeline_str=None, signal=None,current_mode=None):
        self.pipeline_str=pipeline_str
        self.signal=signal
        self.main_loop=None
        self.loop_thread=None
        self.picam_thread=None
        self.current_mode=current_mode
        #self.pipeline_state=False
        self.running=False
    
    def start(self):
        try:
            if self.current_mode=="rpi":
                self.pipeline = Gst.parse_launch(self.pipeline_str)

                identity_element = self.pipeline.get_by_name("identity_callback")

                pad = identity_element.get_static_pad("src")

                pad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer_callback, None)

                appsrc = self.pipeline.get_by_name("app_source")
                appsrc.set_property('caps', Gst.Caps.from_string("video/x-raw,format=RGB,width=1280,height=720,framerate=30/1"))
                appsrc.set_property("format", Gst.Format.TIME)

                self.pipeline.set_state(Gst.State.PLAYING)

                self.running = True


                self.main_loop = GLib.MainLoop()
                self.loop_thread = threading.Thread(target=self.main_loop.run)
                self.loop_thread.daemon = True
                self.loop_thread.start()
                self.picam_thread = threading.Thread(target=self.picamera_thread,args=(appsrc,))
                self.picam_thread.daemon=True
                self.picam_thread.start()
                return True
                




            else:
                self.pipeline = Gst.parse_launch(self.pipeline_str)
                identity_element = self.pipeline.get_by_name("identity_callback")
                pad = identity_element.get_static_pad("src")
                pad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer_callback, None)

                self.pipeline.set_state(Gst.State.PLAYING)
                self.running = True

                self.main_loop = GLib.MainLoop()
                self.loop_thread = threading.Thread(target=self.main_loop.run)
                self.loop_thread.daemon = True
                self.loop_thread.start()
                return True



        except Exception as e:
            print(f"Error starting GStreamer pipeline: {e}")
            return False

    def stop(self):
        if self.running:
            self.running = False
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            if self.main_loop and self.main_loop.is_running():
                self.main_loop.quit()
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=1)
            if self.picam_thread and self.current_mode == "rpi":
                self.picam_thread.join(timeout=1)
    



    def picamera_thread(self,appsrc):
        try:
            with Picamera2() as picam2:
                # Default configuration
                main = {'size': (1280, 720), 'format': 'RGB888'}
                lores = {'size': (1280, 720), 'format': 'RGB888'}
                controls = {'FrameRate': 30}
                config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
                picam2.configure(config)
                picam2.start()
                
                print("Picamera process started")
                while self.running:
                    frame_data = picam2.capture_array('lores')
                    if frame_data is None:
                        print("Failed to capture frame.")
                        continue
                        
                    frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                    frame = np.asarray(frame)
                    buffer = Gst.Buffer.new_wrapped(frame.tobytes())
                    ret = appsrc.emit('push-buffer', buffer)
                    if ret != Gst.FlowReturn.OK:
                        print("Failed to push buffer:", ret)
                        break
                    time.sleep(0.033)

        except Exception as e:
            print(f"Error in picamera thread: {e}")
        finally:
            print("Picamera thread stopped")






    def on_buffer_callback(self, pad, probe_info, user_data):
        if not self.running:
            return Gst.PadProbeReturn.OK
        buffer = probe_info.get_buffer()
        caps = pad.get_current_caps()
        if not caps:
            return Gst.PadProbeReturn.OK
        structure = caps.get_structure(0)
        width, height = structure.get_value("width"), structure.get_value("height")
        # Map the buffer to numpy array
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        try:
            # Create numpy array from buffer data
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()

            self.signal.frame_ready.emit((frame,roi,detections))
        except Exception as e:
            print(f"Error processing buffer: {e}")
        finally:
            buffer.unmap(map_info)
        
        return Gst.PadProbeReturn.OK



class UI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic violation detection system")
        #self.showFullScreen()
        self.setGeometry(100,100,800,600)
        self.video_path=None
        self.current_mode=None
        self.video_signals = VideoSignal()
        self.video_processor=VideoProcessor()
        self.video_signals.frame_ready.connect(self.update_frame, Qt.QueuedConnection)#connect signal to call a slot

        self.setup_ui()
        self.showMaximized()        


    def setup_ui(self):
        main_layout = QVBoxLayout()
        bottom_panel = QHBoxLayout()


        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout()
        self.rpi_rb = QRadioButton("Raspberry Pi Camera")
        self.video_rb = QRadioButton("Video File")
        # Connect radio buttons
        self.rpi_rb.toggled.connect(self.on_source_selected)
        self.video_rb.toggled.connect(self.on_source_selected)

        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_video)
        self.btn_browse.setEnabled(False)
        
        source_button_group = QButtonGroup(self)
        source_button_group.addButton(self.rpi_rb)
        source_button_group.addButton(self.video_rb)

        source_layout.addWidget(self.rpi_rb)
        source_layout.addWidget(self.video_rb)
        source_layout.addWidget(self.btn_browse)

        source_group.setLayout(source_layout)


        #self.polygon_status=QLabel("Press 'A' to add a new polygon. 'ESC' to exit.")

        # Status label
        self.lbl_status = QLabel("Ready")

        # Start/Stop button
        self.btn_start_stop = QPushButton("Start")
        self.btn_start_stop.clicked.connect(self.toggle_streaming)
        self.btn_start_stop.setEnabled(False)






        #self.newbtn=QPushButton("add")




        bottom_panel.addWidget(source_group, alignment=Qt.AlignLeft)
        #bottom_panel.addWidget(self.newbtn,alignment=Qt.AlignCenter)
        #bottom_panel.addWidget(self.polygon_status,alignment=Qt.AlignCenter)
        #bottom_panel.addWidget(self.lbl_status,alignment=Qt.AlignRight)
        bottom_panel.addWidget(self.btn_start_stop, alignment=Qt.AlignRight)

        



        self.frame_display = ClickableLabel()
        self.frame_display.setAlignment(Qt.AlignCenter)
        self.frame_display.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.frame_display.setMinimumSize(640, 480)

        main_layout.addWidget(self.frame_display, stretch=2)
        main_layout.addLayout(bottom_panel, stretch=1)

        self.setLayout(main_layout)

    def on_source_selected(self):
        if self.video_rb.isChecked():
            self.btn_browse.setEnabled(True)
            self.btn_start_stop.setEnabled(True)
            self.current_mode = "video"
        elif self.rpi_rb.isChecked():
            self.btn_browse.setEnabled(False)
            self.btn_start_stop.setEnabled(True)
            self.current_mode = "rpi"
    
    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.btn_start_stop.setEnabled(True)

    def toggle_streaming(self):
            if self.btn_start_stop.text() == "Start":
                success = self.start_streaming()
                if success:
                    self.btn_start_stop.setText("Stop")
                    self.lbl_status.setText("Streaming...")
                else:
                    self.lbl_status.setText("Failed to start streaming")
            else:
                self.stop_streaming()
                self.btn_start_stop.setText("Start")
                self.lbl_status.setText("Ready")

 
    def start_streaming(self):
        # Stop any existing video processing
        self.stop_streaming(update_button=False)
        if self.current_mode == "video" and self.video_path:
            try:
                pipeline_str = get_pipeline("file", self.video_path)
                self.video_processor = VideoProcessor(pipeline_str, self.video_signals,self.current_mode)
                if self.video_processor.start():
                    return True
                else:
                    raise Exception("Failed to start GStreamer pipeline")
            except Exception as e:
                print(f"GStreamer pipeline failed: {e}")
                # Fallback to OpenCV if GStreamer fails
                #return self.simple_capture.start(self.video_path)
                
        elif self.current_mode == "rpi":
            try:
                pipeline_str = get_pipeline("rpi")
                self.video_processor = VideoProcessor(pipeline_str, self.video_signals,self.current_mode)
                if self.video_processor.start():
                    return True
                else:
                    raise Exception("Failed to start GStreamer pipeline")
            except Exception as e:
                print(f"GStreamer pipeline failed: {e}")
            # Start Raspberry Pi camera
                

    def stop_streaming(self, update_button=True):
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor = None
                    
        if update_button:
            self.btn_start_stop.setText("Start")


    @pyqtSlot(np.ndarray)
    def update_frame(self, data):
        try:
            if data[0] is None or data[0].size == 0:
                return
            frame = data[0]
            roi = data[1] if len(data) > 1 else None
            detections = data[2] if len(data) > 2 else None
                
            detection_count = 0
            if detections != None:
                for detection in detections:
                    if detection.get_label() == "car" or detection.get_label() == "truck":
                        label = detection.get_label()
                        bbox = detection.get_bbox()
                        confidence = detection.get_confidence()
                        lane=-1
                        # Get track ID
                        track_id = 0
                        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                        if len(track) == 1:
                            track_id = track[0].get_id()
                        


                        x_det = bbox.xmin() * M_w
                        y_det = bbox.ymin() * M_h
                        x2_det = bbox.xmax() * M_w
                        y2_det = bbox.ymax() * M_h

                        x_unpad  = x_det -p_x
                        y_unpad  = (y_det -p_y) +20
                        x2_unpad = x2_det -p_x
                        y2_unpad = (y2_det -p_y)

                        x1 = int(x_unpad  / s)
                        y1 = int(y_unpad  / s)
                        x2 = int(x2_unpad / s)
                        y2 = int(y2_unpad / s)

                        y_center = (y1 + y2) / 2
                        y_norm = (y_center / H) - 0.5
                        

                        a1=350
                        a3=-350
                        delta_y = a1 * y_norm + a3 * y_norm**3

                        y1 = int(y1 - delta_y)
                        y2 = int(y2 - delta_y)
                        # 4) Clamp to image bounds
                        x1 = max(0, min(W, x1))
                        y1 = max(0, min(H, y1))
                        x2 = max(0, min(W, x2))
                        y2 = max(0, min(H, y2))

                        
                        ycenter=int((y1+y2)/2)
                        xcenter=int((x1+x2)/2)
                        point = (xcenter,ycenter)
                        



                        #cv2.circle(frame,(xcenter,ycenter),1,(0, 0, 255),2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        
                        
                        
                        
                        
                        detection_count += 1
                        for i,p in enumerate(polygons):
                            frame=p.render(frame)
                            if p.contains_point(point):
                               lane=i

                        label_text = f"ID:{track_id} lane:{lane}"
                        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 0, 255), -1,)
                        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                        #cv2.putText(frame,str(lane),(xcenter,ycenter-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)







            height, width, channels = frame.shape
            bytes_per_line = width * channels
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            
            self.frame_display.setPixmap(pixmap)


        except Exception as e:
            print(f"Error updating frame: {e}")
    


    def resizeEvent(self, event):
        current_pixmap = self.frame_display.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            self.frame_display.setPixmap(current_pixmap.scaled(
                self.frame_display.width(),
                self.frame_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        super().resizeEvent(event)
        
    def closeEvent(self, event):
        self.stop_streaming()
        super().closeEvent(event)

        



    def keyPressEvent(self, event: QKeyEvent):
        global polygons
        if(event.text().lower()=='a'):
            print(polygons)
            if not polygons: 
                polygons.append(Polygons(str(len(polygons)),"normal"))
            else:
                poly=polygons[len(polygons)-1]
                if len(poly.points) >=3:
                    print("polygon added")
                    polygons.append(Polygons(str(len(polygons)),"normal"))
                else:
                    print(polygons)
                    print("you can't add a new polygon unless you have at least 3 points, you have",len(poly.points))
                
        elif(event.text().lower()=='s'):
            if polygons:
                Polygons.save_polygons(polygons)
        elif(event.text().lower()=='l'):
            polygons=Polygons.load_polygons()



if __name__ == "__main__":
    Gst.init(None)
    app = QApplication(sys.argv)
    viewer = UI()
    viewer.show()
    sys.exit(app.exec_())