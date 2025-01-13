from ultralytics.models import YOLOv10

model = YOLOv10('weights/bd_01131614.pt')

model.train(data='train_bd/yolov10.yaml', epochs=500, batch=256, imgsz=640)