from ultralytics.models import YOLO

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO('../weights/bd/yolo11m.pt')

model.train(data='yolov11.yaml', epochs=1000, batch=32, imgsz=640, device = [0, 1, 4, 5], project="weights/trained/")