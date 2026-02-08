from ultralytics import YOLO

model_state = YOLO("yolo11s-cls.pt")
model_state.train(
    data="yolo_cls",
    imgsz=224,
    epochs=20,
    batch=16,
    workers=4,
    device=0,
    project="runs",
    name="state_yolo11n_cls",
)

model_count = YOLO("yolo11n-cls.pt")
model_count.train(
    data="count_cls",
    imgsz=320,
    epochs=20,
    batch=16,
    workers=4,
    device=0,
    project="runs",
    name="count_yolo11n_cls",
    val=False,
)

