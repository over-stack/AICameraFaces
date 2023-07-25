import torch
from ultralytics import YOLO


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load a model
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
    model.to(device)

    # Use the model
    model.train(data="data/data.yaml", epochs=8, batch=8, val=False)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    main()
