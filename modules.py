import torch
import cv2

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_tensor, threshold, model):
    """Получает 1 кадр_тензор_торч размера [C,H,W]"""
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to('cuda')
        pred = model([img_tensor]) # Pass the image to the model
        del img_tensor
    mask = pred[0]['labels'] == 1 # маска, чтобы доставать чисто "людей"
    ######################################################
    labels = pred[0]['labels'][mask].cpu().detach().numpy()
    boxes = pred[0]['boxes'][mask].cpu().detach().numpy()
    scores = pred[0]['scores'][mask].cpu().detach().numpy()
    ######################################################
    # print(labels)
    # print(boxes)
    # print(scores)
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(labels)] # Get the Prediction Score
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(boxes)] # Bounding boxes
    pred_score = list(scores)
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    # print('pred_t', pred_t)
    # print(pred_score)
    # print(pred_boxes)
    # print(pred_class)
    return pred_boxes, pred_class

def object_detection_tracking(image, tracker, model, threshold=0.5, rect_th=2, text_size=1, text_th=1):
    """Получает 1 кадр cv2"""
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=torch.float32).permute(2,0,1)
    tensor_image /= 255.
    boxes, pred_cls = get_prediction(tensor_image, threshold, model) # Get predictions
    objects = tracker.update(boxes)
    for i in range(len(boxes)):
        b1 = (boxes[i][0], boxes[i][1])
        b2 = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, b1 , b2, color=(255, 0, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(image, pred_cls[i], b1,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,255,255), thickness=text_th) # Write the prediction class
    
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    return image   

def read_write_video(input_path, output_path, model, tracker=None):
    cap = cv2.VideoCapture(input_path)    
    fwidth, fheight  = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (fwidth, fheight))
    
    while(cap.isOpened()):
        check, frame = cap.read()
        if check:
            frame = object_detection_tracking(frame, tracker, model)
        else:
            break
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()