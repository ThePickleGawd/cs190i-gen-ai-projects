import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config
from data import VOCDataset
from utils.yolov1_utils import cellboxes_to_boxes

def show_sample(dataset, idx, class_names=config.VOC_CLASSES):
    image, target = dataset[idx]  # image: CHW, target: (S, S, 30)
    image_np = image.permute(1, 2, 0).numpy()  # CHW -> HWC
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    # Convert label matrix to list of boxes
    target = target.unsqueeze(0)  # Add batch dim
    boxes = cellboxes_to_boxes(target)[0]  # Get first sample

    img_h, img_w = config.IMG_SIZE[1], config.IMG_SIZE[0]

    for box in boxes:
        class_pred, conf, x, y, w, h = box
        if conf < 1e-6:
            continue

        xmin = (x - w / 2) * img_w
        ymin = (y - h / 2) * img_h
        width = w * img_w
        height = h * img_h

        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, class_names[int(class_pred)], color='white',
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.savefig("sample_output.png", bbox_inches="tight")
    print("Saved to sample_output.png")
if __name__ == "__main__":
    train_ds = VOCDataset("train")
    show_sample(train_ds, 0)
