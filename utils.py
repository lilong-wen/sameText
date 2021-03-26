import cv2
import torch

def test_dataset_class(img, labels):

    for label_item in labels:

        cv2.rectangle(img, label_item[0][0], label_item[0][2], (255,0,0), 3)
        cv2.puttext(img, label_item[1][0], label_item[0][0], cv2.FONT_HERSHEY_SIMPLEX,
                    (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite('test.png', img)

def train_test_split(dataset, split_rate):

    train_num = int(len(dataset) * split_rate)

    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_num, len(dataset) - train_num]
    )

    return train_set, test_set
