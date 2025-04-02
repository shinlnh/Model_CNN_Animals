
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Resize
from torchvision.transforms.v2 import Compose, ToPILImage
from PIL import Image
from sklearn.model_selection import train_test_split

class AnimalV2_Dataset(Dataset):
    def __init__(self, root, train=True,transform=None):
        self.images_path = []
        self.labels = []
        self.all_images_path = []
        self.all_labels = []
        self.root = root
        self.categories = os.listdir(self.root)
        self.transform = transform

        for index, category in enumerate(self.categories):
            category_path = os.path.join(root,category)
            file_names = os.listdir(category_path)
            for file_name in file_names:
                file_path = os.path.join(category_path,file_name)
                self.all_images_path.append(file_path)
                self.all_labels.append(index)
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.all_images_path, self.all_labels, test_size=0.2, random_state=42, stratify=self.all_labels
        )
        if train:
            self.images_path = train_paths
            self.labels = train_labels
        else:
            self.images_path = test_paths
            self.labels = test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image,label

if __name__ == '__main__':
    transform = Compose([
        Resize((32,32)),
        ToTensor()
    ])
    training_dataset = AnimalV2_Dataset(root="Animal_Dataset", train=True,transform=transform)

    # image,label = dataset.__getitem__(11234)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # print(label)
    # test = dataset.__len__()
    # print(test)

    training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    drop_last=True
    )

    # epochs = 10
    # for epoch in range(epochs):

    for images,labels in training_dataloader:
        print(images.shape)
        print(labels)
