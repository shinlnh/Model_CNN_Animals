import cv2
import numpy as np
from Dataset_Animals import AnimalV2_Dataset
from SCNN_Model_Animals import SimpleCNN
from torchvision.transforms import ToTensor, Resize,ColorJitter
from torchvision.transforms.v2 import Compose, ToPILImage, RandomAffine
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import shutil
import wandb
from torchvision import transforms
def get_args():
    parser = ArgumentParser(description="CNN Simple")
    parser.add_argument("--root", "-r", type=str, default="Animal_Dataset", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # num_epochs = 100
    # batch_size = 16
    args = get_args()
    RUN_ID = "single_run_scnn"

    if args.use_wandb:
        wandb.init(
            project="SimpleCNN_Project",
            name="SimpleCNN_OnlyOneRun",
            id=RUN_ID,
            resume="allow",
            config=vars(args)
        )
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    train_transform = Compose([
        RandomAffine(
          degrees=(-5,5),
          translate=(0.05,0.05),
          scale =(0.85,0.85),
          shear = 5
        ),
        Resize((args.image_size, args.image_size)),
        ColorJitter(

            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.25,
            hue = 0.05

        ),
        ToTensor()
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])

    training_dataset = AnimalV2_Dataset(root=args.root, train=True, transform=train_transform)

    image, _ = training_dataset.__getitem__(100)
    image = image.numpy()  # Chuyển từ Tensor → NumPy
    image = np.transpose(image, (1, 2, 0)) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow("test image",image)
    cv2.waitKey(0)
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        drop_last=True
    )

    testing_dataset = AnimalV2_Dataset(root=args.root, train=False, transform=test_transform)
    testing_dataloader = DataLoader(
        dataset=testing_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=False,
        drop_last=False
    )

    # Xóa thư mục tensorboard trước khi ghi log mới (đã bị loại bỏ)
    # if os.path.isdir(args.logging):
    #     shutil.rmtree(args.logging)

    # Tạo thư mục để lưu mô hình nếu chưa có
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    model = SimpleCNN(num_class=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Nếu có checkpoint, load lại trạng thái mô hình và optimizer
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0

        best_acc = 0
    num_iters = len(training_dataloader)
    # if torch.cuda.is_available():
    #     model.cuda()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(training_dataloader, colour="green")

        for iter, (images, labels) in enumerate(progress_bar):
            # if torch.cuda.is_available():
            #     images = images.cuda()
            #     labels = labels.cuda()

            # Forward
            prediction_train = model(images)
            loss_value = criterion(prediction_train, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(
                epoch + 1, args.epochs, iter + 1, num_iters, loss_value
            ))

            # Backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if args.use_wandb:
                wandb.log({"Train/Loss": loss_value.item()})
        # Evaluation after each epoch
        model.eval()
        all_prediction = []
        all_labels = []

        for iter, (images, labels) in enumerate(testing_dataloader):
            all_labels.extend(labels)
            with torch.no_grad():
                prediction_test = model(images)
                indices = torch.argmax(prediction_test.cpu(), dim=1)
                all_prediction.extend(indices)
                loss_value = criterion(prediction_test, labels)

        # Convert labels and predictions to scalar values
        all_labels = [label.item() for label in all_labels]
        all_prediction = [prediction.item() for prediction in all_prediction]

        # Print sau mỗi epoch
        # print("Epoch {}/{}".format(epoch + 1, args.epochs))
        # print(classification_report(all_labels, all_prediction,zero_division=1))
        accuracy = accuracy_score(all_labels, all_prediction)
        print("Epoch {}: Accuracy: {}".format(epoch +1, accuracy))

        if args.use_wandb:
            wandb.log({"val/accuracy": accuracy})

        # Lưu checkpoint sau mỗi epoch
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_SCNN.pt".format(args.trained_models))

        # Lưu mô hình tốt nhất nếu đạt accuracy cao nhất
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_SCNN.pt".format(args.trained_models))
            best_acc = accuracy
    if args.use_wandb:
        wandb.finish()