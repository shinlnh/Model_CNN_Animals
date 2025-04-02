from argparse import ArgumentParser
from SCNN_Model_Animals import SimpleCNN
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchsummary import summary
def get_args():
    parser = ArgumentParser(description="CNN Simple")
    parser.add_argument("--image-path", "-p", type=str, default=None, help="Image path")
    # parser.add_argument("--root", "-r", type=str, default="Animal_Dataset", help="Root of the dataset")
    # parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    # parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    # parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_SCNN.pt")
    # parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    categories = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squirrel"]
    args = get_args()
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    model = SimpleCNN(num_class=10)
    summary(model, (3, 224, 224))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint found")
        exit(0)

    model.eval()

    #Processing Image

    ori_image = cv2.imread(args.image_path) # = Image.open PIL
    image = cv2.cvtColor(ori_image,cv2.COLOR_BGR2RGB) # = convert RGB ( trong PIL, các ảnh đều được đưa về RGB khi image.open
    # Tuy nhiên, ảnh đen trắng sẽ không được chuyển. Do vậy, vẫn cần có convert("RGB")
    image = cv2.resize(image,(args.image_size,args.image_size)) # = Resize của pytorch
    image = np.transpose(image,(2,0,1))/255.0 # = ToTenSor của pytorch, chuyển HWC thành CWH, sau đó đưa về (0,1)
    image = image[None, :, :, :] # Giả lập chiều thứ 4 batch_size trước khi đưa vào mô hình
    image = torch.from_numpy(image).float() # Biến numpy array thành tensor trước khi đưa vào mô hình. Bước này được tích hợp trong ToTenSor
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)

        probs = softmax(output)


    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    print("The test image is about {} with confident score of {}".format(predicted_class,probs[0,max_idx]))
    cv2.imshow("{}: {:.2f}%".format(predicted_class, probs[0, max_idx]*100), ori_image)

    cv2.waitKey(0)