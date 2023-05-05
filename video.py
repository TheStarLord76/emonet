import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import make_grid


from pathlib import Path
import argparse

from emonet.models import EmoNet
from emonet.data import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip


class LiveFrameDataset(Dataset):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.device = 'cuda:0'
    
    def __len__(self):
        return 1000000
    
    def __getitem__(self, idx):
        ret, frame = self.cap.read()
        if ret:
            img0 = frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
            frame = cv2.resize(frame, (256, 256))  # resize to (256, 256)

            # convert the image to the expected tensor format
            img = np.transpose(frame, (2, 0, 1))  # convert HWC to CHW format

            img = img.astype(np.float32) / 255.0  # scale the pixel values to the range [0.0, 1.0]

            # convert the numpy array to a PyTorch tensor
            img = torch.from_numpy(img).to(self.device)

            return img0, img
        else:
            return torch.zeros((1, 3, 224, 224))

        


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 16
device = 'cuda:0'
image_size = 256
subset = 'test'
metrics_valence_arousal = {'CCC':CCC, 'PCC':PCC, 'RMSE':RMSE, 'SAGR':SAGR}
metrics_expression = {'ACC':ACC}

# Create the live video dataset and dataloader
live_dataset = LiveFrameDataset()
live_dataloader = DataLoader(live_dataset)

# Load the pre-trained model
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

# Loop over the live video frames and obtain the emotion predictions
with torch.no_grad():
    for i, imgs in enumerate(live_dataloader):
        if imgs is not None:
            img0, img = imgs

            img0_np = img0.numpy()

            img0_np = np.squeeze(img0_np)

            pred = net(img)

            if pred:
                heatmap = pred['heatmap']
                # print('Heatmap:', heatmap)

                expression = pred['expression']
                print('Expression:', expression)

                expression_val = expression.cpu().numpy()[0]
                print('Expression val:', expression_val)

                valence = pred['valence']
                # print('Valence:', valence)

                valence_val = valence.cpu().numpy()[0]
                print('Valence val:', valence_val)
                
                arousal = pred['arousal']
                # print('Arousal:', arousal)

                arousal_val = arousal.cpu().numpy()[0]
                print('Arousal val:', arousal_val)

            cv2.imshow('Emonet', img0_np)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

print('Reached Here!')
        







# import cv2

# # Open the default camera
# cap = cv2.VideoCapture(0)

# # Check if camera opened successfully
# if not cap.isOpened():
#     print("Error opening video stream or file")

# # Read until video is completed
# while cap.isOpened():
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if ret:
#         # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         # Press Q on keyboard to exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()
