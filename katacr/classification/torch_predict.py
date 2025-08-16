from katacr.classification.train import TrainConfig, ModelConfig
from katacr.classification.torch_train import DatasetBuilder, get_args_and_writer, ResNet
from katacr.build_dataset.utils.split_part import extract_bbox
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np



args, writer = get_args_and_writer()
## Define local path_dataset_root
path_dataset_root = Path(r'C:\Disk_D\RL_Finance\ImageLabeling')
ds_builder=DatasetBuilder(str(path_dataset_root / "images/shawn_card_classification"), args.seed)
args.num_class = len(ds_builder.card_list)
train_cfg = TrainConfig(**vars(args))
model_cfg = ModelConfig(**vars(args))


class CardClassifier:
    def __init__(self, model_path, args):
        # Load config: image size, idx2card, card2idx
        self.idx2card = ds_builder.idx2card
        self.card2idx = ds_builder.card2idx
        model_cfg = ModelConfig(**vars(args))
        train_cfg = TrainConfig(**vars(args))
        self.img_size = train_cfg.image_size

        # Load trained PyTorch model
        self.model =ResNet(model_cfg)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def preprocess(self, imgs):
        processed = []
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if gray.shape[:2][::-1] != self.img_size:
                gray = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_CUBIC)
            gray = gray.astype(np.float32) / 255.
            gray = gray[None, :, :]  # (1, H, W)
            processed.append(gray)
        x = np.stack(processed)  # (B, 1, H, W)
        return torch.tensor(x, dtype=torch.float32)

    def __call__(self, x, keepdim=False, cvt_label=True, verbose=False):
        if not isinstance(x, list): x = [x]
        x_tensor = self.preprocess(x)  # (B, 1, H, W)

        with torch.no_grad():
            logits = self.model(x_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        if cvt_label:
            labels = [self.idx2card[i] for i in preds]
        else:
            labels = preds

        if verbose:
            print("Logits:", logits[0], "Prediction:", labels)
            cv2.imshow("img", x[0])
            cv2.waitKey(0)

        if len(x) == 1 and not keepdim:
            return labels[0]
        return labels
    

    def process_part3(self, x: np.ndarray, pil=False, cvt_label=True, verbose=False):
        """
        Args:
        x (np.ndarray): The part3 split by katacr/build_dataset/utils/split_part.py process_part(),
            only accept one image (x.ndim==3).
        pil (bool): If taggled, the image `x` is RGB format.
        cvt_label (bool): If taggled, the classification index will be converted to label name.
        verbose (bool): If taggled, each card image will be showed.
        Returns:
        result (List[str]): Detection name for each cards: next card, card1, card2, card3, card4
        """
        part3_bbox_params = [  # Configure for card positions in part3, card position is left to right
                            (0.047, 0.63, 0.100, 0.365),  # next card
                            (0.222, 0.100, 0.185, 0.745),  # card1
                            (0.410, 0.100, 0.185, 0.745),  # card2
                            (0.600, 0.100, 0.185, 0.745),  # card3
                            (0.785, 0.100, 0.185, 0.745),  # card4
                            ]
        if not pil: x = x[...,::-1]
        params = part3_bbox_params
        results = []
        for param in params:
            img = extract_bbox(x, *param)  # xywh for next image position
            #cv2.imshow("card", img[...,::-1])
            #cv2.waitKey(0)
            results.append(self(img, cvt_label=cvt_label, verbose=verbose))
        return results




'''
Example for making inference using CardClassifier with a single image:



predictor=CardClassifier("D:\RL_Finance\KataCR\card_resnet.pth", args)

card_selection_pic = "D:\\RL_Finance\\ImageLabeling\\images\\shawn_20250523_card_tables\\frame_0109.jpg"
img=cv2.imread(card_selection_pic)
pred = predictor.process_part3(img, pil=False)
print(pred)
cv2.imshow("pred", img)
cv2.waitKey(0)


'''