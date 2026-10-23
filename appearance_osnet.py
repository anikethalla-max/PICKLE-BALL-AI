"""Extract actual pretrained OSNet features for shared detection crops."""
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from third_party.osnet.osnet import osnet_x0_25


class OSNetFeatures:
    def __init__(self):
        torch.set_num_threads(4)
        weights=Path('models/osnet_x0_25_msmt17.pth')
        state=torch.load(weights,map_location='cpu',weights_only=True)
        state=state.get('state_dict',state)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        classes=state['classifier.weight'].shape[0]
        self.model=osnet_x0_25(num_classes=classes,pretrained=False).eval()
        self.model.load_state_dict(state,strict=True)
        self.transform=transforms.Compose([transforms.Resize((256,128)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __call__(self,frame,boxes):
        crops=[]
        h,w=frame.shape[:2]
        for x1,y1,x2,y2 in boxes:
            x1,y1,x2,y2=map(int,[max(0,x1),max(0,y1),min(w,x2),min(h,y2)])
            if x2<=x1 or y2<=y1:
                raise ValueError('Invalid OSNet crop')
            crops.append(self.transform(Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))))
        if not crops:
            return np.empty((0,512),np.float32)
        with torch.inference_mode():
            features=self.model(torch.stack(crops))
            features=torch.nn.functional.normalize(features,p=2,dim=1)
        return features.numpy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='data/occlusion_0633')
    args=parser.parse_args()
    root=Path(args.data_root)
    records=json.loads((root/'shared_detections.json').read_text())['frames']
    extractor=OSNetFeatures()
    output={}
    started=time.perf_counter()
    for row in records:
        index=row['frame']
        frame=cv2.imread(str(root/'frames'/f'frame_{row["source_frame"]:08d}.jpg'))
        boxes=np.array(row['detections'],np.float32).reshape(-1,5)[:,:4]
        output[f'f{index:03d}']=extractor(frame,boxes)
        if index%30==0:
            print('OSNet',index,flush=True)
    np.savez_compressed(root/'osnet_features.npz',**output)
    (root/'osnet_features_metadata.json').write_text(json.dumps({
        'architecture':'OSNet-x0.25','pretraining':'MSMT17, official Torchreid checkpoint',
        'preprocessing':'RGB, PIL bilinear resize 256x128, ImageNet mean/std',
        'normalization':'L2, cosine distance','feature_dimension':512,
        'seconds':time.perf_counter()-started,'checkpoint':'models/osnet_x0_25_msmt17.pth',
        'source':'third_party/osnet/PROVENANCE.json','frames':len(records)},indent=2))


if __name__=='__main__':
    main()
