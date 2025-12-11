
import os 
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.append('/home/qzlin/Documents/clinical')
from LymphNode.config import DATA_ROOT as root, CHECKPOINT_DIR



def train(data_dir, project_dir, pretrain_model_file, device=[1]):
    model = YOLO(model=pretrain_model_file, task="classify")
    model.train(data=data_dir,
                epochs=50, imgsz=640, device=device,
                project=project_dir,
                amp=False,
                batch=-1,
                )
    
def infer(test_dir, project_dir):
    best_model = YOLO(model=os.path.join(project_dir, 'train/weights/best.pt'), task="classify")
    
    info = {}
    for user in os.listdir(test_dir):
        user_dir = os.path.join(test_dir, user)
        
        for img_file in tqdm(os.listdir(user_dir)):
            img_file =  os.path.join(user_dir, img_file)

            results = best_model.predict(img_file)

            info[img_file] = {
                'gt': user
            }
            for result in results:
                cls_pred = result.probs.top1
                conf = result.probs.top1conf.item()
                info[img_file]['pred'] = best_model.names[cls_pred]
                info[img_file]['conf'] = conf
                
    print(info)
    with open(os.path.join(project_dir, 'test.json'), 'w') as f:
        json.dump(info, f, indent=4)      


def main():
    data_dir = os.path.join(root, 'experiments')
    
    pretrain_model_file = os.path.join("/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df1/models/public/ultralytics", "yolo11n-cls.pt")
    
    model_name = Path(pretrain_model_file).name.replace(".pt", "")
    project_dir = os.path.join(CHECKPOINT_DIR, 'ultralytics', f'runs/classify/{model_name}')
    
    train(data_dir, project_dir, pretrain_model_file)
    
    test_dir = os.path.join(data_dir, 'test')
    infer(test_dir, project_dir)


if __name__ == '__main__':
    main()