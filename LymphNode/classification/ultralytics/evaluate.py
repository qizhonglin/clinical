import os
import json
import logging 
from pathlib import Path
from LymphNode.config import DATA_ROOT as root, CHECKPOINT_DIR


def calc_accuracy(sag):
    correct = 0
    total = len(sag)
    for file, val in sag.items():
        gt = val['gt']
        pred = val['pred']
        conf = val['conf']
        if gt == pred:
            correct += 1
        else:
            logging.info(f"{file}: GT={gt}, Pred={pred}, Conf={conf:.4f}")
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    output_dir = os.path.join(CHECKPOINT_DIR, 'ultralytics/runs/classify/yolo11n-cls')
    json_file = os.path.join(output_dir, 'test.json')
    
    # Basic configuration: log to file and console
    logfile = os.path.join(output_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # optional: also print to console
        ]
    )   

    with open(json_file, 'r') as f:
        info = json.loads(f.read())
        
    info = {Path(file).name: val for file, val in info.items()}    
    benign = {file: val for file, val in info.items() if val['gt'] == 'benign'}
    malignant = {file: val for file, val in info.items() if val['gt'] == 'malignant'}
    
    benign_acc = calc_accuracy(benign)
    malignant_acc = calc_accuracy(malignant) 
    overall_acc = calc_accuracy(info)   
    result = {
        'benign_acc': benign_acc,
        'malignant_acc': malignant_acc,
        'overall_acc': overall_acc
    }
    logging.info(result)

 
    

if __name__ == '__main__':
    main()