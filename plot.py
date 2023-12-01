import json
import sys
sys.path.append("/home/aistudio/external-libraries")
import matplotlib.pyplot as plt

from load_examples import NILMDataSet    

with open('./data/metadata.json', 'r') as file:
        info = json.load(file)
        
train_dataset = NILMDataSet(10, 'redd', info["partition"]['redd']['furnace'], "train")
val_dataset = NILMDataSet(10, 'redd', info["partition"]['redd']['furnace'], "val")
test_dataset = NILMDataSet(10, 'redd', info["partition"]['redd']['furnace'], "test")

for power, target in test_dataset:
    power = power[0,0].numpy()
    if len(target['labels']) == 0:
        continue
    plt.figure(figsize=(16, 10), dpi=300)
    plt.plot(power)
    for label, boxes in zip(target['labels'], target['boxes']):
        plt.axvline(boxes[0], color='lightcoral')
        plt.axvline(boxes[1], color='red')
        plt.text(boxes[1], 0.95 * max(power), str(label), color='red')
    plt.savefig(f"test/{target['idx']}.png", bbox_inches='tight')