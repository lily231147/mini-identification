import json
import sys
sys.path.append("/home/aistudio/external-libraries")
from read_data import NILMDataset, get_datasets
import numpy as np

from pathlib import Path
import click
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import torch
from torch.utils.data import DataLoader
from sl.sl_net import SlConfig, SlNet
from gen.gen_net import GenConfig, GenNet
from yolo.yolo_net import YoloNet, YoloConfig
from utils import cal_metrics, Metric, filter_pred

def train_one_appliance(set_name, app_name, method, info, config):
    app_idx = info['app_ids'][set_name][app_name]
    n_class = info['n_class'][set_name][app_name]
    amplitude_threshold = info["amplitude_threshold"][set_name][app_name]
    stable_threshold = info["stable_threshold"][set_name][app_name]
    # 设备的数据主目录，eg: ./data/ukdale/data8
    # data_dir = Path(f"data/source/{set_name}/data{app_idx}")
    # interval_path =  data_dir / "ImageSets" / "interval.txt"
    set_dir = Path(f"nilm_events_simple/{set_name}")
    intervals = np.loadtxt(set_dir / f"channel_{str(app_idx)}" / "interval.txt")
    print("当前设备是：", app_name, "设备号是：", app_idx, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # get data loaders
    # train_loader, val_loader, test_loader = get_loaders(app_idx, set_name, config.batch_size, partition)
    train_set, val_set, _ = get_datasets(set_dir, set_name, app_idx)
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=NILMDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=NILMDataset.collate_fn)

    if method == "sl":
        model = SlNet(config.in_channels, config.out_channels, config.length, 
                      n_class, config.label_method, config.backbone).to(device)
        checkpoint_dir = Path("weights") / method / config.label_method / config.backbone
        case_dir = Path("case") / method / config.label_method / config.backbone / f"{set_name}_{app_name}"
    elif method == "yolo":
        model = YoloNet(config.in_channels, config.out_channels, config.length, 
                        num_classes=n_class, backbone=config.backbone).to(device)
        checkpoint_dir = Path("weights") / method / config.backbone
        case_dir = Path("case") / method / config.backbone / f"{set_name}_{app_name}"
    elif method == "gen":
        model = GenNet(400, n_class).to(device)
        checkpoint_dir = Path("./weights") / method
        case_dir = Path()
    else:
        raise ValueError(f"{method} must in models `seq_label`, `yolo` or `transformer`")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{set_name}_{app_name}.pth"
    case_dir.mkdir(parents=True, exist_ok=True)
    # optimizer, scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_drop, gamma=config.gama)
    # training init
    start_epoch = 0
    f1_best = -1
    epoch_best = -1
    if config.load_his:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        f1_best = checkpoint['f1_best']
        epoch_best = checkpoint['epoch']
    # training
    with Progress(
            TextColumn("[progress.description]{task.description}", justify='right'),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            TextColumn("[progress.percentage]{task.percentage:>.2f}"), 
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[metrics]}"), ) as progress:
        epoch_task = progress.add_task(f"{set_name}/{app_name}", completed=start_epoch, total=config.epochs, metrics='')
        train_task = progress.add_task("train", total=len(train_loader), metrics='')
        val_task = progress.add_task("val", total=len(val_loader), metrics='')
        for epoch in range(start_epoch, config.epochs):
            # train
            progress.reset(train_task)
            model.train()
            sum_loss = 0
            for images, targets, _ in train_loader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss = model(images, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss
                progress.advance(train_task)
            scheduler.step()
            # validation
            progress.reset(val_task)
            model.eval()
            with torch.no_grad():
                val_metric = Metric()
                for images, targets, stamps in val_loader:
                    images = images.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    pred = model(images, targets)
                    pred = filter_pred(app_idx, pred, images[:, 0, :], stamps, intervals, amplitude_threshold, stable_threshold)
                    tp, fp, fn = cal_metrics(pred, targets, images[:, 0, :], case_dir)
                    val_metric.add(tp, fp, fn)
                    progress.advance(val_task)
                _, _, f1 = val_metric.get_index()
            if f1 > f1_best:
                f1_best = f1
                epoch_best = epoch
                save_files = {'model': model.state_dict(), 'epoch': epoch, 'f1_best': f1_best,
                              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                torch.save(save_files, checkpoint_path)
            loss_avg = sum_loss / len(train_loader)
            metrics_str = f"loss-{loss_avg: .5f}, tp-{val_metric.tp}, fp-{val_metric.fp}, fn-{val_metric.fn}, " \
                          f"f1-{f1:.2f}, f1_best-{f1_best: .2f}/{epoch_best}"
            progress.update(epoch_task, advance=1, metrics=metrics_str)


def set_app_choices(ctx, param, value):
    if value == 'ukdale':
        ctx.command.params[1].type = click.Choice(['kettle', 'rice-cooker', 'microwave', 'all'])
    elif value == 'redd':
        ctx.command.params[1].type = click.Choice(['furnace', 'washer-dryer', 'microwave', 'all'])
    else:
        ctx.command.params[1].type = click.Choice(['all'])
    return value


@click.command()
@click.option('--dataset', type=click.Choice(['ukdale', 'redd', 'all']), prompt="DataSet Name", callback=set_app_choices)
@click.option('--app', type=click.Choice(['kettle', 'rice-cooker', 'microwave', 'furnace', 'washer-dryer', 'all']), prompt="Appliance Name")
@click.option('--method', type=click.Choice(['sl', 'yolo', 'transformer']), default='sl')
def train(dataset, app, method):
    with open('nilm_events_simple/metadata.json', 'r') as file:
        info = json.load(file)
    if method == 'sl':
        config = SlConfig(batch_size=4096)
    elif method == 'yolo':
        config = YoloConfig(batch_size=2048)
    else:
        config = GenConfig()
    if dataset == 'all':
        # train all appliances in all datasets
        for app_name in ('kettle', 'rice-cooker', 'microwave'):
            train_one_appliance('ukdale', app_name, method, info, config)
        for app_name in ('furnace', 'washer-dryer', 'microwave'):
            train_one_appliance('redd', app_name, method, info, config)
    elif dataset == 'ukdale' and app == 'all':
        # train all appliances in ukdale
        for app_name in ('kettle', 'rice-cooker', 'microwave'):
            train_one_appliance('ukdale', app_name, method, info, config)
    elif dataset == 'redd' and app == 'all':
        # train all appliances in redd
        for app_name in ('furnace', 'washer-dryer', 'microwave'):
            train_one_appliance('redd', app_name, method, info, config)
    else:
        # train specific appliance in specific dataset
        train_one_appliance(dataset, app, method, info, config)

if __name__ == "__main__":
    train()
