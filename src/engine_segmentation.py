from tqdm import tqdm
import torch
import config
import numpy as np
from utils.model_utils import save_model_checkpoint


def train_fn(model, data_loader, optimizer, loss_fn, save_model=False):
    model.train()
    fin_loss = 0
    model_outputs = []
    model_targets = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    for data in tk_iterator:
        # an item of the data is available as a dictionary
        for (key, value) in data.items():
            if key == "target_class":
                data[key] = value
            else:
                data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(**data)
            loss = loss_fn(out, data["mask"])
            loss.backward()
            optimizer.step()
            model_targets.extend(data["mask"].detach().cpu().numpy())
            model_outputs.extend(out.detach().cpu().numpy())
        fin_loss += loss.item()

    # if save_model:
    #     save_model_checkpoint(model, optimizer, loss, config.CHECKPOINT_PATH)

    return np.array(model_targets), np.array(model_outputs), fin_loss/len(data_loader)


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    fin_loss = 0
    model_outputs = []
    model_targets = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            for (key, value) in data.items():
                data[key] = value.to(config.DEVICE)
            out = model(**data)
            loss = loss_fn(out, data["targets"])
            model_outputs.extend(out.detach().cpu().numpy())
            model_targets.extend(data["targets"].detach().cpu().numpy())
            fin_loss += loss.item()

        return np.array(model_targets), np.array(model_outputs), fin_loss / len(data_loader)


def predict_fn(model, data_loader):
    model.eval()
    model_outputs = []
    # model_targets = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            out = model(data, torch.empty(1, 1))
            # loss = loss_fn(out, data["targets"])
            # model_outputs.extend(out.detach().cpu().numpy())
            output = out['out'][0]

    return output.argmax(0)