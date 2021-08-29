import sys, os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_device(computer="gpu", verbose=False):
    if computer == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            if verbose:
                print ('Current cuda device: ', torch.cuda.current_device())
    else:
        device = torch.device("cpu")
        if verbose:
            print ('Current device: cpu')

    return device

def r2_score(pred, true):
    if len(pred) == 1:
        return 0
    mean = np.mean(true)
    SS_res = np.sum((true - pred)**2)
    SS_tot = np.sum((true - mean)**2)
    r2 = (1 - SS_res/SS_tot)
    return r2

def MSE(pred, true):
    squared_err = np.sum((pred - true)**2)/len(pred)
    return squared_err

def test_model(model, device, criterion, test_loader, plot_predictions=True, title=None, predictor = "yield", savefig=None, verbose=True):
    num_batches = 0
    model.eval()
    true_list = []
    pred_list = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            true_list.extend(y.cpu().numpy())
            pred_list.extend(pred.cpu().numpy())
            num_batches += 1


    true_list = np.asarray(true_list)
    pred_list = np.asarray(pred_list)

    loss = MSE(pred_list, true_list)
    r2 = r2_score(pred_list, true_list)

    if verbose:
        print(f"{title:10s}: MSE: {loss:.3e}, R2: {r2:.3e}")

    if predictor == "yield":
        std = 0.126
    elif predictor == "residual":
        std = 1.01
    else:
        std = 0

    if plot_predictions:
        x0 = np.min(true_list)
        x1 = np.max(true_list)
        y0 = np.min(pred_list)
        y1 = np.max(pred_list)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(true_list, pred_list, c="b", alpha=0.5)
        ax.plot((x0, x1), (x0, x1), "k--")

        ax.plot((x0, x1), (x0+std, x1+std), "r--")
        ax.plot((x0, x1), (x0-std, x1-std), "r--", label=f"Std. of {predictor} stress in equal pore configurations: {std:.2f}")


        ax.set_xlabel(f"True stress [GPa]")
        ax.set_ylabel(f"Predicted stress [GPa]")
        ax.legend()
        if title is not None:
            ax.set_title(f"{title}\nMSE:{loss:.2e}, R2:{r2:.2f}")
        else:
            ax.set_title(f"MSE:{loss:.2e}, R2:{r2:.2f}")

        if savefig is not None:
            plt.savefig(savefig)

    return true_list, pred_list

def train_model(model,
                device,
                train_loader,
                val_loader,
                epochs,
                criterion,
                optimizer,
                scheduler = None,
                save_model_path = None,
                train_r2_criterion = None,
                verbose=False):

    if save_model_path is not None:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

    loss_val_list = []
    loss_train_list = []
    r2_val_list = []
    r2_train_list = []
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)

    best_model_state = None
    best_r2 = np.NINF
    for epoch in pbar:
        cum_loss = 0
        cum_r2 = 0
        num_batch_train = 0
        num_batch_val = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = criterion(pred, y)
            with torch.set_grad_enabled(False):
                r2 = r2_score(pred.cpu().numpy(), y.cpu().numpy(), False)
                cum_r2 += r2

            # Accumulate errors
            cum_loss += loss.item()
            num_batch_train += 1

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        cum_loss_val = 0
        cum_r2_val = 0

        if val_loader is not None:
            with torch.set_grad_enabled(False):
                for _batch, (X_val, y_val) in enumerate(val_loader):
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    pred_val = model(X_val)
                    loss_val = criterion(pred_val, y_val)
                    r2_val = r2_score(pred_val.cpu().numpy(), y_val.cpu().numpy(), debug=False)

                    cum_loss_val += loss_val.item()
                    cum_r2_val += r2_val
                    num_batch_val += 1

        if scheduler is not None:
            scheduler.step()

        cum_loss /= num_batch_train
        cum_r2 /= num_batch_train
        if val_loader is not None:
            cum_loss_val /= num_batch_val
            cum_r2_val /= num_batch_val

        loss_train_list.append(cum_loss)
        r2_train_list.append(cum_r2)
        if val_loader is not None:
            loss_val_list.append(cum_loss_val)
            r2_val_list.append(cum_r2_val)

        if train_r2_criterion is not None:
            if cum_r2 <= train_r2_criterion and cum_r2 >= train_r2_criterion-0.1:
                if cum_r2_val > best_r2:
                    best_r2 = cum_r2_val
                    best_model_state = model.state_dict()
                    best_idx = epoch
        elif (cum_r2_val > best_r2):
            best_r2 = cum_r2_val
            best_model_state = model.state_dict()

        if verbose:
            if val_loader is not None:
                pbar_items = {
                    "loss_train": f"{cum_loss:.2e}",
                    "r2_train": f"{cum_r2:.2e}",
                    "loss_val": f"{cum_loss_val:.2e}",
                    "r2_val": f"{cum_r2_val:.2e}"
                }
            else:
                pbar_items = {
                    "loss_train": f"{cum_loss:.2e}",
                    "r2_train": f"{cum_r2:.2e}"
                }
            pbar.set_postfix(pbar_items)

        #save model
        if save_model_path is not None:
            torch.save(model.state_dict(), os.path.join(save_model_path, f"epoch_{epoch}.pth"))

    if verbose:
        print("Loss train:", loss_train_list[-1])
        print("R2 train:", r2_train_list[-1])
        if val_loader is not None:
            print("Loss val:", loss_val_list[-1])
            print("R2 val:", r2_val_list[-1])

    if val_loader is not None:
        history = {
            "loss_train": loss_train_list,
            "loss_val": loss_val_list,
            "r2_train": r2_train_list,
            "r2_val": r2_val_list
        }
    else:
        history = {
            "loss_train": loss_train_list,
            "r2_train": r2_train_list,
        }

    return history, best_model_state


def plot_history(history, savefig=None, train_data_only=False):

    loss_train = history['loss_train']
    r2_train = history['r2_train']
    train_idx = np.argmax(r2_train)


    if train_data_only:
        fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
        print(f"best train epoch: {train_idx}, mse: {loss_train[train_idx]}  r2: {r2_train[train_idx]}")
        ax[0].semilogy(loss_train, label=f"best train: {loss_train[train_idx]:.2e}")
        # ax.plot(r2_train, )
        ax[0].legend(loc='best')
        ax[0].set_ylabel("Mean-squared error loss")
        ax[0].set_xlabel("epochs")
        ax[0].legend(loc='best')

        ax[1].plot(r2_train, label=f"best train: {r2_train[train_idx]:.2e}")
        ax[1].set_ylim(-1, 1)
        ax[1].legend(loc='best')
        ax[1].set_ylabel("R2-score")
        ax[1].set_xlabel("epochs")
        ax[1].legend(loc='best')

        return

    loss_val = history['loss_val']
    r2_val = history['r2_val']

    val_idx = np.argmax(r2_val)

    print(f"best val, epoch {val_idx}, r2: {r2_val[val_idx]}")

    fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
    ax[0].semilogy(loss_train, label=f"best train: {loss_train[train_idx]:.2e}")
    ax[0].semilogy(loss_val, label=f"best val: {loss_val[val_idx]:.2e}")
    ax[0].legend(loc='best')
    ax[0].set_ylabel("Mean-squared error loss")


    ax[1].plot(r2_train, label=f"best train: {r2_train[train_idx]:.2e}")
    ax[1].plot(r2_val, label=f"best val: {r2_val[val_idx]:.2e}")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("R2-score")
    ax[1].set_ylim([-2, 1.1])
    ax[1].legend(loc="best")

    if savefig is not None:
        plt.savefig(savefig)
