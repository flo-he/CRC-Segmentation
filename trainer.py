import numpy as np
import os
import time
import torch
from dataloading import CV_Splits

### NEEDS TESTING ONCE U-NET ASSEMBLED
class Trainer(object):
    '''
    The U-Net trainer class. It wraps the main training process including dataloading, cross validation, and progress saving.
    '''

    def __init__(self, model, optimizer, criterion, dataset, device,
                 output_dir, epochs, batch_size, cv_folds, pin_mem, num_workers):
        
        # training "globals"
        self.OUTPUT_DIR = os.path.join(os.getcwd(), output_dir)
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.PIN_MEM = pin_mem
        self.NUM_WORKERS = num_workers
        self.DEVICE = device

        # dataset and cv sampler
        self.dataset = dataset
        self.cv_gen = CV_Splits(cv_folds, dataset)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # running val loss for computing average loss
        self.val_loss = 0.
        self.tr_loss = 0.

        # will hold average loss for each cv fold
        self.avg_val_losses = []
        self.avg_tr_losses = []
        
        # list for holding state dicts
        self.opt_state_dicts = []
        self.model_state_dicts = []
        

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
                    dataset,
                    batch_size = self.BATCH_SIZE,
                    shuffle = True,
                    num_workers = self.NUM_WORKERS,
                    pin_memory = self.PIN_MEM,
                    drop_last = True
                )

    def run_cv_iter(self, d_tr, d_val):
        '''
        Performs one cv iteration, i.e. trains on one of the possible combinations of the different cv folds.
        '''
        # train on splits for training
        for idx, (image, mask) in enumerate(d_tr):
            # get image batch and label batch
            image.to(self.DEVICE, non_blocking=self.PIN_MEM)
            mask.to(self.DEVICE, non_blocking=self.PIN_MEM)

            self.optimizer.zero_grad()

            # model output
            output = self.model(image)

            # image is not needed anymore, delete to overlap dataloading
            # (speedup if non_blocking=True)
            del image

            # loss
            loss = self.criterion(output, mask)
            del mask
            
            # backprop
            loss.backward()
            self.optimizer.step()
            self.tr_loss += loss.detach().item().cpu()  # may need .numpy() or smth like that
        
        # save average training loss
        self.avg_tr_losses.append(self.tr_loss / len(d_tr))
        self.tr_loss = 0.

        self.optimizer.zero_grad()  # necessary?

        # validate of left over fold
        for idx, (image, mask) in enumerate(d_val):
            # get image batch and label batch
            image.to(self.DEVICE, non_blocking=self.PIN_MEM)
            mask.to(self.DEVICE, non_blocking=self.PIN_MEM)

            with torch.no_grad():
                output = self.model(image)
                del image
                loss = self.criterion(output, mask)
                del mask
                self.val_loss += loss.item().cpu()  # may need .numpy() or smth like that

        # save average validation loss
        self.avg_val_losses.append(self.val_loss / len(d_val))
        self.val_loss = 0.

        # save model and optimizer state
        self.model_state_dicts.append(self.model.state_dict())
        self.opt_state_dicts.append(self.optimizer.state_dict())

    def cross_validate(self):
        '''
        Performs the cross validation over all cv folds of the current epoch.
        '''

        # save current state dict
        self.model_st = self.model.state_dict()
        self.opt_st = self.optimizer.state_dict()

        for (d_tr, d_val) in iter(self.cv_gen):
            train_loader = self.get_dataloader(d_tr)
            valid_loader = self.get_dataloader(d_val)

            # run cross validation iteration
            self.run_cv_iter(d_tr, d_val)

            # reset states for next iteration
            self.model.load_state_dict(self.model_st)
            self.optimizer.load_state_dict(self.opt_st)

    def run_training(self):
        '''
        Runs Training for specified amount of epochs. One Epoch is defined as a full sweep of the dataset using cross validation.
        At each epoch, we keep the model of the k-fold CV that produces the lowest validation loss.
        '''

        for epoch in range(1, self.EPOCHS+1):
            # measure time
            t_start = time.perf_counter()

            # run cross validation
            self.cross_validate()

            # keep model with smallest validation loss
            fold_idx = np.argmin(self.avg_val_losses)

            self.model.load_state_dict(self.model_state_dicts[fold_idx])
            self.optimizer.load_state_dict(self.optimizer_state_dicts[fold_idx])

            t_end = time.perf_counter() - t_start

            # print/log info (add logging later)
            print(self.avg_tr_losses)
            print(self.avg_val_losses)
            print(f"Epoch: {epoch} took {t_end}s | avg. tr_loss {self.avg_tr_losses[fold_idx]} | avg. val_loss: {self.avg_val_losses[fold_idx]}")

            self.avg_tr_losses.clear()
            self.avg_val_losses.clear()
            self.model_state_dicts.clear()
            self.optimizer_state_dicts.clear()

            # save model and optimizer state to disc
            if epoch % 5 == 0:
                print(f"Saving Progress to {self.OUTPUT_DIR}")
                self.save_progress(epoch)

    def save_progess(self, epoch):
        # filenames
        model_chkpt = os.path.join(self.OUTPUT_DIR, f"model_chkpt{epoch}")
        opt_chkpt = os.path.join(self.OUTPUT_DIR, f"optimizer_chkpt{epoch}")

        torch.save(self.model.state_dict(), model_chkpt)
        torch.save(self.optimizer.state_dict(), opt_chkpt)

    # maybe add early stopping 

