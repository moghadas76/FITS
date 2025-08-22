from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, SCINet, Film, FITS, Real_FITS, Flow_FITS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.augmentations import augmentation
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'SCINet': SCINet,
            'Film': Film,
            'FITS': FITS,
            'Real_FITS': Real_FITS,
            'Flow_FITS': Flow_FITS,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        print('!!!!!!!!!!!!!!learning rate!!!!!!!!!!!!!!!')
        print(self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_profile(self, model):
        _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_xy = torch.cat([batch_x, batch_y], dim=1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'FITS' in self.args.model:
                    outputs, low = self.model(batch_x)
                elif 'SCINet' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, ft=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print(self.model)
        self._get_profile(self.model)
        print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            if self.args.in_dataset_augmentation:
                train_loader.dataset.regenerate_augmentation_data()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # print(batch_x.shape, batch_y.shape)
                batch_xy = torch.cat([batch_x, batch_y], dim=1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x)
                elif 'SCINet' in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                f_dim = -1 if self.args.features == 'MS' else 0
                # unified full-output tensor
                outputs_full = outputs[:, :, f_dim:]

                # Flow Matching loss (when selected)
                if getattr(self.args, 'loss', 'mse') == 'flow':
                    # choose region for FM: prediction horizon only or full sequence
                    if ft or getattr(self.args, 'flow_on_pred_only', False):
                        x0 = outputs_full[:, -self.args.pred_len:, :]
                        x1 = batch_y[:, -self.args.pred_len:, f_dim:]
                    else:
                        x0 = outputs_full
                        x1 = batch_xy[:, :, f_dim:]

                    # sample t in [t_min, t_max]
                    t_min = getattr(self.args, 'flow_t_min', 0.0)
                    t_max = getattr(self.args, 'flow_t_max', 1.0)
                    t = torch.rand(x0.size(0), 1, 1, device=x0.device) * (t_max - t_min) + t_min
                    x_t = (1.0 - t) * x0 + t * x1
                    v_target = x1 - x0
                    # call flow head
                    flow_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    if hasattr(flow_model, 'flow'):
                        v_pred = flow_model.flow(x_t, t)
                        loss = nn.functional.mse_loss(v_pred, v_target)
                    else:
                        # fallback to MSE if flow head not available
                        if ft:
                            loss = criterion(outputs_full[:, -self.args.pred_len:, :], x1)
                        else:
                            loss = criterion(outputs_full, x1)
                else:
                    if ft:
                        outputs_use = outputs_full[:, -self.args.pred_len:, :]
                        batch_y_use = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs_use, batch_y_use)
                    else:
                        loss = criterion(outputs_full, batch_xy)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        reconx = []
        inputxy = []
        reconxy = []
        lows = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_xy = torch.cat([batch_x, batch_y], dim=1).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                
                if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x)
                elif 'SCINet' in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs_ = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs_ = outputs_.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()


                pred = outputs_  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(test_data.inverse_transform(pred))
                trues.append(test_data.inverse_transform(true))
                inputx.append(batch_x.detach().cpu().numpy())
                inputxy.append(batch_xy.detach().cpu().numpy())
                reconx.append(outputs[:, :-self.args.pred_len, f_dim:].detach().cpu().numpy())
                reconxy.append(outputs.detach().cpu().numpy())
                # lows.append(low.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.array(inputx)
        # reconx = np.array(reconx)
        # reconxy = np.array(reconxy)
        # inputxy = np.array(inputxy)
        # lows = np.array(lows)


        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # reconx = reconx.reshape(-1, reconx.shape[-2], reconx.shape[-1])
        # reconxy = reconxy.reshape(-1, reconxy.shape[-2], reconxy.shape[-1])
        # inputxy = inputxy.reshape(-1, inputxy.shape[-2], inputxy.shape[-1])
        # lows = lows.reshape(-1, lows.shape[-2], lows.shape[-1])

        # try: 
        #     for i in range(0,2800,300):
                
        #         # create a figure with 3 subplots
        #         fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        #         # plot pred and true in the first subplot
        #         axs[0].plot(trues[i, :, -1], label='true')
        #         axs[0].plot(preds[i, :, -1], label='pred')
        #         axs[0].set_title('pred and true')
        #         # plot inputx and reconx in the second subplot
        #         axs[1].plot(inputx[i, :, -1], label='inputx')
        #         axs[1].plot(reconx[i, :, -1], label='reconx')
        #         axs[1].set_title('inputx and reconx')
        #         # plot inputxy and reconxy in the third subplot
        #         axs[2].plot(inputxy[i, :, -1], label='inputxy')
        #         axs[2].plot(reconxy[i, :, -1], label='reconxy')
        #         axs[2].plot(lows[i, :, -1])
        #         axs[2].set_title('inputxy and reconxy')
        #         # show the legend
        #         plt.legend()
        #         # save the figure to file
        #         fig.savefig(os.path.join(folder_path, str(i) + '_F.png'))
        #         # print('plottting')
        # except:
        #     pass

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
