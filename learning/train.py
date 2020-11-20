"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import csv
import time
import random
import numpy as np
import json
import os
import sys
import math
import argparse
import ast
from tqdm import tqdm
import logging
from collections import defaultdict
import h5py
import cloudIO as io
#from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchnet as tnt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH,'..'))

from learning import spg
from learning import graphnet
from learning import pointnet
from learning import metrics

sys.path.append("./utils")

from colorLabelManager import ColorLabelManager
from pathManager import PathManager
from confusionMatrix import ConfusionMatrix
from timer import Timer

def main(args):
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    parser.add_argument('ROOT_PATH', help='name of the folder containing the data directory')
    # parser.add_argument('MODEL_NAME', help='Loads a pretrained model.')

    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[30, 50, 70, 100, 200]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    parser.add_argument('--loss_weights', default='none', help='[none, proportional, sqrt] how to weight the loss function')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--test_train_nth_epoch', default=1, type=int, help='Test each n-th epoch during training on training data')
    parser.add_argument('--test_valid_nth_epoch', default=1, type=int, help='Test each n-th epoch during training on validation data')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int, help='Average logits obtained over runs with different seeds')
    parser.add_argument('--not_only_best', action='store_true', help='Keep the model even if the result is worst')


    # Dataset
    parser.add_argument('--dataset', default='sema3d', help='Dataset name: sema3d|s3dis')
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--db_train_name', default='train')
    parser.add_argument('--db_test_name', default='test')
    parser.add_argument('--use_val_set', type=int, default=0)
    parser.add_argument('--use_pyg', default=0, type=int, help='Wether to use Pytorch Geometric for graph convolutions')
    #parser.add_argument('--resume', default='', help='Loads a previously saved model.')

    # Model
    # Define recurrent network module nature GRU/LSTM, the number of iterations and the number of features
    parser.add_argument('--model_config', default='gru_11,f_11', help='Defines the model as a sequence of layers, see graphnet.py for definitions of respective layers and acceptable arguments. In short: rectype_repeats_mv_layernorm_ingate_concat, with rectype the type of recurrent unit [gru/crf/lstm], repeats the number of message passing iterations, mv (default True) the use of matrix-vector (mv) instead vector-vector (vv) edge filters, layernorm (default True) the use of layernorms in the recurrent units, ingate (default True) the use of input gating, concat (default True) the use of state concatenation')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d', help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='xyzrgbelpsvXYZ', help='Point attributes fed to PointNets, if empty then all possible. xyz = coordinates, rgb = color, e = elevation, lpsv = geometric feature, d = distance to center')
    # Data augmentation for each superpoint
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_xyznormalize', default=1, type=int, help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[32,128,64]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_bnidx', default=2, type=int, help='Layer index to insert batchnorm to. -1=do not insert.')
    parser.add_argument('--edge_mem_limit', default=30000, type=int, help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')

    # Superpoint graph subsample process
    parser.add_argument('--spg_attribs01', default=1, type=int, help='Bool, normalize edge features to 0 mean 1 deviation')
    # Superpoint graph is subsampled randomly 
    # Small subgraphs of order 3 are concatenate
    parser.add_argument('--spg_augm_order', default=3, type=int, help='Order of neighborhoods to sample in SPG')
    # A total of 100 subgraph of order 3 are concatenate
    parser.add_argument('--spg_augm_nneigh', default=100, type=int, help='Number of neighborhoods to sample in SPG')
    # The final graph should be with a maximum number of 512 nodes 
    parser.add_argument('--spg_augm_hardcutoff', default=512, type=int, help='Maximum number of superpoints larger than args.ptn_minpts to sample in SPG')
    parser.add_argument('--spg_superedge_cutoff', default=-1, type=float, help='Artificially constrained maximum length of superedge, -1=do not constrain')

    # Point net
    # All superpoints with less than this amount of point rely solely on contextual classification
    parser.add_argument('--ptn_minpts', default=40, type=int, help='Minimum number of points in a superpoint for computing its embedding.')
    # All superpoints are resampled to this size for pointnet
    parser.add_argument('--ptn_npts', default=128, type=int, help='Number of input points for PointNet.')
    # Pointnet layer configuration
    parser.add_argument('--ptn_widths', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[64,64,128], [128,64]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--ptn_nfeat_stn', default=11, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_mem_monger', default=1, type=int, help='Bool, save GPU memory by recomputing PointNets in back propagation.')

    # Decoder
    parser.add_argument('--sp_decoder_config', default="[]", type=str,
                        help='Size of the decoder : sp_embedding -> sp_class. First layer of size sp_embed (* (1+n_ecc_iteration) if concatenation) and last layer is n_classes')

    parser.add_argument('--resume', action='store_true', help='Resume the model')

    args = parser.parse_args(args)
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.sp_decoder_config = ast.literal_eval(args.sp_decoder_config)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)

    pathManager = PathManager(args.ROOT_PATH)
    
    " Dirty fix in order to keep arguments coherent in the rest of the code "
    " Need refactor but args is propagate all hover the code "
    args.CUSTOM_SET_PATH = pathManager.rootPath
    args.supervized = 0

    set_seed(args.seed, args.cuda)
    logging.getLogger().setLevel(logging.INFO)  #set to logging.DEBUG to allow for more prints

    torch.backends.cudnn.enabled = False
    #if args.use_pyg:
    #    torch.backends.cudnn.enabled = False

    import custom_dataset
    dbinfo = custom_dataset.get_info(args)
    print("Read dataset")
    create_dataset = custom_dataset.get_datasets

    # Create model and optimizer
    if args.resume :
        model, optimizer = resume(args, dbinfo, pathManager.modelFile)
    else:
        print("Setup CUDA model")
        model = create_model(args, dbinfo)
        optimizer = create_optimizer(args, model)

    train_dataset, test_dataset, valid_dataset, scaler = create_dataset(args)

    print('Train dataset: %i elements - Test dataset: %i elements - Validation dataset: %i elements' % (len(train_dataset),len(test_dataset),len(valid_dataset)))
    ptnCloudEmbedder = pointnet.CloudEmbedder(args)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch-1)


    ############
    def train():
        """ Trains for one epoch """
        model.train()

        " collate_fn = function called on each sample per batch in order to concatenate them into a single batch "
        " batch_size = number of sample i.e spg, per batch, default value is 2 "
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=spg.eccpc_collate, num_workers=args.nworkers, shuffle=True, drop_last=True)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

        # tnt.meter.AverageValueMeter() --> just average a value
        loss_meter = tnt.meter.AverageValueMeter()

        # Official confusion matrix from semantic3D
        #officialConfusionMatrix = metrics.ConfusionMatrix(dbinfo['classes'])

        # Custom confusion matrix
        confusionMatrix = ConfusionMatrix(dbinfo['classes'])
        confusionMatrixSpp = ConfusionMatrix(dbinfo['classes'])

        timer = Timer(2)
        timer.start(0)

        # iterate over dataset in batches
        for targets, GIs, clouds_data in loader:
            timer.stop(0)

            model.ecc.set_info(GIs, args.cuda)
            " Here label_mode is the ground truth, stored on the first column of targets"
            " label_vec_cpu is how many points of each class has been classified, usefull later for confusion matrix, start from 2: cause 1: is unknown label "
            " segm_size_cpu is the total amount of points on the point cloud, unknown included "
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1)
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()

            optimizer.zero_grad()
            timer.start(1)

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)
            
            loss = nn.functional.cross_entropy(outputs, Variable(label_mode), weight=dbinfo["class_weights"])

            loss.backward()
            ptnCloudEmbedder.bw_hook()

            if args.grad_clip>0:
                for p in model.parameters():
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            optimizer.step()

            timer.stop(1)
            loss_meter.add(loss.item()) # pytorch 0.4

            " Remove all nodes without ground truth "
            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())

            for i, label in enumerate(np.argmax(o_cpu, axis=1)):
                confusionMatrix.addBatchPredictionVec(label, tvec_cpu[i, :])
            confusionMatrixSpp.addPrediction(np.argmax(o_cpu, axis=1), t_cpu)
            
            #officialConfusionMatrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))
            # officialConfusionMatrix.get_average_intersection_union() == confusionMatrix.getAvgIoU()

            #print(timer.getFormattedTimer(["Loader", "Trainer"]))
            timer.reset()
            timer.start(0)

        #acc, loss, oacc, avg_iou
        # acc = acc on spp
        # oacc = acc on point
        # avg_iou = iou on point
        return loss_meter.value()[0], confusionMatrix, confusionMatrixSpp

    ############
    def eval(is_valid = False):
        """ Evaluated model on test set """
        model.eval()
        
        if is_valid: # validation dataset
            loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
        else: # test dataset
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
            
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

        loss_meter = tnt.meter.AverageValueMeter()

        #confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'])
        confusionMatrix = ConfusionMatrix(dbinfo['classes'])
        confusionMatrixSpp = ConfusionMatrix(dbinfo['classes'])

        # DOOOOC
        # tvec_cpu = groundtruth = [0:2000] with in each case [0:9] number of point for each label
        # o_cpu    = predictions = [0:2000] one value for each prediction
        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)
            
            loss = nn.functional.cross_entropy(outputs, Variable(label_mode), weight=dbinfo["class_weights"])
            loss_meter.add(loss.item()) 

            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            if t_cpu.size > 0:
                confusionMatrixSpp.addPrediction(np.argmax(o_cpu, axis=1), t_cpu)
                for i, label in enumerate(np.argmax(o_cpu, axis=1)):
                    confusionMatrix.addBatchPredictionVec(label, tvec_cpu[i, :])

        return loss_meter.value()[0], confusionMatrix, confusionMatrixSpp

    ############
    def eval_final():
        """ Evaluated model on test set in an extended way: computes estimates over multiple samples of point clouds and stores predictions """
        model.eval()

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'])
        collected, predictions = defaultdict(list), {}

        # collect predictions over multiple sampling seeds
        for ss in range(args.test_multisamp_n):
            test_dataset_ss = create_dataset(args, ss)[1]
            loader = torch.utils.data.DataLoader(test_dataset_ss, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

            # iterate over dataset in batches
            for bidx, (targets, GIs, clouds_data) in enumerate(loader):
                model.ecc.set_info(GIs, args.cuda)
                label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()

                embeddings = ptnCloudEmbedder.run(model, *clouds_data)
                outputs = model.ecc(embeddings)

                fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
                collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))

        # aggregate predictions (mean)
        for fname, lst in collected.items():
            # o_cpu    = PREDICTIONS
            # tvec_cpu = ground truth
            o_cpu, t_cpu, tvec_cpu = list(zip(*lst))
            if args.test_multisamp_n > 1:
                o_cpu = np.mean(np.stack(o_cpu,0),0)
            else:
                o_cpu = o_cpu[0]
            t_cpu, tvec_cpu = t_cpu[0], tvec_cpu[0]
            predictions[fname] = np.argmax(o_cpu,1)
            o_cpu, t_cpu, tvec_cpu = filter_valid(o_cpu, t_cpu, tvec_cpu)
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

        per_class_iou = {}
        perclsiou = confusion_matrix.get_intersection_union_per_class()
        for c, name in dbinfo['inv_class_map'].items():
            try:
                per_class_iou[name] = perclsiou[c]
            except IndexError:
                print("Missing one label in data: " + name)

        return meter_value(acc_meter), confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), per_class_iou, predictions,  confusion_matrix.get_mean_class_accuracy(), confusion_matrix.confusion_matrix

    ############
    # Training loop
    best_iou = 0
    TRAIN_COLOR = '\x1b[96;1m'
    VAL_COLOR =   '\x1b[94;1m' 
    TEST_COLOR =  '\x1b[96;1m' 
    BEST_COLOR =  '\x1b[93;7m'
    BAD_COLOR =   '\x1b[91;7m'
    IOU_COLOR =   '\x1b[33;1m'
    CLOSE =       '\x1b[0m'
    epoch = args.start_epoch
    
    " Epoch is number of time you parse all data "
    for epoch in range(args.start_epoch, args.epochs):
        isBest = False
        
        firstEpoch = (epoch==args.start_epoch)
        print('#################')
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.ROOT_PATH))

        " Update tensor if grad is computed "
        scheduler.step() 

        # 1. Train
        loss_train, CM_train_pt, CM_train_spp = train()

        print(TRAIN_COLOR + '-> Train Loss: %1.4f' % (loss_train) + CLOSE)

        if math.isnan(loss_train): break

        # 2. Test on validation dataset
        if firstEpoch or (epoch % args.test_valid_nth_epoch == 0):
            loss_val, CM_val_pt, CM_val_spp = eval(True)
            acc_pt_val, avg_iou_val, avg_prec_val, avg_rec_val, iou_per_class_val, prec_per_class_val, rec_per_class_val = CM_val_pt.getStats()
            acc_spp_val = CM_val_spp.getAccuracy()
            print(VAL_COLOR + 'Validation -> loss: %1.4f,  acc pt: %3.2f%%,  acc spp: %3.2f%%,  avgIoU: %3.2f%%' % (loss_val*100, acc_pt_val*100, acc_spp_val*100, avg_iou_val*100) + CLOSE)

        # 3. Test on test dataset
        if firstEpoch or (epoch % args.test_nth_epoch == 0): 
            loss_test, CM_test_pt, CM_test_spp = eval(False)
            acc_pt_test, avg_iou_test, avg_prec_test, avg_rec_test, iou_per_class_test, prec_per_class_test, rec_per_class_test = CM_test_pt.getStats()
            acc_spp_test = CM_test_spp.getAccuracy()
            print(IOU_COLOR + 'IoU: %3.2f%%' % (avg_iou_test*100) + CLOSE)
            print(TEST_COLOR + 'Test       -> loss: %1.4f,  acc pt: %3.2f%%,  acc spp: %3.2f%%,  avgIoU: %3.2f%%' % (loss_test*100, acc_pt_test*100, acc_spp_test*100, avg_iou_test*100) + CLOSE)

        # 4. Check if isBest
        if avg_iou_val > best_iou or epoch < 5:
            print(BEST_COLOR + 'New best model achieved!' + CLOSE)
            best_iou = avg_iou_val
            isBest = True
        else:
            print(BAD_COLOR + 'Bad model' + CLOSE)

        # 5 Bonus. If needed resume last best model
        if not isBest and not args.not_only_best:
            args.resume = pathManager.modelFile 
            model, optimizer = resume(args, dbinfo, pathManager.modelFile)
            #args.start_epoch += 1
            #epoch += 1
        
        # 6. Write the csv stat file

        # Validation
        classNames = list(ColorLabelManager().label2Name.values())[1:] # Without "unknown"

        def rename(l, name):
            return [name + i for i in l]

        reportPath = pathManager.getTrainingCsvReport("validation")
        if isBest:
            header = ["epoch", "acc_pt", "acc_spp", "loss", "avg_iou", "avg_prec", "avg_rec"] + rename(classNames, "avg_iou_val_") + rename(classNames, "avg_prec_val_") + rename(classNames, "avg_rec_val_") 
            data = np.concatenate([[int(epoch), acc_pt_val, acc_spp_val, loss_val, avg_iou_val, avg_prec_val, avg_rec_val], iou_per_class_val, prec_per_class_val, rec_per_class_val])
            io.writeCsv(reportPath, header, data)
        else:
            io.duplicateLastLineCsv(reportPath)

        reportPath = pathManager.getTrainingCsvReport("test")
        if firstEpoch or (epoch % args.test_nth_epoch == 0): 
            header = ["epoch", "acc_pt", "acc_spp", "loss", "avg_iou", "avg_prec", "avg_rec"] + rename(classNames, "avg_iou_test_") + rename(classNames, "avg_prec_test_") + rename(classNames, "avg_rec_test_")
            data = np.concatenate([[int(epoch), acc_pt_test, acc_spp_test, loss_test, avg_iou_test, avg_prec_test, avg_rec_test], iou_per_class_test, prec_per_class_test, rec_per_class_test])
            io.writeCsv(reportPath, header, data)
        else:
            io.duplicateLastLineCsv(reportPath)

        # 7. Save the model
        if (firstEpoch and args.not_only_best) or (isBest and not args.not_only_best) or (epoch % args.save_nth_epoch == 0):
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'scaler': scaler}, pathManager.modelFile)

    # 8. Final evaluation
    if args.test_multisamp_n>0 and 'test' in args.db_test_name:
        acc_test, oacc_test, avg_iou_test, per_class_iou_test, predictions_test, avg_acc_test, confusion_matrix = eval_final()
        with h5py.File(pathManager.predictionFile, 'w') as hf:
            for fname, o_cpu in predictions_test.items():
                hf.create_dataset(name=fname, data=o_cpu) #(0-based classes)

        print('-> Multisample {}: Test accuracy: {}, \tTest oAcc: {}, \tTest avgIoU: {}, \tTest mAcc: {}'.format(args.test_multisamp_n, acc_test, oacc_test, avg_iou_test, avg_acc_test))

def resume(args, dbinfo, modelFile):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(modelFile)
    
    checkpoint['args'].model_config = args.model_config #to ensure compatibility with previous arguments convention
    #this should be removed once new models are uploaded
    
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change
    optimizer = create_optimizer(args, model)
    
    #model.load_state_dict(checkpoint['state_dict'])
    #to ensure compatbility of previous trained models with new InstanceNormD behavior comment line below and uncomment line above if not using our trained  models
    model.load_state_dict({k:checkpoint['state_dict'][k] for k in checkpoint['state_dict'] if k not in ['ecc.0._cell.inh.running_mean','ecc.0._cell.inh.running_var','ecc.0._cell.ini.running_mean','ecc.0._cell.ini.running_var']})

    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    args.start_epoch = checkpoint['epoch']
    return model, optimizer
    
def create_model(args, dbinfo):
    """ Creates model """

    if not 'use_pyg' in args:
        args.use_pyg = 0

    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit, use_pyg = args.use_pyg, cuda = args.cuda)

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)

    #print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    #print(model)    
    if args.cuda: 
        model.cuda()
    return model 

def create_optimizer(args, model):
    if args.optim=='sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim=='adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    

def filter_valid(output, target, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target!=-100
    if other is not None:
        return output[idx,:], target[idx], other[idx,...]
    return output[idx,:], target[idx]
    
def meter_value(meter):   
    return meter.value()[0] if meter.n>0 else 0

if __name__ == "__main__": 
    main(sys.argv[1:])
