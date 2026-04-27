import math
import sys
from typing import Iterable
import swanlab
import torch
import numpy as np

import utils
from models.RADDet_finetune.yolo_head import decodeYolo, decodeYolo_fpn, yoloheadToPredictions, nms
import models.RADDet_finetune.mAP as mAP
import torch.distributed as dist
import time

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    config_model=None,
    anchor_boxes=None,
    ):
    
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # add metrics
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("box_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("conf_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("category_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))


    header = "Epoch: [{}]".format(epoch)
    print_freq = 20     # 每20个batch打印一次

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, labels) in enumerate(             
    #     metric_logger.log_every(data_loader, print_freq, header)
    # ):

    for data_iter_step, (samples, label0, label1, label2, raw_boxes) in enumerate(              
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        if data_iter_step % accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            # utils.adjust_learning_rate_v2(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        # samples是将要进入模型的数据
        samples = samples.to(device, non_blocking=True)


        label0 = label0.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        raw_boxes = raw_boxes.to(device)
        # print(f"samples shape: {samples.shape}")
        # print(f"label0 shape: {label0.shape}")
        # print(f"label1 shape: {label1.shape}")
        # print(f"label2 shape: {label2.shape}")
        # print(f"raw_boxes shape: {raw_boxes.shape}")
        # with torch.amp.autocast("cuda"):    # 自动混合精度（Automatic Mixed Precision, AMP）FP16和FP32混合训练
        #     _, feature = model(samples)
        #     pred_raw, pred = decodeYolo(feature,
        #                                 input_size=input_size,
        #                                 anchor_boxes=anchor_boxes,
        #                                 scale=config_model["yolohead_xyz_scales"][0])
        #     box_loss, conf_loss, category_loss = criterion(pred_raw, pred, label, raw_boxes[..., :6])
        #     loss = box_loss + conf_loss + category_loss
                
        #     loss_value = loss.item()
            
        feature0, feature1, feature2 = model(samples)
        # print(f"Output feature0 shape: {feature0.shape}")
        # print(f"Output feature1 shape: {feature1.shape}")
        # print(f"Output feature2 shape: {feature2.shape}")
        pred_raw0, pred0 = decodeYolo_fpn(feature0,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw1, pred1 = decodeYolo_fpn(feature1,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw2, pred2 = decodeYolo_fpn(feature2,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        box_loss0, conf_loss0, category_loss0 = criterion(pred_raw0, pred0, label0, raw_boxes[..., :6])
        box_loss1, conf_loss1, category_loss1 = criterion(pred_raw1, pred1, label1, raw_boxes[..., :6])
        box_loss2, conf_loss2, category_loss2 = criterion(pred_raw2, pred2, label2, raw_boxes[..., :6])
        loss = box_loss0 + conf_loss0 + category_loss0 + \
                        box_loss1 + conf_loss1 + category_loss1 + \
                        box_loss2 + conf_loss2 + category_loss2
        conf_loss = conf_loss0 + conf_loss1 + conf_loss2
        box_loss = box_loss0 + box_loss1 + box_loss2
        category_loss = category_loss0 + category_loss1 + category_loss2
        loss_value = loss.item()    

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            # clip_grad=args.clip_grad,   # 梯度裁剪
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()    # 同步cuda 等待所有在 GPU 上进行的操作完成
        
        lr = optimizer.param_groups[0]["lr"]
        total_loss_r, box_loss_r, conf_loss_r, category_loss_r = loss.cpu().detach(), \
            box_loss.cpu().detach(), conf_loss.cpu().detach(), category_loss.cpu().detach()
            
        # dist reduce
        total_loss_r = utils.all_reduce_mean(total_loss_r)
        box_loss_r = utils.all_reduce_mean(box_loss_r)
        conf_loss_r = utils.all_reduce_mean(conf_loss_r)
        category_loss_r = utils.all_reduce_mean(category_loss_r)
            
        # update metrics
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss_r)
        metric_logger.update(box_loss=box_loss_r)
        metric_logger.update(conf_loss=conf_loss_r)
        metric_logger.update(category_loss=category_loss_r)

        loss_value_reduce = utils.all_reduce_mean(total_loss_r)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('reduce_train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            log_writer.add_scalar("loss/total_loss", total_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/box_loss", box_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/category_loss", category_loss_r, global_step=epoch_1000x)

        if args.wandb and utils.is_main_process():
            swanlab.log({
                        'lr': lr,
                        'total_loss': total_loss_r,
                        'box_loss': box_loss_r,
                        'conf_loss': conf_loss_r,
                        'category_loss': category_loss_r
                    })


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(    
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    config_model=None,
    anchor_boxes=None,
    num_classes=None,
    args = None,
    ):

    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    # metric_logger.add_meter("ap_all", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_person", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_bicycle", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_car", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_motorcycle", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_bus", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("ap_truck", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    metric_logger.add_meter("val_total_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_box_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_conf_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_category_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    
    mean_ap_test = 0.0
    ap_all_class_test = []
    ap_all_class = []
    total_losstest = []
    box_losstest = []
    conf_losstest = []
    category_losstest = []
    for class_id in range(num_classes):
        ap_all_class.append([])
        
    # switch to evaluation mode
    model.eval()
    
    print_freq = 10
    for data_iter_step, (samples, label0, label1, label2, raw_boxes) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):
        
        # if data_iter_step >= 10:        # for test only
        #     break
        
        # samples是将要进入模型的数据
        samples = samples.to(device, non_blocking=True)

        ## 2 channels
        # samples = samples.permute(0, 4, 3, 1, 2)  # NRADC -> NCDRA
        ##
        
        label0 = label0.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        raw_boxes = raw_boxes.to(device)

        # with torch.amp.autocast("cuda"):    # 自动混合精度（Automatic Mixed Precision, AMP）FP16和FP32混合训练
            # _, feature = model(samples)
            # pred_raw, pred = decodeYolo(
            #     feature,
            #     input_size=input_size,
            #     anchor_boxes=anchor_boxes,
            #     scale=config_model["yolohead_xyz_scales"][0]
            # )
            # box_loss, conf_loss, category_loss = criterion(
            #     pred_raw, pred, label, raw_boxes[..., :6]
            # )
            # loss = box_loss + conf_loss + category_loss
            
        feature0, feature1, feature2 = model(samples)
        pred_raw0, pred0 = decodeYolo_fpn(feature0,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw1, pred1 = decodeYolo_fpn(feature1,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw2, pred2 = decodeYolo_fpn(feature2,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        box_loss0, conf_loss0, category_loss0 = criterion(pred_raw0, pred0, label0, raw_boxes[..., :6])
        box_loss1, conf_loss1, category_loss1 = criterion(pred_raw1, pred1, label1, raw_boxes[..., :6])
        box_loss2, conf_loss2, category_loss2 = criterion(pred_raw2, pred2, label2, raw_boxes[..., :6])
        loss = box_loss0 + conf_loss0 + category_loss0 + \
                        box_loss1 + conf_loss1 + category_loss1 + \
                        box_loss2 + conf_loss2 + category_loss2
        conf_loss = conf_loss0 + conf_loss1 + conf_loss2
        box_loss = box_loss0 + box_loss1 + box_loss2
        category_loss = category_loss0 + category_loss1 + category_loss2
        
        box_loss_b = box_loss.item()
        conf_loss_b = conf_loss.item()
        category_loss_b = category_loss.item()
        
        # box_loss_b, conf_loss_b, category_loss_b = box_loss.item(), conf_loss.item(), category_loss.item()
        total_losstest.append(loss.item())
        box_losstest.append(box_loss_b)
        conf_losstest.append(conf_loss_b)
        category_losstest.append(category_loss_b)
        
        pred0 = pred0.cpu().detach().numpy()
        pred1 = pred1.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()
        raw_boxes_np = raw_boxes.cpu().numpy()
        
        # start_time = time.time()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes_np[batch_id]
            pred_frame0 = pred0[batch_id]
            pred_frame1 = pred1[batch_id]
            pred_frame2 = pred2[batch_id]
            predicitons0 = yoloheadToPredictions(pred_frame0,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons1 = yoloheadToPredictions(pred_frame1,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons2 = yoloheadToPredictions(pred_frame2,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons = np.concatenate((predicitons0, predicitons1, predicitons2), axis=0)
            nms_pred = nms(
                predicitons, 
                config_model["nms_iou3d_threshold"],
                config_model["input_shape"],
                sigma=0.3,
                method="nms"
            )
            mean_ap, ap_all_class = mAP.mAP(
                nms_pred,
                raw_boxes_frame,
                config_model["input_shape"],
                ap_all_class,
                tp_iou_threshold=config_model["mAP_iou3d_threshold"]
            )
            mean_ap_test += mean_ap
        # print("ap time: ", time.time() - start_time)
        
        # ap_metric = [0 if len(class_ap) == 0 else np.mean(class_ap) for class_ap in ap_all_class]
        # metric_logger.update(ap_all=np.mean(ap_metric))
        # metric_logger.update(ap_person=np.mean(ap_metric[0]))
        # metric_logger.update(ap_bicycle=np.mean(ap_metric[1]))
        # metric_logger.update(ap_car=np.mean(ap_metric[2]))
        # metric_logger.update(ap_motorcycle=np.mean(ap_metric[3]))
        # metric_logger.update(ap_bus=np.mean(ap_metric[4]))
        # metric_logger.update(ap_truck=np.mean(ap_metric[5]))
        metric_logger.update(val_total_loss=loss.item())
        metric_logger.update(val_box_loss=box_loss_b)
        metric_logger.update(val_conf_loss=conf_loss_b)
        metric_logger.update(val_category_loss=category_loss_b)
        
        # temp_all_ap, temp_all_ap_list = decouple_mAP(ap_all_class)
        # metric_logger.update(ap_all=temp_all_ap)
        # metric_logger.update(ap_person=temp_all_ap_list[0])
        # metric_logger.update(ap_bicycle=temp_all_ap_list[1])
        # metric_logger.update(ap_car=temp_all_ap_list[2])
        # metric_logger.update(ap_motorcycle=temp_all_ap_list[3])
        # metric_logger.update(ap_bus=temp_all_ap_list[4])
        # metric_logger.update(ap_truck=temp_all_ap_list[5])
        
    
    # for ap_class_i in ap_all_class:
    #     if len(ap_class_i) == 0:
    #         class_ap = 0.
    #     else:
    #         class_ap = np.mean(ap_class_i)
    #     ap_all_class_test.append(class_ap)
    # mean_ap_test = np.mean(ap_all_class_test)
    
    mean_ap_test, ap_all_class_test = dist_decouple_mAP(ap_all_class) 

    # 打印结果
    print("-------> ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
          "ap_truck: %.6f" % (mean_ap_test, ap_all_class_test[0], ap_all_class_test[1], ap_all_class_test[2],
                              ap_all_class_test[3], ap_all_class_test[4], ap_all_class_test[5]))
    print("-------> total_loss: %.6f, box_loss: %.6f, conf_loss: %.6f, category_loss: %.6f" %
          (np.mean(total_losstest), np.mean(box_losstest), np.mean(conf_losstest), np.mean(category_losstest)))
    
    # 构建返回字典
    map_dict = {
        'ap_all': mean_ap_test,
        'ap_person': ap_all_class_test[0],
        'ap_bicycle': ap_all_class_test[1],
        'ap_car': ap_all_class_test[2],
        'ap_motorcycle': ap_all_class_test[3],
        'ap_bus': ap_all_class_test[4],
        'ap_truck': ap_all_class_test[5],
        'val_total_loss': np.mean(total_losstest),
        'val_box_loss': np.mean(box_losstest),
        'val_conf_loss': np.mean(conf_losstest),
        'val_category_loss': np.mean(category_losstest)
    }
    print('map_dict: ', map_dict)
    
    # 日志记录
    if log_writer is not None:
        log_writer.add_scalar('ap/ap_all', map_dict['ap_all'], epoch)
        log_writer.add_scalar('ap/ap_person', map_dict['ap_person'], epoch)
        log_writer.add_scalar('ap/ap_bicycle', map_dict['ap_bicycle'], epoch)
        log_writer.add_scalar('ap/ap_car', map_dict['ap_car'], epoch)
        log_writer.add_scalar('ap/ap_motorcycle', map_dict['ap_motorcycle'], epoch)
        log_writer.add_scalar('ap/ap_bus', map_dict['ap_bus'], epoch)
        log_writer.add_scalar('ap/ap_truck', map_dict['ap_truck'], epoch)
        log_writer.add_scalar('val_loss/total_loss', map_dict['val_total_loss'], epoch)
        log_writer.add_scalar('val_loss/box_loss', map_dict['val_box_loss'], epoch)
        log_writer.add_scalar('val_loss/conf_loss', map_dict['val_conf_loss'], epoch)
        log_writer.add_scalar('val_loss/category_loss', map_dict['val_category_loss'], epoch)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if args.wandb and utils.is_main_process():
        swanlab.log({
            'ap/all': mean_ap_test.item() if hasattr(mean_ap_test, 'item') else mean_ap_test,
            'ap/person': ap_all_class[0].item() if hasattr(ap_all_class[0], 'item') else ap_all_class[0],
            'ap/bicycle': ap_all_class[1].item() if hasattr(ap_all_class[1], 'item') else ap_all_class[1],
            'ap/car': ap_all_class[2].item() if hasattr(ap_all_class[2], 'item') else ap_all_class[2],
            'ap/motorcycle': ap_all_class[3].item() if hasattr(ap_all_class[3], 'item') else ap_all_class[3],
            'ap/bus': ap_all_class[4].item() if hasattr(ap_all_class[4], 'item') else ap_all_class[4],
            'ap/truck': ap_all_class[5].item() if hasattr(ap_all_class[5], 'item') else ap_all_class[5],
        })
    
    return map_dict

    # {**map_dict, **{k: meter.global_avg for k, meter in metric_logger.meters.items()}}
    
    
def dist_decouple_mAP(ap_all_class__):
    ap_all_class_test__ = []
    for ap_class_i in ap_all_class__:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = torch.tensor(np.sum(ap_class_i)).cuda()
            class_num = torch.tensor(len(ap_class_i)).cuda()
            dist.all_reduce(class_ap)
            dist.all_reduce(class_num)
            ap = class_ap / class_num
        ap_all_class_test__.append(ap.item())
    mean_ap_test__ = np.mean(ap_all_class_test__)
    
    return mean_ap_test__, ap_all_class_test__