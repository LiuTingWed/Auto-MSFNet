import os
import sys
import glob
import numpy as np
import torch
import utils as utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import data
import time
import torch.nn.functional as F
from scipy import misc
from matplotlib.pyplot import imsave
from torch.autograd import Variable
from model_resnet50 import Network_Resnet50 as Network_Resnet50
from model_vgg16 import Network_vgg16 as Network_vgg16

from PIL import Image
from torchvision.transforms import transforms
from metric import *
from skimage import img_as_ubyte
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("test_model")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--test_size', type=int, default=256, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=128, help='num of init channels')
parser.add_argument('--model_path', type=str, default='./checkpoint/Auto_MSFNet_resnet50.pt',
                    help='path of pretrained checkpoint')
parser.add_argument('--backbone', type=str, default='resnet50', help='test dataset')
parser.add_argument('--fu_arch', type=str, default='fusion_genotype_resnet50', help='which architecture to use')
parser.add_argument('--note', type=str, default='fusion_genotype_resnet50', help='test dataset')

args = parser.parse_args()
args.save = '{}-{}'.format(args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

dataset = ['HKU-IS-WI1D', 'DUTS', 'DUT-OMRON', 'ECSSD', 'PASCAL-S']


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    torch.cuda.set_device(args.gpu)
    genotype_fu = eval("genotypes.%s" % args.fu_arch)
    if args.backbone == "vgg16":
        model = Network_vgg16(genotype_fu)
    elif args.backbone == "resnet50":
        model = Network_Resnet50(genotype_fu)

    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    for i, dataset_name in enumerate(dataset):
        test_image_root = '/home/oip/testData/' + dataset_name + '/test_images/'
        test_gt_root = '/home/oip/testData/' + dataset_name + '/test_masks/'

        test_data = data.MyTestData(test_image_root, test_gt_root, args.test_size)
        test_queue = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        num_test = len(test_data)
        Fmax_measure, Fm_measure, mae, S_measure = infer(test_queue, model, dataset_name, num_test)
        logging.info('dataset_name {}'.format(dataset_name))
        logging.info('Fmax-measuree %f', Fmax_measure)
        logging.info('Fm-measuree %f', Fm_measure)
        logging.info('mae %f', mae)
        logging.info('S-measure %f', S_measure)


def infer(test_queue, model, dataset_name, num_test):
    model.eval()
    savepath = './prediction/' + dataset_name
    cal_fm = CalFM(num=num_test)  # cal是一个对象
    cal_mae = CalMAE(num=num_test)
    cal_sm = CalSM(num=num_test)
    for step, (input, target, name) in enumerate(test_queue):
        input = input.cuda()
        target = torch.squeeze(target)
        with torch.no_grad():
            h_nopool2,_,_= model(input)
        test_output_root = os.path.join(args.save, savepath)
        if not os.path.exists(test_output_root):
            os.makedirs(test_output_root)
        H,W = target.shape

        h_nopool2 = F.interpolate(h_nopool2,(H,W),mode='bilinear')
        output_rgb = torch.squeeze(h_nopool2)
        predict_rgb = output_rgb.sigmoid().data.cpu().detach().numpy()
        predict_rgb = img_as_ubyte(predict_rgb)
        cv2.imwrite(test_output_root + '/' + name[0] + '.png', predict_rgb)
        target = target.cpu().detach().numpy()
        max_pred_array = predict_rgb.max()
        min_pred_array = predict_rgb.min()

        if max_pred_array == min_pred_array:
            predict_rgb = predict_rgb / 255
        else:
            predict_rgb = (predict_rgb - min_pred_array) / (max_pred_array - min_pred_array)

        max_target = target.max()
        min_target = target.min()
        if max_target == min_target:
            target = target / 255
        else:
            target = (target - min_target) / (max_target - min_target)

        cal_fm.update(predict_rgb, target)
        cal_mae.update(predict_rgb, target)
        cal_sm.update(predict_rgb, target)


        if step % 50 == 0 or step == len(test_queue) - 1:
            logging.info(
                "TestDataSet:{} Step {:03d}/{:03d}  ".format(
                    dataset_name, step, len(test_queue) - 1))
    _, maxf, mmf, _, _ = cal_fm.show()
    mae = cal_mae.show()
    sm = cal_sm.show()
    return maxf, mmf, mae, sm,


if __name__ == '__main__':
    main()

