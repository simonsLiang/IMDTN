import os.path
import logging
from collections import OrderedDict
import torch
from utils import utils_logger
from utils import utils_image as util
from models.architecture import IMDTN


def main():

    utils_logger.logger_info('NTIRE2022-EfficientSR', log_path='NTIRE2022-EfficientSR.log')
    logger = logging.getLogger('NTIRE2022-EfficientSR')

    # --------------------------------
    # basic settings
    # --------------------------------
    # testsets = 'DIV2K'
    testsets = '/content/IMDTN/'
    testset_L = 'DIV2K_valid_LR_bicubic'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join('model_zoo', 'model_x4.pth')
    model = IMDTN(upscale=4)
    checkpoint = torch.load(model_path)

    new_state_dcit = OrderedDict()
    for k, v in checkpoint.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    model.load_state_dict(pretrained_dict, strict=False)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)


    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L)
    E_folder = os.path.join(testsets, testset_L+'_results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0
    window_size=6
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        _, _, h_old, w_old = img_L.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_L = torch.cat([img_L, torch.flip(img_L, [2])], 2)[:, :, :h_old + h_pad, :]
        img_L = torch.cat([img_L, torch.flip(img_L, [3])], 3)[:, :, :, :w_old + w_pad]

        start.record()
        img_E = model(img_L)[..., :h_old *4, :w_old * 4]
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        util.imsave(img_E, os.path.join(E_folder, img_name[:4]+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))
if __name__ == '__main__':

    main()