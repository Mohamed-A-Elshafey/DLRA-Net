import torch
import os
import numpy as np
import cv2
from DLRA import DLRA
import glob
import hdf5storage as hdf5
import time
import argparse
from thop import profile
from thop import clever_format


def model_profile(model):
  print("**************************************")  
  print("enter model Profiler")  
  device = 'cuda:0'
  #input_tensor = torch.rand(1, 3, 482, 512).to(device)
  input_tensor = torch.rand(1, 3, 64, 64).to(device)  
  model = model.to(device)  
  macs, params = profile(model, inputs = (input_tensor, ))
  macs, params = clever_format([macs, params], "%.3f")
  print("FLOPS: ",macs)
  print("Number of parameters: ",params)
  model.eval()
  print('Model is loaded, start forwarding.')
  # 'number of runs to compute average forward timing. default is 50'
  Num_runs = 50
  # number of warmup runs to avoid initial slow speed. default is 5
  Num_warmUp = 5
  with torch.no_grad():
     for i in range(Num_runs):
        if i == Num_warmUp:
            start_time = time.time()
            pred = model(input_tensor)
  end_time = time.time()
  total_forward = end_time - start_time
  print('Total forward time is %4.2f seconds' % total_forward)
  actual_num_runs = Num_runs - Num_warmUp
  latency = total_forward / actual_num_runs
    #fps = (cfg.CONFIG.DATA.CLIP_LEN * cfg.CONFIG.DATA.FRAME_RATE) * actual_num_runs / total_forward
  #model = model.cpu()  
  print("FPS: ",50, "; Latency: ", latency)
  print("**************************************")  
  print("End model Profiler")


def get_reconstruction_gpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        var_output1 = model(var_input[:,:,:-2,:])
        var_output2 = model(var_input[:,:,2:,:])
        var_output = torch.cat([var_output1, var_output2[:,:,-2:,:]], 2)
        end_time = time.time()

    return end_time-start_time, var_output.cpu()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SSR_test")
parser.add_argument("--RGB_dir", type=str, default='./Valid_RGB', help="absolute Input_RGB_path")
parser.add_argument("--model_dir", type=str, default='./model/trained.pth', help="absolute Model_path")
parser.add_argument("--result_dir", type=str, default='./test_results', help="absolute Save_Result_path")
opt = parser.parse_args()

img_path = opt.RGB_dir
model_path = opt.model_dir
result_path = opt.result_dir

var_name = 'cube'
# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
model = DLRA(3, 31, 80, 10)
save_point = torch.load(model_path,weights_only=True)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
img_path_name.sort()

## Calculate the total number of Params and Macs
model_profile(model)

for i in range(len(img_path_name)):
      # load rgb images
      print(img_path_name[i].split('/')[-1])
      rgb = cv2.imread(img_path_name[i])
      rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
      rgb = np.float32(rgb)
      rgb = rgb / rgb.max()
      rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
      rgb = torch.from_numpy(rgb).float()
      use_time, temp_hyper = get_reconstruction_gpu(rgb, model)
      img_res = temp_hyper.numpy() * 1.0
      img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
      img_res_limits = np.minimum(img_res, 1.0)
      img_res_limits = np.maximum(img_res_limits, 0)

      mat_name = img_path_name[i].split('\\')[-1][:-4] + '.mat'
      print(mat_name)
      mat_dir = os.path.join(result_path, mat_name)
      hdf5.savemat(mat_dir, {var_name: img_res}, format='7.3', store_python_metadata=True)