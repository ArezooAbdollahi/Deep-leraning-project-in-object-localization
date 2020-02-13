from utils import *
import torch
import torchvision.utils
import torch.nn.init as init
from modelResnet18 import *
from tensorboardX import SummaryWriter
from tqdm import tqdm 

device = 'cuda'
def main():
	test_dataset = CUBDataset(
		root='/home/arezoo/ObjectLocalizationProject/',
		split='test_images' # name of file train.txt
	)

	test_loader = torch.utils.data.DataLoader(
		dataset=test_dataset,
		batch_size=32,
		num_workers=32,
		shuffle=False,
	)

	print("lenTestLoader",test_loader.__len__())
	Mymodel=modelResnet18().to(device)
	Mymodel.load_state_dict(torch.load('model.pt'))
	f = open("output.txt","w+")

	with torch.no_grad():
	    for data in test_loader:
	        images, size = data
	        images = images.to(device).float()
	        outputs = Mymodel(images)
	        outputs = box_transform_inv(outputs.data.cpu(), size)
	        print(outputs.shape)
	        for output in outputs: 
	        	f.write(str(output).lstrip('tensor([ ').rstrip('])').replace(',','') + '\n')



	   

main()