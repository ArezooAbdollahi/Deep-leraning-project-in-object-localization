from utils import *
import torch
import torchvision.utils
import torch.nn.init as init
from modelResnet18 import *
from tensorboardX import SummaryWriter
from tqdm import tqdm 

device = "cuda"

def main():
	writer = SummaryWriter()

	train_dataset = CUBDataset(
		root='/home/arezoo/ObjectLocalizationProject/',
		split='train_images' # name of file train.txt
	)

	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=32,
		num_workers=32,
		shuffle=True,
	)

	print("lenTrainLoader",train_loader.__len__())
	global_train_iter = 0
	Mymodel=modelResnet18().to(device)
	optimizer = torch.optim.Adam(Mymodel.parameters(), lr = 1e-4)
	for epoch in tqdm(range(10),desc='epoch: '):
		model= train(Mymodel, train_loader, optimizer, epoch, writer, global_train_iter)
		global_train_iter += 1
	torch.save(model.state_dict(), 'model.pt')


# main()							


# optimizer = torch.optim.Adam(Mymodel.parameters(), lr = 1e-5)
def train(model,train_loader, optimizer, epoch, writer, global_train_iter):
	# import ipdb; ipdb.set_trace()
	# best_model_weights = copy.deepcopy(Mymodel.state_dict())
	sum_loss = 0.0 
	sum_accuracy = 0.0
	length=train_loader.__len__()
	print(" length in train func", length)
	model.train()
	for i, data in tqdm(enumerate(train_loader),desc = "training: "):
		image, target, size = data
		
		
		image = image.to(device).float()
		target = target.to(device)
		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			output = model(image)
			loss = model.loss(output, target)
			loss.backward()
			optimizer.step()
			sum_loss += loss.item() 
			sum_accuracy += compute_acc(output.data.cpu(), target.data.cpu(), size) 
	
	avg_loss = sum_loss / length
	avg_acc = sum_accuracy / length

	writer.add_scalar('train_loss', avg_loss, global_train_iter)
	writer.add_scalar('train_accuracy', avg_acc, global_train_iter)
	
	print('Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
	# model.load_state_dict(best_model_weights)
	return model

if __name__ == '__main__':
	main()