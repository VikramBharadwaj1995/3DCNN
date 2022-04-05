import torch
import os
from egoexo import EgoExo
from C3D import C3D
import torch.nn as nn
from torch.autograd import Variable

class Trainer:
    def __init__(self):
        self.batch_size = 1
        self.learning_rate = 0.001
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = C3D()

    def train(self, data_directory, current_epoch):
        dataset = os.listdir(data_directory)
        train_size = int(0.7*len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # dataloaders
        # Train dataloader
        train_dataset = EgoExo(train_dataset, data_directory)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Validation dataloader
        val_dataset = EgoExo(val_dataset, data_directory)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Model and the Loss function to use
        # You may also use a combination of more than one loss function 
        # or create your own.

        print("Model loaded on GPU ->", torch.cuda.is_available())
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        
        # train loop
        # Define Running Loss to keep track later in the train loop
        running_loss = 0.0
        
        for i, (input, target) in enumerate(train_dataloader):
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            target = target.unsqueeze(1)

            output = self.model(input)
            loss = self.criterion(output, target)

            # Set the gradient of tensors to zero.
            optimizer.zero_grad()
            # Compute the gradient of the current tensor by employing chain rule and propogating it back in the network.
            loss.backward()
            # Update the parameters in the direction of the gradient.
            optimizer.step()
            # Update scheduler. 
            scheduler.step()
            
            running_loss += loss.item()
            if i % 50 == 0:
                print("Current Epoch = ", current_epoch, "\nCurrent loss = ", loss)

        final_loss = running_loss / len(train_dataloader)
        
        return val_dataloader, final_loss

    def validate(self, val_dataloader, current_epoch):
        # Validation loop begin
        # ------
        running_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                input = Variable(input).cuda()
                target = Variable(target).cuda()
                target = target.unsqueeze(1)

                output = self.model(input)
                loss = self.criterion(output, target)

                running_loss += loss.item()

                if i % 50 == 0:
                    print("Current Epoch = ", current_epoch, "\nCurrent loss = ", loss)

        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        final_loss = running_loss / len(val_dataloader)

        return self.model, final_loss