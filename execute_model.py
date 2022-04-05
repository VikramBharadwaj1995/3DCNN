from train import Trainer
import torch
import sys, os
# import wandb  

# wandb.init(project="Ego", entity="3dcnn")

def main(data_dir):
    
    # Initialize best loss and best psnr
    current_best_loss = 10000.0
    
    # Initialize training object from the Train class
    epochs = 100
    trainer = Trainer()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Define lists to keep track of training/validation loss and PSNR
    training_loss = []
    validation_loss = []

    # Begin training
    for epoch in range(0, epochs):
        
        #Train
        val_dataloader, train_loss = trainer.train(data_directory=data_dir, current_epoch=epoch)
        training_loss.append(train_loss)
        
        # Validate
        model, current_loss = trainer.validate(val_dataloader=val_dataloader, current_epoch=epoch)
        validation_loss.append(current_loss)
        
        if current_loss < current_best_loss:
            current_best_loss = current_loss
            
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')

            torch.save(model.state_dict(), 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))
        
        print("-----------------------------------")
        print("Train & Validation complete for epoch - ", epoch)
        print("Final Validation Loss = ", current_best_loss)
        print("===================================")

    # print("Training and validation complete - current Loss = ", current_best_loss)
    
    return current_best_loss, epoch

if __name__ == '__main__':
    data_dir = sys.argv[1]
    print("Starting->")
    final_loss, epoch = main(data_dir)
    print("Train and validation complete! -> Epochs = ", epoch, "\tFinal Loss = ", final_loss)