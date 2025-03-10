import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
                    
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None, epoch_save_step =None, scheduler=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_model_path = save_path + f'_best.pth'
         
    train_losses = []
    val_losses = []
    val_metrics = []
    accuracies = []
    
    best_val_metric = -np.inf 
    best_epoch=-1
    
    if save_path:
        with open(save_path + "_telemetry.txt", "a") as myfile:
            dt = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            myfile.write("\n\n--------------------------------------------------------------\nTraining "
                            + dt
                            + "\n--------------------------------------------------------------\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
          
        for inputs, targets in train_loader:
            
            inputs = inputs.to(device,non_blocking=True)
            targets = targets.to(device,non_blocking=True)
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
        epoch_loss = running_loss / total_samples
        if scheduler:
            scheduler.step(epoch_loss)
        train_losses.append(epoch_loss) 

           
        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_val_samples = 0
        
        out = []
        tar = []
        with torch.no_grad():
            for inputs, targets in val_loader:
               
                inputs = inputs.to(device,non_blocking=True)
                targets = targets.to(device,non_blocking=True)
                outputs = model(inputs)
                           
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * targets.size(0)
                total_val_samples += targets.size(0)
                val_loss = val_running_loss / total_val_samples
                
                outputs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == targets).sum().item()
                
                out.append(outputs.detach().cpu())
                tar.append(targets.detach().cpu())

        accuracy = correct_predictions / total_val_samples
        val_loss = loss / total_val_samples
        
        val_losses.append((val_loss.detach().cpu()).numpy())    
        accuracies.append(accuracy)
        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)
        metrics = []
        
        F1 = torcheval.metrics.MulticlassF1Score()
        F1.update(all_targets, torch.argmax(all_outputs, dim=1))
        metrics = F1.compute()
        val_metrics.append(metrics.numpy())
        
        train_loss_str = f'{epoch_loss: .4f}'
        val_loss_str = f'{val_loss: .4f}'
        val_acc_str = f'{accuracy: .4f}'
        metric_str =  f'F1 Score: {metrics:.4f}'
        
        msg = f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | Val acc: {val_acc_str} | {metric_str}\n'
        print(msg)
        with open(save_path + "_telemetry.txt", "a") as myfile:
            myfile.write(msg)
        
        if save_path and  epoch_save_step:
           if (epoch + 1) % epoch_save_step == 0:
                checkpoint_path = os.path.join(save_path, f"_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model saved at epoch {epoch + 1} to {checkpoint_path}')
                
                
        # Save the best model based on validation metric
        current_val_metric =  metrics
        if save_path and ( current_val_metric > best_val_metric) :
            best_val_metric = current_val_metric
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
            print(f'Model saved at epoch {epoch + 1} to {best_model_path}')
            

    if save_path:
        last_model_path = save_path + '_last.pth'
        torch.save(model.state_dict(), last_model_path)
        print(f'Last model saved to {last_model_path}')
        with open(save_path + "_telemetry.txt", "a") as myfile:
            myfile.write(f'\nLast model saved to {last_model_path}\n')
    else:
        last_model_path = None
        
    with open(save_path + "_telemetry.txt", "a") as myfile:
            myfile.write(f'best epoch: {best_epoch} for F1 = {best_val_metric}')

    results = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_metrics': val_metrics,
    'accuracies': accuracies,
    'best_epoch': best_epoch
    }

    if save_path:
        results['best_model_path'] = best_model_path
        results['last_model_path'] = last_model_path
   
    return results
###############################################################################