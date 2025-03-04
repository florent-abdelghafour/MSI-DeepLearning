import torch
 
def test(model, model_path, test_loader) : 
    
    Y = torch.tensor([]).cpu() 
    y_pred = torch.tensor([]).cpu()  
    
    model.load_state_dict(torch.load(model_path,weights_only=True))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            Y = torch.cat((Y, targets.detach().cpu()), dim=0)
            inputs = inputs.to(device,non_blocking=True)
            outputs = model(inputs)
            y_pred = torch.cat((y_pred, outputs.detach().cpu()), dim=0)


    return(y_pred,Y)