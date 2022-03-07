import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity, get_dataloader

class batchData:
    imgL : torch.tensor



def main():  
    type = "train"
    batch_size = 16
    shuffle = False
    percentDataToUse = 0.1
    loader = get_dataloader(type, percentDataToUse, batch_size, shuffle)
    i = 0
    for tup in loader:
        i+=1
        print(f"\nI have completed {i} iterations out of {len(loader)}")
        print(tup.focalLength)
        print(tup.depthL.size())
        #print(tup.imgR)
        
        
    


if __name__ == "__main__":
    main()