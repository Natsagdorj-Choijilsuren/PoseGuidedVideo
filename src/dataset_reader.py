from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pandas as pd


class DataLoader(Dataset):

    def __init__(self,transformer):

        self.transformer = transformer

    def __getitem__(self,idx):

        pass

    def __len__(self):

        #return len()
        pass

def get_dataloader(image_folder,csv_path):

    csv_data = pd.read_csv(csv_path)
    
if __name__ == "__main__":

    pass

    
