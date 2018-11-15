import sys
import torch
import torch.nn as nn
from torch import optim
from .train import Trainer
from .validate import Validator
from src.models.model import DynamicEncoderRNN,BahdanauAttnDecoderRNN
import warnings
warnings.filterwarnings("ignore")

class Handler:
    def __init__(self,input_voc_size,output_voc_size,embed_size = 64,hidden_size = 64,learning_rate = 0.01, step_size=5, gamma=0.1):
        self.input_voc_size = input_voc_size
        self.encoder = DynamicEncoderRNN(input_voc_size,embed_size,hidden_size)
        self.decoder = BahdanauAttnDecoderRNN(hidden_size,embed_size,output_voc_size)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.encoder_optimizer_exp_lr_scheduler = optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=step_size, gamma=gamma)
        self.decoder_optimizer_exp_lr_scheduler = optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=step_size, gamma=gamma)
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.criterion = self.criterion.cuda()

    def update_progress(self,progress):
        '''
        update_progress() : Displays or updates a console progress bar
        Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        '''
        barLength = 50 # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rPercent: [{:s}] {:.2f}% {:s}".format( "█"*block + "-"*(barLength-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def train(self,dataloader,n_iter=1):
        trainer = Trainer(self,dataloader)
        for i in range(1,n_iter+1):
            print("Epoch %s"%i)
            trainer.train()

    def validate(self,dataloader):
        validator = Validator(self,dataloader)
        validator.validate()
