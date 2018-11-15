import torch

class Trainer:
    def __init__(self,handler,dataloader):
        self.encoder = handler.encoder
        self.decoder = handler.decoder
        self.criterion = handler.criterion
        self.encoder_optimizer = handler.encoder_optimizer
        self.decoder_optimizer = handler.decoder_optimizer
        self.encoder_optimizer_exp_lr_scheduler = handler.encoder_optimizer_exp_lr_scheduler
        self.decoder_optimizer_exp_lr_scheduler = handler.decoder_optimizer_exp_lr_scheduler
        self.update_progress = handler.update_progress
        self.dataset_len = dataloader.sampler.__len__() or dataloader.dataset.__len__()
        self.dataloader = dataloader

    def train(self):
        completed = 0
        total_loss = torch.zeros(1)
        total_accuracy = torch.zeros(1)
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer_exp_lr_scheduler.step()
        self.decoder_optimizer_exp_lr_scheduler.step()
        for batch_idx,(input_seqs, input_lens,output_seqs, output_lens) in enumerate(self.dataloader):
            
            input_seqs = input_seqs.long()
            output_seqs = output_seqs.long()
            encoder_hidden = self.encoder.init_hidden(len(input_lens))
            word_input = torch.Tensor([0]*len(output_lens)).long()
            output_seqs = output_seqs.transpose(0,1) # B,T => T,B
            batch_loss = torch.zeros(1)
            batch_accuracy = torch.zeros(1)
            
            if torch.cuda.is_available():
                input_seqs = input_seqs.cuda()
                input_lens = input_lens.cuda()
                output_seqs = output_seqs.cuda()
                output_lens = output_lens.cuda()
                encoder_hidden = encoder_hidden.cuda()
                word_input = word_input.cuda()
                batch_loss = batch_loss.cuda()
                total_loss = total_loss.cuda()
                batch_accuracy = batch_accuracy.cuda()
                total_accuracy = total_accuracy.cuda()
                
            encoder_outputs,last_hidden = self.encoder(input_seqs,input_lens,encoder_hidden)
            for seq in output_seqs:
                word_output,last_hidden = self.decoder(word_input,last_hidden,encoder_outputs)
                word_input = seq
                topv, topi = word_output.topk(1)
                batch_loss += self.criterion(word_output,word_input)
                batch_accuracy += topi.eq(word_input.view_as(topi)).sum().item()
                
            # backward + optimize
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            batch_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            max_length,batch_size = output_seqs.size()
            batch_loss /= max_length*batch_size
            batch_accuracy /= max_length*batch_size
            total_loss += batch_loss     
            total_accuracy += batch_accuracy
            
            # print statistics
            completed += batch_size
            self.update_progress(completed/self.dataset_len)
            # print("Batch Accuracy:",round(batch_accuracy.item()*100,4),'Batch Loss:',round(batch_loss.item(),4))
        print("\nTrain Accuracy:",round(total_accuracy.item()*100/(batch_idx+1),2),'Train Loss:',round(total_loss.item()/(batch_idx+1),5))
