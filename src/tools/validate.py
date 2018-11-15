import torch

class Validator:
    def __init__(self,handler,dataloader):
        self.encoder = handler.encoder
        self.decoder = handler.decoder
        self.criterion = handler.criterion
        self.update_progress = handler.update_progress
        self.dataset_len = dataloader.sampler.__len__() or dataloader.dataset.__len__()
        self.dataloader = dataloader

    def validate(self):
        completed = 0
        total_loss = torch.zeros(1)
        total_accuracy = torch.zeros(1)
        self.encoder.eval()
        self.decoder.eval()
        for batch_idx,(input_seqs, input_lens,output_seqs, output_lens) in enumerate(self.dataloader):
            input_seqs = input_seqs.long()
            output_seqs = output_seqs.long()
            encoder_hidden = self.encoder.init_hidden(len(input_lens))
            word_input = torch.Tensor([0]*len(output_lens)).long()
            output_seqs = output_seqs.transpose(0,1) # B,T => T,B
            
            if torch.cuda.is_available():
                input_seqs = input_seqs.cuda()
                input_lens = input_lens.cuda()
                output_seqs = output_seqs.cuda()
                output_lens = output_lens.cuda()
                encoder_hidden = encoder_hidden.cuda()
                word_input = word_input.cuda()
                total_loss = total_loss.cuda()
                total_accuracy = total_accuracy.cuda()
            encoder_outputs,last_hidden = self.encoder(input_seqs,input_lens,encoder_hidden)
            for seq in output_seqs:
                word_output,last_hidden = self.decoder(word_input,last_hidden,encoder_outputs)
                topv, topi = word_output.topk(1)
                word_input = topi.squeeze(1)
                total_loss += self.criterion(word_output,seq).item()
                total_accuracy += topi.eq(seq.view_as(topi)).sum().item()
            max_length,batch_size = output_seqs.size()
            total_loss /= max_length*batch_size
            total_accuracy /= max_length*batch_size
            # print statistics
            completed += batch_size
            self.update_progress(completed/self.dataset_len)
        print("\nTest Accuracy:",round(total_accuracy.item()*100/(batch_idx+1),2),'Test Loss:',round(total_loss.item()/(batch_idx+1),5))
