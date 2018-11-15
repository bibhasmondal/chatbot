import torch

class Predictor:
    def __init__(self,handler,train_dataset):
        self.encoder = handler.encoder
        self.decoder = handler.decoder
        self.criterion = handler.criterion
        self.update_progress = handler.update_progress
        self.train_dataset = train_dataset

    def predict(self,input_seq,input_len):
        input_len = input_len.view(1)
        pred = torch.LongTensor()
        self.encoder.eval()
        self.decoder.eval()
        input_seq = input_seq.long()
        encoder_hidden = self.encoder.init_hidden(len(input_len))
        word_input = torch.Tensor([0]*len(input_len)).long()
        if torch.cuda.is_available():
            input_seq = input_seq.cuda()
            input_len = input_len.cuda()
            encoder_hidden = encoder_hidden.cuda()
            word_input = word_input.cuda()
            pred = pred.cuda()
        encoder_outputs,last_hidden = self.encoder(input_seq,input_len,encoder_hidden)
        for _ in range(self.train_dataset.MAX_LENGTH):
            word_output,last_hidden = self.decoder(word_input,last_hidden,encoder_outputs)
            topv, topi = word_output.topk(1)
            word_input = topi.squeeze(1)
            pred = torch.cat((pred,topi),dim=1)
        return pred

    def predict_from_loader(self,dataloader):
        for batch_idx,(input_seqs, input_lens) in enumerate(dataloader):
            batch_pred = self.predict(input_seqs,input_lens)
            test_pred = torch.cat((test_pred,batch_pred),dim=0)
        return test_pred

    def predict_from_sentence(self,line):
        line = self.train_dataset.normalize_string(line)
        input_seq = self.train_dataset.indexes_from_sentence(self.train_dataset.input_voc,line)+[self.train_dataset.EOS_token]
        input_len = torch.Tensor([len(input_seq)]).int()
        input_seq = input_seq + [self.train_dataset.PAD_token]*(self.train_dataset.MAX_LENGTH-len(input_seq))
        input_seq = torch.Tensor(input_seq).view(1,-1).long()
        pred = self.predict(input_seq,input_len)
        input_seq = input_seq.squeeze(0)
        pred = pred.squeeze(0)
        # Print Input
        decoded_words = []
        for index in input_seq:
            decoded_words.append(self.train_dataset.input_voc.index2word[index.item()])
            if index.item() == self.train_dataset.EOS_token:
                break
        print("Input:"," ".join(decoded_words))
        # Print Predict
        decoded_words = []
        for index in pred:
            decoded_words.append(self.train_dataset.output_voc.index2word[index.item()])
            if index.item() == self.train_dataset.EOS_token:
                break
        print("Predicted:"," ".join(decoded_words))
