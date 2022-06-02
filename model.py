import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class TripletNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(args.model_name) 
        self.projection = nn.Sequential(nn.Linear(self.model.config.d_model, args.hidden_size),
                                        nn.ReLU())
        #self.register_buffer("final_logits_bias", torch.zeros((1, self.model.config.vocab_size)))
        
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels, positive_masks=None, negative_masks=None, triplet=False):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        
        encoder_outputs = encoder(input_ids = input_ids,
                                  attention_mask = attention_mask)
        
        encoder_hidden_states = encoder_outputs[0] # [batch, seq_len, hidden_size]
                
        decoder_outputs = decoder(input_ids = decoder_input_ids,
                                  attention_mask = decoder_attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask)
        
        decoder_hidden_states = decoder_outputs[0] # [batch, dec_seq_len, hidden_size]
        
        lm_logits = self.model.lm_head(decoder_hidden_states) # + self.final_logits_bias # [batch, dec_seq_len, vocab_size]
        predictions = lm_logits.argmax(dim=2)
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        nll = criterion(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        
        if triplet:
            masked_encoder_positives = encoder_hidden_states.mul(positive_masks.unsqueeze(2)) # [b, s, h] * [b, s, 1] = [b, s, h]
            masked_encoder_negatives = encoder_hidden_states.mul(negative_masks.unsqueeze(2))
            
            proj_p = self.projection(masked_encoder_positives) # [b, s, h]
            proj_n = self.projection(masked_encoder_negatives) # [b, s, h]
            proj_s = self.projection(decoder_hidden_states) # [b, ds, h]
            
            proj_p = self.avg_pool(proj_p, attention_mask)
            proj_n = self.avg_pool(proj_n, attention_mask)
            proj_s = self.avg_pool(proj_s, decoder_attention_mask)
            
            t_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1.0)(proj_s, proj_p, proj_n)
            
            return nll, t_loss
            '''
            cos = nn.CosineSimilarity(dim=-1) / pair = nn.PairwiseDistance()
            sim_p = cos(hidden_p, hidden_s) # [b]
            sim_n = cos(hidden_n, hidden_s)
            torch.relu(sim_p - sim_n + 1.0).mean()
            '''
            
        return nll, torch.tensor(0), predictions
            
    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length
        
        return avg_hidden