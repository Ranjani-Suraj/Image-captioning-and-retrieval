'''
    Module contains final Model and all pieces of it.
'''
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

class ImageEncoder(nn.Module):
    '''
        Encodes image and returns it's embedding.
    '''

    def __init__(self, model, device='cpu'):
        super(ImageEncoder, self).__init__()

        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        print("img dimensions:", image.shape)
        image = self.preprocessor(images=image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)

        return image_features.pooler_output
        #returns pooled image embedding -=> [1, 768]


class Mapping(nn.Module):
    '''
        Maps image embedding to GPT-2 embedding.
        Uses TransformerEncoder to process image embedding,
        and uses a linear layer to get ep_len sequence of embeddings
        of dimension embed_size

        [1, embed_size] = [ep_len, embed_size]
    '''

    def __init__(
        self,
        ep_len,
        num_layers,
        embed_size,
        n_heads,
        forward_expansion,
        dropout,
        device='cpu'
    ):
        super(Mapping, self).__init__()

        self.ep_len = ep_len
        self.embed_size = embed_size

        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=embed_size*forward_expansion,
                dropout=dropout,
                batch_first=True,
                device=device
            ),
            num_layers=num_layers
        ).to(self.device)

        self.mapping = nn.Linear(embed_size, ep_len * embed_size).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapping(x)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size]
                if train_mode else
                [self.ep_len, self.embed_size]
            )
        ) # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class TextDecoder(nn.Module):
    '''
        Processes embedding into caption.
        takes input embeds and generates tokens.
    '''

    def __init__(self, model, device='cpu'):
        super(TextDecoder, self).__init__()

        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size



    def forward(self, embedding, attention_mask=None):
        text_features = self.model(inputs_embeds=embedding, attention_mask=attention_mask)

        return text_features.logits



class CaptioningModel(nn.Module):
    '''
        Final Model class. Puts all pieces together and generates caption based on image.
    '''

    def __init__(self, clip_model, text_model, ep_len, num_layers, n_heads, forward_expansion, dropout, max_len, device='cpu'):
        '''
            Model constructor.
            Args:
                num_layers: number of layers in the TransformerEncoder
                n_heads: number of heads in the MultiHeadAttention
                forward_expansion: expansion factor for the feedforward layer
                dropout: dropout probability
                max_len: maximum length of the generated text
        '''
        super(CaptioningModel, self).__init__()

        self.device = device
        self.ep_len = ep_len

        self.image_encoder = ImageEncoder(model=clip_model, device=device)
        self.mapping = Mapping(ep_len=self.ep_len, num_layers=num_layers, embed_size=self.image_encoder.model.config.hidden_size, n_heads=n_heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
        self.text_decoder = TextDecoder(model=text_model, device=device)
        print(self.text_decoder.parameters())
        assert self.image_encoder.model.config.hidden_size == self.text_decoder.model.config.n_embd, "Embedding size of models mismatch"

        self.max_len = max_len

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.text_decoder.tokenizer.pad_token_id) # chanded on epoch 91
        self.criterion = nn.CrossEntropyLoss()

        self.freeze_layers()

    def freeze_layers(self):
        for p in [*list(self.image_encoder.parameters()), *list(self.text_decoder.parameters())[14:-14]]: # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False

    def forward(self, img_embedded, temperature=1.0):
        '''
            Caption generation for a single image.
            Args:
                img_embedded: Image features.
            Returns:
                caption: generated caption [str]
                tokens: generated tokens [torch.Tensor]
        '''

        if temperature <= 0.0:
            temperature = 1.0
            print('Temperature must be positive. Setting it to 1.0')

        with torch.no_grad():
            #  Map image to text space. See class Mapping in this file.

            # img embedded shape torch.Size([1, 768])
            # img mapped shape torch.Size([4, 768])
            # torch.Size([1, 768])
            # emb shape torch.Size([5, 768])
            # pos emb shape torch.Size([5, 768])
            # emb shape torch.Size([1, 5, 768])
            # pred shape torch.Size([1, 5, 50257])


            # print("img embedded shape", img_embedded.shape)  #[1, 768]

            img_mapped = self.mapping.forward(img_embedded, False)

            # print("img mapped shape", img_mapped.shape)
            # Obtain <BOS> token embedding. You should get the <BOS> token ID from the tokenizer of GPT-2.
            # Then, the <BOS> token ID is converted to an embedding by an Embedding layer named "wte" in GPT-2.
            # See https://huggingface.co/docs/transformers/model_doc/gpt2.
            # Adjust the dimensions/shape of the bos token embeddding and then, concatenate the <BOS> embedding and mapped image embedding to get a start embedding.
            bos_emb = self.text_decoder.model.transformer.wte(torch.tensor(self.text_decoder.tokenizer.bos_token_id).to(self.device)) # obtain <bos> embedding
            bos_emb = bos_emb.unsqueeze(0)
            # print(bos_emb.shape) #[1, 768]
            start_emb = torch.cat([bos_emb, img_mapped], dim = 0) # obtain start embedding [5, 768]
            # start_emb = torch.cat([img_mapped, bos_emb], dim = 0) # 
            # print("start emb shape", start_emb.shape)
            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    # text embeddings of the tokens found so far
                    tok_emb = self.text_decoder.model.transformer.wte(torch.tensor(tokens).to(self.device))

                    emb = torch.cat([start_emb, tok_emb], dim = 0)
                else:
                    emb = start_emb #[5, 768]

                # Obtain position embedding and add it into token embedding. The position embedding is stored as an Embedding layer named "wpe" in GPT-2.
                # See https://huggingface.co/docs/transformers/model_doc/gpt2.
                pos_emb = self.text_decoder.model.transformer.wpe(torch.arange(emb.shape[0]).to(self.device)) # obtain position embedding
                # adds [5, 768] position embeddings for positions 0-4
                # print("emb shape", emb.shape)
                # print("pos emb shape", pos_emb.shape)

                emb += pos_emb # add position embedding to token embedding
                # so to our current embedding we add the positional embedding from the text decoder so it can be decoded

                emb = emb.unsqueeze(0)
                # print("emb shape", emb.shape)
                # Generate the next token. See https://huggingface.co/docs/transformers/model_doc/gpt2.
                pred = self.text_decoder(emb)  # generate the next logits
                # pred [x, 50257] => seq length, vocabulary
                # pred = pred.reshape(-1)

                pred = pred[:, -1, :]
                # print("pred shape", pred.shape)
                pred = torch.softmax(pred / temperature, dim=-1).to(self.device)
                # print("109", pred.shape)
                _, pred = torch.max(pred, dim=1)
                # print("111", pred.shape)
                last_token = pred[-1].item()
                # get the token with highest probability
                tokens.append(last_token)

                #stop whne you hit the eos token
                if last_token == self.text_decoder.tokenizer.eos_token_id:
                    break

            decoded = self.text_decoder.tokenizer.decode(tokens[:-1])

            decoded = decoded.strip()
            if not decoded:
                return None, None
            decoded = decoded[0].upper() + decoded[1:]
            # print("tok", decoded)
            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        for p in self.mapping.parameters():
          p.requires_grad = True

        device = self.device
        # ensure inputs are on device (you do this in Trainer already but be explicit)
        img_emb = img_emb.to(device)

        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]
        x = x.to(device)
        x_mask = x_mask.to(device)
        # print("img emb shape", img_emb.shape)
        # print("x shape", x.shape)
        # print("mask shape", x_mask.shape)
        # print("y shape", y.shape)
        # Map image to text space. See class Mapping in this file.
        img_mapped = self.mapping(img_emb, train_mode=True).to(self.device) # mapping. Now we are training the model, so the train_model parameter should be set to True
        #[N, self.ep_len, self.embed_size] N being the batch size
        # print("img mapped shape", img_mapped.shape)
        # N, len, embed_size = 64, 44, 768
        text_emb = self.text_decoder.model.transformer.wte(x)
        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat([torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1)

        # Obtain position embedding and add it into token embedding. The position embedding is stored as an Embedding layer named "wpe" in GPT-2.
        # See https://huggingface.co/docs/transformers/model_doc/gpt2.

        pos_emb = self.text_decoder.model.transformer.wpe(torch.arange(x.shape[1]).to(self.text_decoder.device)) # obtain position embedding
        pos_emb = pos_emb.expand_as(x)

        # print("pos_emb shape", torch.tensor(pos_emb).shape)

        x += pos_emb # add position embedding to token embedding
        # print("new x shape", x.shape)
        # Generate tokens. Note that it is slightly different from codes in forward() function because of attention masks.
        # See https://huggingface.co/docs/transformers/model_doc/gpt2.

        res = self.text_decoder(x, attention_mask = x_mask) # generate tokens

        res = torch.softmax(res, dim=2)
        #Calculate loss value, fill arguments inside the below function.
        loss = self.criterion(res[:, self.ep_len:, :].reshape(-1, res.shape[-1]), y.reshape(-1)) # cross-entropy loss


        return loss
