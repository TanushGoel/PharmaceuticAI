import torch
DEVICE = torch.device("mps")

import warnings
warnings.filterwarnings("ignore")

VERSIONS = {"esm2_t6_8M_UR50D":6, "esm2_t12_35M_UR50D":12, "esm2_t30_150M_UR50D":30, "esm2_t33_650M_UR50D":33, "esm2_t36_3B_UR50D":36, "esm2_t48_15B_UR50D":48}
class ESMFold():

    def __init__(self, v=0): 
        self.version = list(VERSIONS.keys())[v]
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", self.version) # https://github.com/facebookresearch/esm
        self.model.eval().to(DEVICE)
        self.converter = self.alphabet.get_batch_converter()
        
    def tokenize(self, seq):
        if isinstance(seq, str):
            data = [("", seq)]
        else:
            data = [("", s) for s in seq]
        _, _, batch_tokens = self.converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        return batch_tokens.to(DEVICE), batch_lens

    def __call__(self, seq, contacts=False):
        batch_tokens, batch_lens = self.tokenize(seq)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[VERSIONS[self.version]], return_contacts=contacts)
        tokens = results["representations"][VERSIONS[self.version]]
        
        embedding = []
        for i, tokens_len in enumerate(batch_lens):
            embedding.append(tokens[i, 1:tokens_len-1].mean(0))
        embedding = torch.stack(embedding)
        
        if contacts:
            return embedding, results["contacts"]
        return embedding