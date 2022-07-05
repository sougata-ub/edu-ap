import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils.parametrize as parametrize


class Encoder(nn.Module):
    def __init__(self, transformer, n_layers):
        super().__init__()
        self.enc = transformer
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask):
        hidden = self.enc(input_ids=input_ids,
                          attention_mask=attention_mask)
        token_hidden = hidden["hidden_states"][-self.n_layers:]
        token_hidden = torch.sum(torch.stack(token_hidden), dim=0)  # batch, seq, hidden

        return token_hidden


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h):
        return self.dropout(F.relu(self.ff(h)))


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):#,add_attn: bool = False):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
#         self.add_attn = add_attn
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor: #, attn: torch.Tensor
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):#, add_attn: bool):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)#, add_attn=add_attn)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor: #, attn: torch.Tensor
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class Diagonal(nn.Module):
    def forward(self, X):
        return torch.diag_embed(X.diagonal(dim1=-2, dim2=-1))


class Convolution(nn.Module):
    def __init__(self, ic, oc, k, drop=0.2):
        super().__init__()
        self.regular_l1 = nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=k, stride=1,  padding="same",
                                    groups=ic)
        # self.regular_l1 = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=1, padding="same")#, groups=ic)
        # self.diagonal_l1 = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k,stride=1, padding="same")#, groups=ic)
        # self.regular_l2 = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=1, padding="same", groups=ic)
        # self.diagonal_l2 = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k,stride=1, padding="same", groups=ic)
        # self.dropout = nn.Dropout(drop)

        # parametrize.register_parametrization(self.diagonal_l1, "weight", Diagonal())
        # parametrize.register_parametrization(self.diagonal_l2, "weight", Diagonal())

    def forward(self, X):
        # (batch, nc, n_edu, n_edu)
        l1_reg = self.regular_l1(X)  # (batch, nc, n)
        # l1_reg = self.regular_l1(X)  # (batch, nc, n_edu, n_edu)
        # l1_diag = self.diagonal_l1(X)  # (batch, nc, n_edu, n_edu)

        # l2_reg = torch.tanh(self.dropout(self.regular_l2(l1_reg)))  # (batch, nc, n_edu, n_edu)
        # l2_diag = torch.tanh(self.dropout(self.diagonal_l2(l1_diag)))  # (batch, nc, n_edu, n_edu)
        # l2_pooled = l1_reg + l1_diag #l2_reg + l2_diag  # (batch, nc, n_edu, n_edu)
        return l1_reg # l2_pooled


class SumTranspose(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight1 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight2 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight1.size(0))
        nn.init.uniform_(self.weight1, -bound, bound)
        nn.init.uniform_(self.weight2, -bound, bound)

    def forward(self, X, Y):
        w1 = self.weight1.triu(1) + (1 - self.weight1).triu(1).transpose(-1, -2)
        w2 = self.weight2.tril(-1) + (1 - self.weight2).tril(-1).transpose(-1, -2)
        return X * torch.tanh(w1) + Y * torch.tanh(w2)


class UnitSymmetric(nn.Module):
    def forward(self, X):
        X = torch.sigmoid(X)
        return torch.cat([X[:, :1], (1-X).flip(1).flip(0)[:, 1:]], -1)


class Parser(nn.Module):
    def __init__(self, transformer, n_layers, in_dim, out_dim, n_deprel_classes=None, n_type_classes=None,
                 n_token_classes=None, add_convolution=False, sum_transpose=False, prompt_attention=False,
                 n_heads=4, n_attn_layers=2, add_context="none"):
        super().__init__()
        self.encoder = Encoder(transformer, n_layers)  # , add_attn)

        if add_context != "none":
            self.mha = nn.ModuleList([nn.MultiheadAttention(in_dim, n_heads, batch_first=True, dropout=0.1)\
                                      for _ in range(n_attn_layers)])
            self.mha_drop = nn.Dropout(0.1)
            # self.mha_fc = nn.Linear(in_dim * 2, in_dim)

        self.mlp_src_edge = MLP(in_dim, out_dim)
        self.mlp_trg_edge = MLP(in_dim, out_dim)
        self.biaf_head = Biaffine(out_dim, out_dim, 1)  # , add_attn)

        self.predict_segments = True if n_token_classes is not None else False
        self.predict_edu_type = True if n_type_classes is not None else False
        self.predict_edge_labels = True if n_deprel_classes is not None else False
        self.add_convolution, self.sum_transpose, self.prompt_attention = add_convolution, sum_transpose, \
                                                                          prompt_attention
        self.add_context = add_context # prompt, left, all, none

        if self.predict_edge_labels:
            self.mlp_src_label = MLP(in_dim, out_dim)
            self.mlp_trg_label = MLP(in_dim, out_dim)
            self.biaf_deprel = Biaffine(out_dim, out_dim, n_deprel_classes)  # , add_attn)

        if self.predict_edu_type:
            self.mlp_edu_type = MLP(in_dim, out_dim)
            self.type_classifier = nn.Linear(out_dim, n_type_classes)

        if self.predict_segments:
            self.mlp_adu_boundary = MLP(in_dim, out_dim)
            self.adu_boundary_classifier = nn.Linear(out_dim, n_token_classes)

        if self.add_convolution:
            self.conv_edge = Convolution(1, 1, 3)
            self.conv_deprel = Convolution(n_deprel_classes, n_deprel_classes, 3)

        if self.sum_transpose:
            self.sum_trans = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
            # parametrize.register_parametrization(self.sum_trans[0], "weight", UnitSymmetric())

        # if self.transpose_result:
        #     self.transpose_linear = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

    def forward(self, input_ids, attention_mask, edu_idx, prompt_input_ids=None, prompt_attention_mask=None,
                lhc_input_ids=None, lhc_attention_mask=None, lhc_edu_idx=None, rhc_input_ids=None,
                rhc_attention_mask=None, rhc_edu_idx=None):
        enc = self.encoder(input_ids, attention_mask)  # batch, seq, in_dim
        para_enc = enc[torch.arange(enc.size(0)), edu_idx].transpose(0, 1)  # batch, n_edu, hidden

        if self.add_context == "prompt":
            prompt_enc = self.encoder(prompt_input_ids, prompt_attention_mask)  # batch, n_seq, in_dim
            ctx_enc = prompt_enc[:, 0, :].unsqueeze(1)  # batch, 1, in_dim

        elif self.add_context == "left":
            prompt_enc = self.encoder(prompt_input_ids, prompt_attention_mask)  # batch, n_seq, in_dim
            prompt_enc = prompt_enc[:, 0, :].unsqueeze(1)  # batch, 1, in_dim

            lhc_enc = self.encoder(lhc_input_ids, lhc_attention_mask)  # n_lhc, lhc_seq, in_dim
            lhc_enc = lhc_enc[:, 0, :].unsqueeze(0)
            ctx_enc = torch.cat([prompt_enc, lhc_enc], 1)

        elif self.add_context == "all":
            prompt_enc = self.encoder(prompt_input_ids, prompt_attention_mask)  # batch, n_seq, in_dim
            prompt_enc = prompt_enc[:, 0, :].unsqueeze(1)  # batch, 1, in_dim

            lhc_enc = self.encoder(lhc_input_ids, lhc_attention_mask)  # n_lhc, lhc_seq, in_dim
            rhc_enc = self.encoder(rhc_input_ids, rhc_attention_mask)  # n_rhc, rhc_seq, in_dim

            lhc_enc = lhc_enc[:, 0, :].unsqueeze(0)  # 1, n_lhc, in_dim
            rhc_enc = rhc_enc[:, 0, :].unsqueeze(0)

            ctx_enc = torch.cat([prompt_enc, lhc_enc, rhc_enc], 1)  # batch, cat seq, hidden
        else:
            ctx_enc = None

        if self.add_context != "none" and ctx_enc is not None:
            # para_root = para_enc[:, 0, :].unsqueeze(1)
            for mha in self.mha:
                mha_op, _ = mha(para_enc, ctx_enc, ctx_enc)  # batch, cat seq, hidden
                para_enc = para_enc + self.mha_drop(mha_op)

            # para_enc[:, 0, :] = self.mha_fc(torch.cat([para_root[:, 0, :], para_enc[:, 0, :]], -1))

        # Edge prediction
        hs_src_edge = self.mlp_src_edge(para_enc)  # batch, n_edu, out_dim
        hs_trg_edge = self.mlp_trg_edge(para_enc)  # batch, n_edu, out_dim
        logits_head = self.biaf_head(hs_src_edge, hs_trg_edge).squeeze_(3)  # (batch, n_edu, n_edu)

        if self.add_convolution:
            l_diag = self.conv_edge(logits_head.diagonal(-1, dim1=-2, dim2=-1).unsqueeze(1)).squeeze(1)  # batch, l_diag
            mask = 1 - torch.diag_embed(torch.ones_like(logits_head).diagonal(-1, dim1=-2, dim2=-1), offset=-1)
            l_diag = torch.diag_embed(l_diag, offset=-1) + (logits_head * mask)# (batch, n_edu, n_edu)
            logits_head = 0.5*logits_head + 0.5*l_diag
            # logits_head = self.conv_edge(logits_head.unsqueeze(1)).squeeze_(1)  # (batch, n_edu, n_edu)

        if self.sum_transpose:
            logits_head_stack = torch.stack([logits_head, logits_head.triu(1).transpose(1, 2)], -1)
            # logits_head_stack = torch.stack([logits_head, logits_head.transpose(1, 2)], -1)
            logits_head = self.sum_trans(logits_head_stack).squeeze(-1)

        # Edge Labelling
        if self.predict_edge_labels:
            hs_src_label = self.mlp_src_label(para_enc)  # batch, n_edu, out_dim
            hs_trg_label = self.mlp_trg_label(para_enc)  # batch, n_edu, out_dim
            logits_deprel = self.biaf_deprel(hs_src_label, hs_trg_label)  # (batch, n_edu, n_edu, n_deprel_classes)
            if self.add_convolution:
                logits_deprel = logits_deprel.permute(0, 3, 1, 2)
                l_diag_deprel = self.conv_deprel(logits_deprel.diagonal(-1, dim1=-2, dim2=-1)) #batch, nc, l_diag
                mask = 1 - torch.diag_embed(torch.ones_like(logits_deprel).diagonal(-1, dim1=-2, dim2=-1), offset=-1)
                l_diag_deprel = torch.diag_embed(l_diag_deprel, offset=-1) + (logits_deprel * mask)  # (batch, n_edu, n_edu)
                logits_deprel = (0.5*logits_deprel + 0.5*l_diag_deprel).permute(0, 2, 3, 1)

            if self.sum_transpose:
                # logits_deprel_stack = torch.stack([logits_deprel, logits_deprel.transpose(1, 2)], -1)
                logits_deprel_stack = torch.stack([logits_deprel, logits_deprel.triu(1).transpose(1, 2)], -1)
                logits_deprel = self.sum_trans(logits_deprel_stack).squeeze(-1)
        else:
            logits_deprel = None

        # EDU type prediction
        if self.predict_edu_type:
            logits_adu_type = self.type_classifier(self.mlp_edu_type(para_enc))  # batch, n_edu, adu_type
        else:
            logits_adu_type = None

        """ Token Level Processing """
        # EDU span prediction
        if self.predict_segments:
            logits_adu_boundary = self.adu_boundary_classifier(self.mlp_adu_boundary(enc))  # batch, seq, tok_type
        else:
            logits_adu_boundary = None

        dct = {"logits_head": logits_head, "logits_deprel": logits_deprel,
               "logits_adu_type": logits_adu_type, "logits_adu_boundary": logits_adu_boundary}
        return dct


class BaselineEncoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.enc = transformer

    def forward(self, input_ids, attention_mask):
        hidden = self.enc(input_ids=input_ids,
                          attention_mask=attention_mask)
        token_hidden = hidden["last_hidden_state"] # batch, seq, hidden

        return token_hidden


class BaselineParser(nn.Module):
    def __init__(self, transformer, in_dim, out_dim, n_deprel_classes):
        super().__init__()
        self.bert = BaselineEncoder(transformer)
        self.fnn = nn.Linear(in_dim, in_dim)

        self.mlp_src_edge = MLP(in_dim, out_dim)
        self.mlp_trg_edge = MLP(in_dim, out_dim)

        self.mlp_src_label = MLP(in_dim, out_dim)
        self.mlp_trg_label = MLP(in_dim, out_dim)

        self.biaf_head = Biaffine(out_dim, out_dim, 1)
        self.biaf_deprel = Biaffine(out_dim, out_dim, n_deprel_classes)

    def forward(self, input_ids, attention_mask):
        r_s = self.bert(input_ids, attention_mask)
        r_root = self.fnn(torch.mean(r_s, 1))

        R = torch.cat([r_root.unsqueeze(1), r_s], 1)

        H_edge_parent = self.mlp_src_edge(R)
        H_edge_child = self.mlp_trg_edge(R)

        H_label_parent = self.mlp_src_label(R)
        H_label_child = self.mlp_trg_label(R)

        sc_label = self.biaf_head(H_edge_parent, H_edge_child).squeeze_(3)
        sc_score = self.biaf_deprel(H_label_parent, H_label_child)

        dct = {"logits_head": sc_label, "logits_deprel": sc_score}
        return dct
