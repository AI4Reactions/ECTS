import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add, scatter_softmax, scatter_max
from einops import rearrange, repeat, reduce
from .equiops import Equiformer_Block
from .consistency import pad_dims_like, skip_scaling, output_scaling
from .gnn.GeoVec import *
from .gnn.GNN import *
from tqdm import tqdm 
from ..comparm import GP

class RPEncoder(torch.nn.Module):
    def __init__(self,n_head=8):
        super(RPEncoder, self).__init__()
        self.x_sca_emb=make_embed(len(GP.atom_types), GP.x_sca_hidden)
        self.x_vec_emb=VecExpansion(1,GP.x_vec_hidden)
        self.edge_sca_emb=make_embed(len(GP.bond_types)+1,GP.edge_sca_hidden)
        self.edge_vec_emb=VecExpansion(1,GP.edge_vec_hidden)
        self.GVGraphTransformer_layers = torch.nn.ModuleList(
            [GateNormUpdateCoor(
                GP.x_sca_hidden,
                GP.x_vec_hidden,
                GP.edge_sca_hidden,
                GP.edge_vec_hidden,
                n_head,
                GP.dropout,
                full_gate=True,
                only_update_coor=False,
                update_coor_clamp=GP.update_coor_clamp
            ) for _ in range(GP.c_block)])

    def forward(self, x_sca_vec,edge_sca_vec,coords,x_mask):
        x_sca,x_vec=x_sca_vec
        edge_sca,edge_vec=edge_sca_vec

        x_sca=self.x_sca_emb(x_sca)
        x_vec=self.x_vec_emb(x_vec)
        edge_sca=self.edge_sca_emb(edge_sca)
        edge_vec=self.edge_vec_emb(edge_vec)

        x_sca_vec=(x_sca,x_vec)
        edge_sca_vec=(edge_sca,edge_vec)

        update_mask =x_mask.float()
        coor_hidden = []
        for i in range(len(self.GVGraphTransformer_layers)):
            x_sca_vec, edge_sca_vec, coords = self.GVGraphTransformer_layers[i](
                x_sca_vec, edge_sca_vec, x_mask, coords,
                update_mask=update_mask)
            coor_hidden.append(coords)
        coor_hidden = torch.stack(coor_hidden, dim=-2)
        return x_sca_vec,edge_sca_vec,coords

class TsGen(torch.nn.Module):
    def __init__(self):
        super(TsGen, self).__init__()
        self.rencoder=RPEncoder()
        self.pencoder=RPEncoder()
        self.x_gate = GVGateResidue(GP.x_sca_hidden, GP.x_vec_hidden, full_gate=True)
        self.edge_gate = GVGateResidue(GP.edge_sca_hidden, GP.edge_vec_hidden, full_gate=True)
        self.sigmas_emb=make_embed(1, GP.x_sca_hidden)
        self.TsDecoder_layers = torch.nn.ModuleList(
            [GateNormUpdateCoor(
                GP.x_sca_hidden,
                GP.x_vec_hidden,
                GP.edge_sca_hidden,
                GP.edge_vec_hidden,
                GP.n_head,
                GP.dropout,
                full_gate=True,
                only_update_coor=False,
                update_coor_clamp=GP.update_coor_clamp
            ) for _ in range(GP.c_block)]) 
        return 

    def forward(self, ratoms,patoms,radjs,padjs,rcoords,pcoords,tscoords,sigmas,masks,sigma_data=0.5,sigma_min=0.002):
        
        c_skip = skip_scaling(sigmas, sigma_data, sigma_min)
        c_out = output_scaling(sigmas, sigma_data, sigma_min)
        c_skip = pad_dims_like(c_skip, tscoords).to(tscoords.device)
        c_out = pad_dims_like(c_out, tscoords).to(tscoords.device)
        sigmas=sigmas.to(tscoords.device)

        updated_x_sca,updated_edge_sca,denoised_coords=self.denoise(ratoms,patoms,radjs,padjs,rcoords,pcoords,tscoords,sigmas,masks)
        pos_output=c_skip*tscoords+c_out*denoised_coords
        return pos_output

    def denoise(self, ratoms,patoms,radjs,padjs,rcoords,pcoords,tscoords,sigmas,masks=None):
        # for pregen dat
        if GP.coords_scale:
            rcoords=rcoords/GP.coords_scale
            pcoords=pcoords/GP.coords_scale
            tscoords=tscoords/GP.coords_scale

        r_x_sca=ratoms.float()
        r_x_vec=(rcoords-rcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)

        p_x_sca=patoms.float()
        p_x_vec=(pcoords-pcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)
        r_edge_sca=radjs.float()

        r_edge_vec=(rcoords.unsqueeze(dim=-2)-rcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        p_edge_sca=padjs.float()
        p_edge_vec=(pcoords.unsqueeze(dim=-2)-pcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        r_x_sca_vec=(r_x_sca,r_x_vec)
        r_edge_sca_vec=(r_edge_sca,r_edge_vec)
        p_x_sca_vec=(p_x_sca,p_x_vec)
        p_edge_sca_vec=(p_edge_sca,p_edge_vec)

        p_x_sca_vec,p_edge_sca_vec,pcoords=self.rencoder(p_x_sca_vec,p_edge_sca_vec,pcoords,masks)
        r_x_sca_vec,r_edge_sca_vec,rcoords=self.pencoder(r_x_sca_vec,r_edge_sca_vec,rcoords,masks)
        updated_x_sca_vec=self.x_gate(r_x_sca_vec,p_x_sca_vec)
        
        updated_edge_sca_vec=self.edge_gate(r_edge_sca_vec,p_edge_sca_vec)
        updated_x_sca,updated_x_vec=updated_x_sca_vec
        sigmas=repeat(sigmas,'b -> b n c',n=p_x_sca.shape[1],c=1)

        sigma_control=self.sigmas_emb(sigmas)
        updated_x_sca=updated_x_sca+sigma_control
        updated_x_sca_vec=(updated_x_sca,updated_x_vec)

        x_sca_encoding=updated_x_sca.clone()
        edge_sca_encoding=updated_edge_sca_vec[0].clone()
        updated_mask =masks.float()

        for i in range(len(self.TsDecoder_layers)):
            updated_x_sca_vec, updated_edge_sca_vec, tscoords = self.TsDecoder_layers[i](
                updated_x_sca_vec, updated_edge_sca_vec, masks, tscoords,
                update_mask=updated_mask)

        if GP.coords_scale:
            tscoords=tscoords*GP.coords_scale

        return x_sca_encoding,edge_sca_encoding,tscoords

class TseGen(torch.nn.Module):
    def __init__(self):
        super(TseGen, self).__init__()
        self.rencoder=RPEncoder(n_head=4)
        self.pencoder=RPEncoder(n_head=4)
        self.x_gate = GVGateResidue(GP.x_sca_hidden, GP.x_vec_hidden, full_gate=True)
        self.edge_gate = GVGateResidue(GP.edge_sca_hidden, GP.edge_vec_hidden, full_gate=True)
        self.sigmas_emb=make_embed(1, GP.x_sca_hidden)

        self.tse_decoder=Equiformer_Block(token_input_dim=GP.x_sca_hidden,
                                          token_hidden_dim=32,
                                          edge_input_dim=GP.edge_sca_hidden,
                                          edge_hidden_dim=16,
                                          depth=4,
                                          reduce_dim_out = True,
                                          attend_self=True,
                                          l2_dist_attention=False) 
        return 

    def forward(self, ratoms,patoms,radjs,padjs,rcoords,pcoords,tscoords,sigmas,masks=None):
        # for pregen dat

        r_x_sca=ratoms.float()
        r_x_vec=(rcoords-rcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)

        p_x_sca=patoms.float()
        p_x_vec=(pcoords-pcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)
        r_edge_sca=radjs.float()

        r_edge_vec=(rcoords.unsqueeze(dim=-2)-rcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        p_edge_sca=padjs.float()
        p_edge_vec=(pcoords.unsqueeze(dim=-2)-pcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        r_x_sca_vec=(r_x_sca,r_x_vec)
        r_edge_sca_vec=(r_edge_sca,r_edge_vec)
        p_x_sca_vec=(p_x_sca,p_x_vec)
        p_edge_sca_vec=(p_edge_sca,p_edge_vec)

        p_x_sca_vec,p_edge_sca_vec,pcoords=self.rencoder(p_x_sca_vec,p_edge_sca_vec,pcoords,masks)
        r_x_sca_vec,r_edge_sca_vec,rcoords=self.pencoder(r_x_sca_vec,r_edge_sca_vec,rcoords,masks)
        updated_x_sca_vec=self.x_gate(r_x_sca_vec,p_x_sca_vec)
        
        updated_edge_sca_vec=self.edge_gate(r_edge_sca_vec,p_edge_sca_vec)
        updated_x_sca,updated_x_vec=updated_x_sca_vec
        sigmas=repeat(sigmas,'b -> b n c',n=p_x_sca.shape[1],c=1)
        sigma_control=self.sigmas_emb(sigmas)
        updated_x_sca=updated_x_sca+sigma_control
        updated_x_sca_vec=(updated_x_sca,updated_x_vec)

        x_sca_encoding=updated_x_sca.clone()
        edge_sca_encoding=updated_edge_sca_vec[0].clone()

        energy_output=self.tse_decoder(inputs=x_sca_encoding,edges=edge_sca_encoding,coors=tscoords,mask=masks.bool())
        
        ts_energy,ts_forces=energy_output.type0,energy_output.type1
        
        ts_energy=ts_energy*masks.long()
        
        ts_energy=torch.sum(ts_energy,dim=-1)
        #print (ts_energy)
        ts_forces=ts_forces*masks.unsqueeze(-1).long()

        return ts_energy,ts_forces
    
class MLP(torch.nn.Module):
    """
    Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
    ----
        in_features (int)         : Size of each input sample.
        hidden_layer_sizes (list) : Hidden layer sizes.
        out_features (int)        : Size of each output sample.
        dropout_p (float)         : Probability of dropping a weight.
    """

    def __init__(self, in_features : int, hidden_layer_sizes : list, out_features : int,
                 dropout_p : float) -> None:
        super().__init__()

        activation_function = torch.nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f : int, out_f : int, activation : torch.nn.Module,
                      dropout_p : float) -> torch.nn.Sequential:
        """
        Returns a linear block consisting of a linear layer, an activation function
        (SELU), and dropout (optional) stack.

        Args:
        ----
            in_f (int)                   : Size of each input sample.
            out_f (int)                  : Size of each output sample.
            activation (torch.nn.Module) : Activation function.
            dropout_p (float)            : Probability of dropping a weight.

        Returns:
        -------
            torch.nn.Sequential : The linear block.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        torch.nn.init.xavier_uniform_(linear.weight)
        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    def forward(self, layers_input : torch.nn.Sequential) -> torch.nn.Sequential:
        """
        Defines forward pass.
        """
        return self.seq(layers_input)

class GraphGather(torch.nn.Module):
    """
    GGNN readout function.
    """
    def __init__(self, node_features : int, hidden_node_features : int,
                 out_features : int, att_depth : int, att_hidden_dim : int,
                 att_dropout_p : float, emb_depth : int, emb_hidden_dim : int,
                 emb_dropout_p : float, big_positive=1e6) -> None:

        super().__init__()

        self.big_positive = big_positive

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            dropout_p=att_dropout_p
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            dropout_p=emb_dropout_p
        )

    def forward(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass.
        """
        Softmax     = torch.nn.Softmax(dim=1)

        cat         = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * self.big_positive
        energies    = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention   = Softmax(energies)
        embedding   = self.emb_nn(hidden_nodes)

        return torch.sum(attention * embedding, dim=1)
    
class TsConfidence(torch.nn.Module):
    def __init__(self):
        super(TsConfidence, self).__init__()
        self.rencoder=RPEncoder()
        self.pencoder=RPEncoder()
        self.x_gate = GVGateResidue(GP.x_sca_hidden, GP.x_vec_hidden, full_gate=True)
        self.edge_gate = GVGateResidue(GP.edge_sca_hidden, GP.edge_vec_hidden, full_gate=True)
        self.Tsencoder = torch.nn.ModuleList(
            [GateNormUpdateCoor(
                GP.x_sca_hidden,
                GP.x_vec_hidden,
                GP.edge_sca_hidden,
                GP.edge_vec_hidden,
                GP.n_head,
                GP.dropout,
                full_gate=True,
                only_update_coor=False,
                update_coor_clamp=GP.update_coor_clamp
            ) for _ in range(GP.c_block)]) 

        self.tsmlp=MLP(GP.x_sca_hidden,[GP.x_sca_hidden,GP.x_sca_hidden],GP.x_sca_hidden,GP.dropout)
        self.ts_graph_gather=GraphGather(node_features=GP.x_sca_hidden,
                                         hidden_node_features=GP.x_sca_hidden,
                                         out_features=1,
                                         att_depth=4,
                                         att_hidden_dim=GP.x_sca_hidden,
                                         att_dropout_p=0.1,
                                         emb_depth=4,
                                         emb_hidden_dim=GP.x_sca_hidden,
                                         emb_dropout_p=0.1)
        
        return 

    def forward(self, ratoms,patoms,radjs,padjs,rcoords,pcoords,tscoords,masks=None):
        # for pregen dat
        if GP.coords_scale:
            rcoords=rcoords/GP.coords_scale
            pcoords=pcoords/GP.coords_scale
            tscoords=tscoords/GP.coords_scale

        r_x_sca=ratoms.float()
        r_x_vec=(rcoords-rcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)

        p_x_sca=patoms.float()
        p_x_vec=(pcoords-pcoords.mean(dim=1,keepdims=True)).unsqueeze(-2)
        r_edge_sca=radjs.float()

        r_edge_vec=(rcoords.unsqueeze(dim=-2)-rcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        p_edge_sca=padjs.float()
        p_edge_vec=(pcoords.unsqueeze(dim=-2)-pcoords.unsqueeze(dim=-3)).unsqueeze(dim=-2)
        r_x_sca_vec=(r_x_sca,r_x_vec)
        r_edge_sca_vec=(r_edge_sca,r_edge_vec)
        p_x_sca_vec=(p_x_sca,p_x_vec)
        p_edge_sca_vec=(p_edge_sca,p_edge_vec)

        p_x_sca_vec,p_edge_sca_vec,pcoords=self.rencoder(p_x_sca_vec,p_edge_sca_vec,pcoords,masks)
        r_x_sca_vec,r_edge_sca_vec,rcoords=self.pencoder(r_x_sca_vec,r_edge_sca_vec,rcoords,masks)
        updated_x_sca_vec=self.x_gate(r_x_sca_vec,p_x_sca_vec)
        
        updated_edge_sca_vec=self.edge_gate(r_edge_sca_vec,p_edge_sca_vec)
        updated_mask =masks.float()

        for i in range(len(self.Tsencoder)):
            updated_x_sca_vec, updated_edge_sca_vec, tscoords = self.Tsencoder[i](
                updated_x_sca_vec, updated_edge_sca_vec, masks, tscoords,
                update_mask=updated_mask)

        x_sca_embs=updated_x_sca_vec[0]
        x_sca_embs_mlp=self.tsmlp(x_sca_embs)
        ts_confidence=self.ts_graph_gather(x_sca_embs_mlp,x_sca_embs,masks.bool())

        return ts_confidence