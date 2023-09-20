import h3
import numpy as np
import torch_geometric as tg
import torch as th
import einops
from torch_scatter import scatter
from torch import nn

def build_Grid2Mesh(lat_lon: list, resolution: int):
    h3_nodes = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
    h3_to_idx = {h_i: i for i, h_i in enumerate(h3_nodes)}
    ll_nodes = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lon]

    grid, mesh, features = [], [], []
    for ll_idx, h in enumerate(ll_nodes):
        mesh.append(h3_to_idx[h] + len(lat_lon))
        grid.append(ll_idx)
        d = h3.point_dist(lat_lon[ll_idx], h3.h3_to_geo(h), unit='rads')
        features.append([np.sin(d), np.cos(d)])
    return tg.data.Data(edge_index = th.tensor([grid, mesh], dtype = th.long), 
                        num_nodes = len(lat_lon) + len(h3_nodes),
                        edge_attr = th.tensor(features, dtype = th.float))

def build_Mesh2Mesh(levels: list, k_hop: int = 1):
    levels = sorted(levels) #make sure that coarse -> fine
    srcs, dsts, features = [], [], []
    pos = []
    hex_to_idx = {}
    num_hexes = 0
    for resolution in levels:
        #get nodes at this level
        h3_nodes = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution))) #not sure if sort is needed here
        #update index dictionary
        hex_to_idx.update({h_i: i + num_hexes for i, h_i in enumerate(h3_nodes)}) #only works with unique hex codes!
        num_hexes += h3.num_hexagons(resolution) #ensure correct number of nodes in indexing
        for h in h3_nodes:
            pos.append(h3.h3_to_geo(h)) #only needed for verification
            #if parent hex is in the graph, add a bidirectional edge
            if resolution - 1 in levels: 
                n = h3.h3_to_parent(h, resolution - 1)
                d = h3.point_dist(h3.h3_to_geo(n), h3.h3_to_geo(h), unit='rads')
                #node -> parent edge
                srcs.append(hex_to_idx[h])
                dsts.append(hex_to_idx[n])
                features.append([np.sin(d), np.cos(d)])
                #parent -> node edge
                srcs.append(hex_to_idx[n])
                dsts.append(hex_to_idx[h])
                features.append([np.sin(d), np.cos(d)])
            #k-hop nearest neighbors
            nbrs = h3.k_ring(h, k_hop)
            for n in nbrs:
                d = h3.point_dist(h3.h3_to_geo(n), h3.h3_to_geo(h), unit='rads')
                #node -> neighbor edge
                srcs.append(hex_to_idx[h])
                dsts.append(hex_to_idx[n])
                features.append([np.sin(d), np.cos(d)])
        
    return tg.data.Data(edge_index = th.tensor([srcs, dsts], dtype = th.long),
                        pos = th.tensor(pos, dtype = th.float),
                        num_nodes = num_hexes,
                        edge_attr = th.tensor(features, dtype = th.float))

def build_Mesh2Grid(lat_lon: list, resolution: int, out_edges: int = 3):
    assert 0 < out_edges < 7, 'out_edges should be in [1,6]'
    h3_nodes = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
    h3_to_idx = {h_i: i for i, h_i in enumerate(h3_nodes)}
    ll_nodes = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lon]
    
    grid, mesh, features = [], [], []
    for ll_idx, h in enumerate(ll_nodes):
        #determine the nearest neighbors to the grid point ll_idx
        nbrs = sorted(list(h3.k_ring(h, 1)))
        ds = [h3.point_dist(h3.h3_to_geo(n), lat_lon[ll_idx], unit='rads') for n in nbrs]
        idcs = np.argsort(ds)[:out_edges]
        #add unidirectional edges from the out_edges nearest neighbors
        for i in idcs:
            mesh.append(h3_to_idx[nbrs[i]] + len(lat_lon))
            grid.append(ll_idx)
            features.append([np.sin(ds[i]), np.cos(ds[i])])
    return tg.data.Data(edge_index = th.tensor([mesh, grid], dtype = th.long), 
                        num_nodes = len(h3_nodes) + len(ll_nodes),
                        edge_attr = th.tensor(features, dtype = th.float))

class FFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_norm = True):
        super().__init__()
        
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels) if use_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x):
        return self.norm(self.residual(x) + self.ffn(x))
    
class EdgeModel(nn.Module):
    def __init__(self, node_channels, edge_channels = 2, hidden_channels = 128):
        super().__init__()
        self.edge_mlp = FFN(in_channels = node_channels * 2 + edge_channels,
                            hidden_channels = hidden_channels, 
                            out_channels = edge_channels)

    def forward(self, src, dest, edge_attr, u = None, batch = None):
        out = th.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out) + edge_attr
        return out

class NodeModel(th.nn.Module):
    def __init__(self, node_channels, edge_channels = 2, hidden_channels = 128):
        super().__init__()
        self.node_mlp_1 = FFN(in_channels = node_channels + edge_channels, 
                              hidden_channels = hidden_channels, 
                              out_channels = node_channels + edge_channels)
        self.node_mlp_2 = FFN(in_channels = node_channels*2 + edge_channels, 
                              hidden_channels = hidden_channels, 
                              out_channels = node_channels)
        
    def forward(self, x, edge_index, edge_attr, u = None, batch = None):
        row, col = edge_index
        out = th.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(src = out, index = col, dim = 0, reduce = 'mean', dim_size = x.shape[0])
        out = th.cat([x, out], dim=1)
        out = self.node_mlp_2(out) + x
        return out
    
class GraphEncoder(nn.Module):
    def __init__(self, 
                 data_channels, 
                 node_channels, 
                 lat_lon,
                 levels = [0, 1, 2],
                 hidden_channels = 128, 
                 edge_channels = 2):
        super().__init__()
        
        self.graph = build_Grid2Mesh(lat_lon, resolution = max(levels))
        self.num_h3_nodes = h3.num_hexagons(max(levels))
        self.lat_lon = lat_lon
        self.node_channels = node_channels
        
        self.node_encoder = FFN(in_channels = data_channels, 
                                hidden_channels = hidden_channels, 
                                out_channels = node_channels)        
        self.node_processor = tg.nn.MetaLayer(node_model = NodeModel(node_channels = node_channels,
                                                                edge_channels = edge_channels,
                                                                hidden_channels = hidden_channels),
                                              edge_model = EdgeModel(node_channels = node_channels,
                                                                edge_channels = edge_channels,
                                                                hidden_channels = hidden_channels)
                                             )
        
    def forward(self, x):
        x_hat = self.node_encoder(x)
        z = th.zeros((self.num_h3_nodes, self.node_channels))
        nodes = th.cat([x_hat, z], dim = 0)
        nodes, _ , _ = self.node_processor(
            x = nodes, 
            edge_index = self.graph.edge_index,
            edge_attr = self.graph.edge_attr
        )
        out = nodes[len(self.lat_lon):]
        return out

class GraphProcessor(nn.Module):
    def __init__(self, 
                 node_channels,
                 levels,
                 num_layers = 8,
                 hidden_channels = 128,
                 edge_channels = 2
                ):
        super().__init__()
        self.graph = build_Mesh2Mesh(levels)
        self.layers = nn.ModuleList([tg.nn.MetaLayer(
            node_model = NodeModel(node_channels = node_channels,
                                    edge_channels = edge_channels,
                                    hidden_channels = hidden_channels),
            edge_model = EdgeModel(node_channels = node_channels,
                                    edge_channels = edge_channels,
                                    hidden_channels = hidden_channels)) for layer in range(num_layers)])
        
    def forward(self, x):
        z = th.zeros(self.graph.num_nodes - x.shape[0], x.shape[1])
        nodes = th.cat([z, x], dim = 0)
        edges = self.graph.edge_attr
        for processor in self.layers:
            nodes , edges , _ = processor(
                x = nodes, 
                edge_index = self.graph.edge_index,
                edge_attr = edges)
        out = nodes[z.shape[0]:]
        return out

class GraphDecoder(nn.Module):
    def __init__(self,
                 data_channels,
                 node_channels,
                 lat_lon,
                 edge_channels = 2,
                 hidden_channels = 128,
                 levels = [0, 1, 2],
                ):
        super().__init__()
        self.lat_lon = lat_lon
        self.graph = build_Mesh2Grid(lat_lon, resolution = max(levels))
        self.processor = tg.nn.MetaLayer(node_model = NodeModel(node_channels = node_channels,
                                                                edge_channels = edge_channels,
                                                                hidden_channels = hidden_channels
                                                               ),
                                         edge_model = EdgeModel(node_channels = node_channels,
                                                                edge_channels = edge_channels,
                                                                hidden_channels = hidden_channels))
        self.decoder = FFN(in_channels = node_channels, 
                           hidden_channels = hidden_channels,
                           out_channels = data_channels, 
                           use_norm = False)
        
    def forward(self, x):
        z = th.zeros((len(self.lat_lon), x.shape[1]))
        nodes = th.cat([z, x], dim = 0)
        nodes, _, _ = self.processor(x = nodes, 
                                     edge_index = self.graph.edge_index, 
                                     edge_attr = self.graph.edge_attr)
        out = self.decoder(nodes[:z.shape[0]])
        return out 

class GraphNetwork(nn.Module):
    def __init__(self,
                 data_channels,
                 node_channels,
                 lat_lon,
                 levels = [0, 1, 2],
                 num_layers = 4,
                 hidden_channels = 128,
                 edge_channels = 2
                ):
        super().__init__()
        
        self.ge = GraphEncoder(data_channels = data_channels,
                               node_channels = node_channels, 
                               hidden_channels = hidden_channels,
                               levels = levels,
                               lat_lon = lat_lon)
        self.gp = GraphProcessor(node_channels = node_channels,
                                 hidden_channels = hidden_channels,
                                 num_layers = num_layers,
                                 levels = levels)
        self.gd = GraphDecoder(data_channels = data_channels,
                               node_channels = node_channels,
                               hidden_channels = hidden_channels,
                               levels = levels,
                               lat_lon = lat_lon)
        
    def forward(self, x):
        x_hat = einops.rearrange(x, 'c h w -> (h w) c')
        x_hat = self.ge(x_hat)
        x_hat = self.gp(x_hat)
        x_hat = self.gd(x_hat)
        x_hat = einops.rearrange(x_hat, '(h w) c -> c h w', c = x.shape[0], h = x.shape[1], w = x.shape[2])
        x = x + x_hat
        return x
