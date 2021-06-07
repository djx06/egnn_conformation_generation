import torch
import numpy as np
from torch import nn
import torch.autograd as autograd


def kabsch(pos: torch.Tensor, fit_pos: torch.Tensor, batch_size, n_nodes, mol_node_matrix: torch.Tensor=None, use_cuda=False) \
        -> (torch.Tensor, torch.Tensor):
    if mol_node_matrix is None:
        mol_node_matrix = torch.ones([1, pos.shape[0]]).type(torch.float32)
    pos_list = []
    fit_pos_list = []
    ones = torch.ones(n_nodes)
    if use_cuda:
        ones = ones.cuda()
    # for mask in mol_node_matrix:
    for i in range(batch_size):
        mask = torch.zeros(mol_node_matrix.shape[0])
        if use_cuda:
            mask = mask.cuda()

        # print(ones.shape, mol_node_matrix[i*n_nodes:(i+1)*n_nodes,:].squeeze().shape)
        mask[i*n_nodes:(i+1)*n_nodes] = ones*mol_node_matrix[i*n_nodes:(i+1)*n_nodes,:].squeeze()
        n = torch.sum(mask)
        p0 = pos[mask > 0, :]
        q0 = fit_pos[mask > 0, :]
        p = p0 - torch.sum(p0, dim=0) / n
        q = q0 - torch.sum(q0, dim=0) / n
        c = p.t() @ q
        det = torch.det(c)
        v, s, w = torch.svd(c)
        rd1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        rd2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
        if use_cuda:
            rd1 = rd1.cuda()
            rd2 = rd2.cuda()
        r1 = w @ rd1 @ v.t()
        r2 = w @ rd2 @ v.t()
        p1 = p @ r1
        p2 = p @ r2
        nd1 = torch.norm(p1 - q)
        nd2 = torch.norm(p2 - q)
        if det > 1e-5:
            pos_list.append(p1)
        elif det < -1e-5:
            pos_list.append(p2)
        else:
            if nd1 < nd2:
                pos_list.append(p1.detach())
            else:
                pos_list.append(p2.detach())

        ConstantPad = nn.ConstantPad2d(padding=(0, 0, 0, n_nodes-q.shape[0]), value=0)
        q = ConstantPad(q)
        p = ConstantPad(pos_list[-1])

        pos_list[-1] = p
        fit_pos_list.append(q)

    ret_pos = torch.cat(pos_list, dim=0)
    ret_fit_pos = torch.cat(fit_pos_list, dim=0)
    return ret_pos, ret_fit_pos


def rmsd(src: torch.Tensor, tgt: torch.Tensor, mass: torch.Tensor=None, use_cuda=False) -> torch.Tensor:
    if mass is None:
        mass = torch.ones([src.shape[0], 1]).type(torch.float32)
    if use_cuda:
        mass = mass.cuda()
    md2 = mass * torch.pow(src - tgt, 2).sum(dim=1, keepdim=True)
    loss = torch.sqrt(md2.sum() / mass.sum())
    return loss

def rmsd_loss(src: torch.Tensor, tgt: torch.Tensor, batch_size, n_nodes, mass: torch.Tensor=None, use_cuda=False) -> torch.Tensor:
    if mass is None:
        mass = torch.ones([src.shape[0], 1]).type(torch.float32)
    mass = mass.view(batch_size, n_nodes).unsqueeze(-1)
    n = mass.sum(dim=(1,2))
    if use_cuda:
        mass = mass.cuda()
        n = n.cuda()

    src = src.view(batch_size, n_nodes, -1)
    tgt = tgt.view(batch_size, n_nodes, -1)

    # print(src.shape, mass.shape)

    md2 = (mass * torch.pow(src - tgt, 2)).sum(dim=(1,2))
    loss = (torch.sqrt(md2/n).sum())/batch_size

    return loss

def distance_loss(src: torch.Tensor, tgt: torch.Tensor, edge_index, batch_size, n_nodes, node_mask, use_cuda=False):
    row, col = edge_index
    coord_diff_diff = torch.pow(torch.pow(src[row] - src[col], 2).sum(dim=1) - torch.pow(tgt[row] - tgt[col],2).sum(dim=1), 2)
    coord_diff_diff = coord_diff_diff.view(batch_size, n_nodes**2)
    n = node_mask.view(batch_size, n_nodes).sum(dim=1)
    loss = (coord_diff_diff.sum(dim=1)/(n*n)).sum() / batch_size

    return loss


if __name__ == '__main__':
    pos = torch.tensor([
        [1.2872716402317572, 0.10787202861021278, 0.0],
        [-0.09007753136792773, -0.40715148832140396, 0.0],
        [-1.1971941088638294, -0.8382876125923721, 0.0],
        [1.2872716402317572, 0.10787202861021278, 0.0],
        [-0.09007753136792773, -0.40715148832140396, 0.0],
        [-1.1971941088638294, -0.8382876125923721, 0.0],
        [0.7520094407284719, 0.0, 0.0],
        [-0.7520094407284719, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    fit_pos = torch.tensor([
        [-0.0178, 1.4644, 0.0101],
        [0.0021, 0.0095, 0.0020],
        [0.0183, -1.1918, -0.0045],
        [-1.2872716402317572, 0.10787202861021278, 0.0],
        [0.09007753136792773, -0.40715148832140396, 0.0],
        [1.1971941088638294, -0.8382876125923721, 0.0],
        [-0.0187, 1.5256, 0.0104],
        [0.0021, -0.0039, 0.0020],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    mnm = torch.tensor([
        # [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ], dtype=torch.float32)
    # print(mnm.shape)
    node_mask = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1], [0]]).to(dtype=torch.float32)
    pos, fit_pos = kabsch(pos, fit_pos, 3, 3, node_mask)
    np.set_printoptions(precision=3, suppress=True)
    # print(pos.numpy())
    # print(fit_pos.numpy())

    r = rmsd(pos, fit_pos, torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1], [0]], dtype=torch.float32))
    print(r)
    r = rmsd_loss(pos, fit_pos, 3, 3, node_mask)
    print(r)
    edge_index = [torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8]),
                  torch.tensor([0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,6,7,8,6,7,8,6,7,8])]
    d = distance_loss(pos, fit_pos, edge_index, 3, 3, node_mask)
    print(d)