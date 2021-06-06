from qm9 import dataset
from qm9.models import EGNN
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
from kabsch import kabsch, rmsd, distance_loss

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)

model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device,
                 n_layers=args.n_layers, coords_weight=1.0, attention=args.attention, node_attr=args.node_attr)

# print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()

def my_loss(atom_position, atom_position_pred, batch_size, n_nodes):
    position_list = atom_position.split(n_nodes, dim=0)
    position_pred_list = atom_position_pred.split(n_nodes, dim=0)
    loss = torch.tensor(0).type(torch.float32)
    total_rmsd_loss = torch.tensor(0).type(torch.float32)
    total_dis_loss = torch.tensor(0).type(torch.float32)

    for i, data in enumerate(zip(position_list,position_pred_list)):
        non_zero_rows = torch.abs(data[0]).sum(dim=1) > 0
        pos, fit_pos = kabsch(data[0][non_zero_rows], data[1][non_zero_rows], use_cuda=False)
        rmsd_loss = rmsd(pos, fit_pos)
        dis_loss = distance_loss(pos, fit_pos)
        total_rmsd_loss += rmsd_loss
        total_dis_loss += dis_loss
        loss += rmsd_loss+dis_loss

    loss = loss/batch_size
    total_rmsd_loss = total_rmsd_loss/batch_size
    total_dis_loss = total_dis_loss/batch_size
    return loss, total_rmsd_loss, total_dis_loss


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'rmsd_loss':0, 'dis_loss':0, 'counter': 0, 'loss_arr':[]}
    print(len(loader))
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, dim_coord = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_positions_random = torch.Tensor(batch_size * n_nodes, dim_coord)
        torch.nn.init.normal_(atom_positions_random, mean=0, std=1)
        # print(atom_positions.shape)
        # print(atom_positions_random.shape)
        # print(atom_positions)
        # print(atom_positions_random)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        # print(n_nodes,batch_size)
        # print(len(edges),edges[0].shape,edges[1].shape)
        # print(edges)
        label = data[args.property].to(device, dtype)

        pred = model(h0=nodes, x=atom_positions_random, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        loss, rmsd_loss, dis_loss = my_loss(atom_positions, pred, batch_size, n_nodes)
        if partition == 'train':
            # loss = loss_l1(pred, (label - meann) / mad)
            # loss = my_loss(atom_positions, pred, batch_size, n_nodes)
            loss.backward()
            optimizer.step()
        # else:
        #     # loss = loss_l1(mad * pred + meann, label)
        #     loss = my_loss(atom_positions, pred, batch_size, n_nodes)


        res['loss'] += loss.item() * batch_size
        res['rmsd_loss'] += rmsd_loss.item() * batch_size
        res['dis_loss'] += dis_loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append([loss.item(),rmsd_loss.item(),dis_loss.item()])

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:][0])/len(res['loss_arr'][-10:][0])))
    return res['loss'] / res['counter'], res['rmsd_loss']/res['counter'], res['dis_loss']/res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': [], 'best_test': [], 'best_epoch': 0}

    for epoch in range(0, args.epochs):
        train(epoch, dataloaders['train'], partition='train')
        if epoch % args.test_interval == 0:
            val_loss, val_rmsd_loss, val_dis_loss = train(epoch, dataloaders['valid'], partition='valid')
            test_loss, test_rmsd_loss, test_dis_loss = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append([test_loss, test_rmsd_loss, test_dis_loss])

            if val_loss < res['best_val']:
                res['best_val'] = [val_loss, val_rmsd_loss, val_dis_loss]
                res['best_test'] = [test_loss, test_rmsd_loss, test_dis_loss]
                res['best_epoch'] = epoch
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'][0], res['best_test'][0], res['best_epoch']))
            print("Best Rmsd: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'][1], res['best_test'][1], res['best_epoch']))
            print("Best Dis: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'][2], res['best_test'][2], res['best_epoch']))


        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)

    torch.save(model, '/model.pkl')