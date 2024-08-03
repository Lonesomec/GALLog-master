
import sys

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def vgae_train(gdata, v_model, opt, device, bs):
    v_model = v_model.to(device)
    v_model.train()
    criterion = nn.MSELoss()
    train_loader = DataLoader(gdata, batch_size=bs, num_workers=0, pin_memory=False)
    for epoch in range(100):
        total_loss = 0
        for real_data in train_loader:

            real_data = real_data.to(device)
            real_features = real_data.x
            edge_index = real_data.edge_index

            opt.zero_grad()
            # Forward pass
            z = v_model.encode(real_features, edge_index)
            recon_loss = v_model.rec_loss_small(real_features, z, edge_index, criterion)
            kl_loss = 0
            v_loss = recon_loss + kl_loss
            # Backward pass and optimization
            if torch.isnan(v_loss):
                print(real_data)
                print(z)
                print(recon_loss)
                print(kl_loss)
                sys.exit()
            total_loss += v_loss.item()
            v_loss.backward()
            opt.step()
        # if (epoch + 1) % 10 == 0:
        #     print(f'VGAE loss :{total_loss / len(gdata)}')
    return v_model


def vgae_generate(gdata, model, device):
    model.eval()
    generated_gdata = []
    for g in gdata:
        g = g.to(device)
        real_features = g.x
        edge_index = g.edge_index
        with torch.no_grad():
            z = model.encode(real_features, edge_index)
            generated_features = model.decoder(z)
        generated_gdata.append(Data(x=generated_features, edge_index=edge_index, y=g.y).cpu())
    return generated_gdata
