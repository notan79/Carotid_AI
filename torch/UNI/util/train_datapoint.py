import torch
def train_graph(k: str, v, edges: list, model: torch.nn.Module)-> torch.Tensor:
    edges = all_edges[k]
    
    node_emb = v['enc'].squeeze(0)
    label = v['label'].squeeze(0)
    
    edge_index = torch.tensor(edges)
    
    data = Data(x=node_emb, edge_index=edge_index.t().contiguous()).to(device)
    data.validate(raise_on_error=True)
    
    cluster_data = ClusterData(data, num_parts=35, save_dir=None) # 25 too small
    
    valid_clusters = True
    
    for cluster in cluster_data:
        if cluster.num_nodes== 0 or cluster.num_edges == 0:
            print(f'Bad clusters on patient: {k}')
            valid_clusters = False
            break
            
    loader = None
    # Workaround when can't split into clusters
    if not valid_clusters:
        loader =[data] 
    else:
        loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, generator=torch.Generator(device))
    
    valid_batches = len(loader)
    logits = []
    for batch in loader:
        if batch.num_nodes == 0 or batch.num_edges == 0:
            print(f'Skipping bad batch on patient: {k}')
            valid_batches -= 1
            continue
        
        batch = batch.to(device)
        y = model(batch)
        logits.append(y)
        del batch

    if valid_batches == 0:
        print(f'Skipping patient, no valid batches: {k}')
        return None
   
    y = torch.cat(logits, dim=0).mean().unsqueeze(0)
    print(f'Mean logits: {y}')
    
    loss = class_weight[label.item()] * binary_cross_entropy(y.unsqueeze(0), label.float())
    print(f'{loss=}')
    
    del data, cluster_data, loader, node_emb, label
    torch.cuda.empty_cache()
    return loss