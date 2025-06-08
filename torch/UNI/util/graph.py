# Return a list of adjacent nodes from index on image split into patches: num_w x num_h
def get_adj(idx: int, num_w: int, num_h: int) -> list:
    row = idx // num_w
    col = idx % num_w
    
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1,1) ]
    
    adj = []
    for dir_r, dir_c in directions:
        r, c = row+dir_r, col + dir_c
        
        if (r>=0 and r < num_h) and (c >= 0 and c < num_w):
            tmp_idx = r * num_w + c
            adj.append(tmp_idx)
    return adj