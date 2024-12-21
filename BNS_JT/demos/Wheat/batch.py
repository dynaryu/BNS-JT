from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import pickle
import numpy as np
import typer
from typing_extensions import Annotated

from BNS_JT import model, config, trans, variable, brc, branch, cpm, operation


HOME = Path(__file__).parent

app = typer.Typer()


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))

    trans.plot_graph(cfg.infra['G'], HOME.joinpath('wheat.png'))


@app.command()
def main(key: Annotated[str, typer.Argument()] = 'od1',
         max_sf: Annotated[int, typer.Argument()] = 100):

    rnd_state = np.random.RandomState(1)
    thres = 5 # if it takes longer than this, we consider the od pair is disconnected
    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]

    # randomly assign probabilities
    ## Here we assume binary states for each edge, assuming that 
    ## connectivity is the most concerned factor for emergency activities.
    probs_set = {0: {0: 0.05, 1: 0.95},
                1: {0: 0.15, 1: 0.85},
                2: {0: 0.30, 1: 0.70},
                }

    probs_key = rnd_state.choice(3, size=len(cfg.infra['edges']))

    probs = {k: probs_set[v] for k, v in zip(cfg.infra['edges'].keys(), probs_key)}

    # variables
    varis = {}
    cpms = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = [np.inf, 1.0*v['weight']])
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [probs[k][0], probs[k][1]])

    # Intact state of component vector: zero-based index
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()}
    d_time_itc, _, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, cfg.infra['G'], od_pair, varis)

    G = cfg.infra['G']
    all_paths = nx.all_simple_paths(G, od_pair[0], od_pair[1])

    valid_paths = []

    for path in all_paths:
        # Calculate the total weight of the path
        path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
        path_weight = sum(G[u][v]['weight'] for u, v in path_edges)
        if path_weight < thres * d_time_itc:
            # Collect the edge names if they exist, otherwise just use the edge tuple
            edge_names = [G[u][v].get('key', (u, v)) for u, v in path_edges]
            valid_paths.append((path, edge_names, path_weight))

    # Sort valid paths by weight
    valid_paths = sorted(valid_paths, key=lambda x: x[2])

    # Print the paths with edge names and weights
    for path, edge_names, weight in valid_paths:
        print(f"Path (nodes): {path}")
        print(f"Path (edges): {edge_names}")
        print(f"Total weight: {weight}\n")

    path_names = []
    for idx, (path, edge_names, weight) in enumerate(valid_paths):
        name = od_pair[0] + '_' + od_pair[1] + '_' + str(idx)
        path_names.append(name)

        varis[name] = variable.Variable(name=name, values = [np.inf, weight])

        n_edge = len(edge_names)

        # Event matrix of series system
        # Initialize an array of shape (n+1, n+1) filled with 1
        Cpath = np.ones((n_edge + 1, n_edge+1), dtype=int)
        # Fill the diagonal below the main diagonal with 0
        for i in range(1, n_edge + 1):
            Cpath[i, 0] = 0
            Cpath[i, i] = 0
        # Fill the lower triangular part (excluding the diagonal) with 2
        for i in range(1, n_edge + 1):
            Cpath[i, i+1:] = 2
        ppath = np.array([1.0]*(n_edge+1))

        cpms[name] = cpm.Cpm(variables = [varis[name]] + [varis[e] for e in edge_names], no_child=1, C=Cpath, p=ppath)

    od_name = '_'.join(od_pair)
    vals = [np.inf]
    for p in path_names[::-1]:
        vals.append(varis[p].values[1])

    n_path = len(path_names)
    Csys = np.zeros((n_path+1, n_path+1), dtype=int)
    for i in range(n_path):
        Csys[i, 0] = n_path - i
        Csys[i, i+1] = 1
        Csys[i, i+2:] = 2
    psys = np.array([1.0]*(n_path+1))

    varis[od_name] = variable.Variable(name=od_name, values = vals)
    cpms[od_name] = cpm.Cpm(variables = [varis[od_name]] + [varis[p] for p in path_names], no_child=1, C=Csys, p=psys)

    # save 
    dump = {'cpms': cpms,
            'varis': varis,
            'probs': probs,
            'path_names': path_names}

    with open(cfg.output_path.joinpath('dump.pk'), 'wb') as f:
        pickle.dump(dump, f)


@app.command()
def inference():

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(cfg.output_path.joinpath('dump.pk'), 'rb') as f:
        dump = pickle.load(f)
        cpms = dump['cpms']
        varis = dump['varis']
        probs = dump['probs']
        path_names = dump['path_names']

    od_pair = cfg.infra['ODs']['od1']
    od_name = '_'.join(od_pair)
    VE_ord = list(probs.keys()) + path_names
    vars_inf = operation.get_inf_vars(cpms, od_name, VE_ord)

    Mod = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v != od_name])
    #print(Mod)

    plt.figure()
    p_flat = Mod.p.flatten()
    plt.bar(range(len(p_flat)), p_flat[::-1], tick_label=varis[od_name].values[::-1])

    plt.xlabel("Travel time")
    plt.ylabel("Probability")
    plt.savefig(cfg.output_path.joinpath('travel_time.png'), dpi=100)

    #st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #csys_by_od, varis_by_od = {}, {}
    # branches by od_pair
    #for k, od_pair in cfg.infra['ODs'].items():

@app.command()
def route():

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(cfg.output_path.joinpath('dump.pk'), 'rb') as f:
        dump = pickle.load(f)
        cpms = dump['cpms']
        varis = dump['varis']
        probs = dump['probs']
        path_names = dump['path_names']

    od_pair = cfg.infra['ODs']['od1']
    od_name = '_'.join(od_pair)

    paths_name = [v.name for v in cpms[od_name].variables[1:]]

    paths_rel = []
    for path_i in paths_name:
        VE_ord = list(probs.keys()) + path_names
        vars_inf = operation.get_inf_vars( cpms, path_i, VE_ord )

        Mpath = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v!=path_i])
        paths_rel.append( Mpath.p[1][0] )

    plt.figure()
    plt.bar(range(len(paths_rel)), paths_rel, tick_label=range(len(paths_rel)))

    plt.xlabel(f"Route: {od_name}")
    plt.ylabel("Reliability")
    plt.savefig(cfg.output_path.joinpath('route.png'), dpi=100)


if __name__=='__main__':

    #create_model()
    #main()
    app()

