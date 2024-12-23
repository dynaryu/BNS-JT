from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import pickle
import numpy as np
import typer
from scipy import stats
from typing_extensions import Annotated

from BNS_JT import model, config, trans, variable, brc, branch, cpm, operation

app = typer.Typer()

HOME = Path(__file__).parent

DAMAGE_STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']

KEYS = ['LATITUDE', 'LONGITUDE', 'HAZUS_CLASS', 'K3d', 'Kskew', 'Ishape']


def get_kshape(Ishape, sa03, sa10):
    """
    sa03, sa10 can be vector
    """
    if isinstance(sa03, float):
        if Ishape:
            Kshape = np.min([1.0, 2.5*sa10/sa03])
        else:
            Kshape  = 1.0
    else:
        if Ishape:
            Kshape = np.minimum(1.0, 2.5*sa10/sa03)
        else:
            Kshape  = np.ones_like(sa03)
    return Kshape


def compute_pe_by_ds(ps, kshape, Kskew, k3d, sa10):

    factor = {'Slight': kshape,
              'Moderate': Kskew*k3d,
              'Extensive': Kskew*k3d,
              'Complete': Kskew*k3d}

    return pd.Series({ds: stats.lognorm.cdf(sa10, 0.6, scale=factor[ds]*ps[ds])
            for ds in DAMAGE_STATES})


@app.command()
def dmg_bridge(file_bridge: str, file_gm: str):

    # HAZUS Methodology
    bridge_param = pd.read_csv(
        HOME.joinpath('bridge_classification_damage_params.csv'), index_col=0)

    # read sitedb data 
    df_bridge = pd.read_csv(file_bridge, index_col=0)[KEYS].copy()

    # read gm
    gm = pd.read_csv(file_gm, index_col=0, skiprows=1)

    # weighted
    tmp = []
    for i, row in df_bridge.iterrows():
        _df = (gm['lat']-row['LATITUDE'])**2 + (gm['lon']-row['LONGITUDE'])**2
        idx = _df.idxmin()

        if _df.loc[idx] < 0.01:
            row['SA03'] = gm.loc[idx, 'gmv_SA(0.3)']
            row['SA10'] = gm.loc[idx, 'gmv_SA(1.0)']
            row['Kshape'] = get_kshape(row['Ishape'], row['SA03'], row['SA10'])

            df_pe = compute_pe_by_ds(
                bridge_param.loc[row['HAZUS_CLASS']], row['Kshape'], row['Kskew'], row['K3d'], row['SA10'])

            #df_pb = get_pb_from_pe(df_pe)
            #df_pe['Kshape'] = row['Kshape']
            #df_pe['SA10'] = row['SA10']

            tmp.append(df_pe)

        else:
            print('Something wrong {}:{}'.format(i, _df.loc[idx]))

    tmp = pd.DataFrame(tmp)
    tmp.index = df_bridge.index
    #df_bridge = pd.concat([df_bridge, tmp], axis=1)

    # convert to edge prob
    dmg = convert_dmg(tmp)

    dir_path = Path(file_gm).parent
    file_output = dir_path.joinpath(Path(file_gm).stem + '_dmg.csv')
    dmg.to_csv(file_output)
    print(f'{file_output} saved')

    return dmg


def convert_dmg(dmg):

    cfg = config.Config(HOME.joinpath('./config.json'))

    tmp = {}
    for k, v in cfg.infra['edges'].items():

        try:
            p0 = 1 - dmg.loc[v['origin']]['Extensive']
        except KeyError:
            p0 = 1.0

        try:
            p1 = 1 - dmg.loc[v['destination']]['Extensive']
        except KeyError:
            p0 = 1.0
        finally:
            tmp[k] = {'F': 1 - p0 * p1}

    tmp = pd.DataFrame(tmp).T
    tmp['S'] = 1 - tmp['F']

    return tmp


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))

    trans.plot_graph(cfg.infra['G'], HOME.joinpath('wheat.png'))


def generate_default():

    rnd_state = np.random.RandomState(1)
    cfg = config.Config(HOME.joinpath('./config.json'))

    # randomly assign probabilities
    ## Here we assume binary states for each edge, assuming that 
    ## connectivity is the most concerned factor for emergency activities.
    probs_set = {0: {0: 0.05, 1: 0.95},
                1: {0: 0.15, 1: 0.85},
                2: {0: 0.30, 1: 0.70},
                }

    probs_key = rnd_state.choice(3, size=len(cfg.infra['edges']))

    probs = {k: probs_set[v] for k, v in zip(cfg.infra['edges'].keys(), probs_key)}

    pd.DataFrame(probs).T.to_csv(HOME.joinpath('random.csv'))


@app.command()
def main(file_dmg: str,
         key: Annotated[str, typer.Argument()] = 'od1'):

    rnd_state = np.random.RandomState(1)
    thres = 5 # if it takes longer than this, we consider the od pair is disconnected
    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]

    # randomly assign probabilities
    probs = pd.read_csv(file_dmg, index_col=0)
    probs.index = probs.index.astype('str')
    probs = probs.to_dict('index')

    # variables
    varis = {}
    cpms = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = [np.inf, 1.0*v['weight']])
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [probs[k]['F'], probs[k]['S']])

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

    dir_path = Path(file_dmg).parent
    file_output = dir_path.joinpath(Path(file_dmg).stem + '_dump.pk')
    with open(file_output, 'wb') as f:
        pickle.dump(dump, f)
    print(f'{file_output} saved')


@app.command()
def inference(file_dump):

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(file_dump, 'rb') as f:
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
    dir_path = Path(file_dump).parent
    file_output = dir_path.joinpath(Path(file_dump).stem + '_travel.png')
    plt.savefig(file_output, dpi=100)

    #st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #csys_by_od, varis_by_od = {}, {}
    # branches by od_pair
    #for k, od_pair in cfg.infra['ODs'].items():

@app.command()
def route(file_dump):

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(file_dump, 'rb') as f:
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
    dir_path = Path(file_dump).parent
    file_output = dir_path.joinpath(Path(file_dump).stem + '_route.png')
    plt.savefig(file_output, dpi=100)


if __name__=='__main__':

    #create_model()
    #main()
    app()

