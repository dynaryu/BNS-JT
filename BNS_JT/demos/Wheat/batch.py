from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import geopandas as gpd
import pickle
import numpy as np
import typer
import shapely
import polyline
from scipy import stats
from itertools import islice
from typing_extensions import Annotated

from BNS_JT import model, config, trans, variable, brc, branch, cpm, operation

app = typer.Typer()

HOME = Path(__file__).parent

DAMAGE_STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']

KEYS = ['LATITUDE', 'LONGITUDE', 'HAZUS_CLASS', 'K3d', 'Kskew', 'Ishape']

GDA94 = 'EPSG:4283'  # GDA94
GDA94_ = 'EPSG:3577'


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


@app.command()
def create_shpfile(path_direction: str):

    path_direction = Path(path_direction)

    i = 0
    _dic = {}
    for _file in path_direction.glob('*-*.pk'):

        with open(_file, 'rb') as f:

            directions_result = pickle.load(f)

        for _, item in enumerate(directions_result):
            poly_line = item['overview_polyline']['points']
            geometry_points = [(x[1], x[0]) for x in polyline.decode(poly_line)]
            _dic[i] = {'distance': item['legs'][0]['distance']['text'],
                       'duration': item['legs'][0]['duration']['text']}
            _dic[i].update({'line': shapely.LineString(geometry_points),
                            'start': item['legs'][0]['start_address'],
                            'end': item['legs'][0]['end_address'],
                            })
            i += 1

    df = pd.DataFrame.from_dict(_dic).T
    df = gpd.GeoDataFrame(df, geometry='line', crs= GDA94)
    pdb.set_trace()
    df.to_file('./test.shp')

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
    dmg = []
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

            dmg.append(df_pe)

        else:
            print('Something wrong {}:{}'.format(i, _df.loc[idx]))

    dmg = pd.DataFrame(dmg)
    dmg.index = df_bridge.index
    #df_bridge = pd.concat([df_bridge, tmp], axis=1)

    # convert to edge prob
    #dmg = convert_dmg(tmp)

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
def plot_alt():

    G = nx.Graph()

    cfg = config.Config(HOME.joinpath('./config.json'))
    #cfg.infra['G'].edges()

    # read routes
    for route_file in Path('../bridge/').glob('route*.txt'):
        #print(_file)

        route = [x.strip() for x in pd.read_csv(route_file, header=None, dtype=str, )[0].to_list()]

        for item in zip(route[:-1], route[1:]):
            try:
                label = cfg.infra['G'].edges[item]['label']
            except KeyError:
                try:
                    label = cfg.infra['G'].edges[item[::-1]]['label']
                except KeyError:
                    print(f'Need to add {item}')
                else:
                    G.add_edge(item[0], item[1], label=label, key=label)
            else:
                G.add_edge(item[0], item[1], label=label, key=label)

            for i in item:
                G.add_node(i, pos=(0, 0), label=i, key=i)

    config.plot_graphviz(G, outfile=HOME.joinpath('wheat_graph_a'))
    # check against model
    print(nx.is_isomorphic(G, cfg.infra['G']))
    a = set([(x[1], x[0]) for x in G.edges]).union(G.edges)
    b = set([(x[1], x[0]) for x in cfg.infra['G'].edges]).union(cfg.infra['G'].edges)
    print(set(a).difference(b))
    print(set(b).difference(a))
    print(len(G.edges), len(cfg.infra['G'].edges))
    print(len(G.nodes), len(cfg.infra['G'].nodes))

    """
    origin = route[0]
    if origin in bridges.index:
        origin = bridges.loc[origin][['lat', 'lng']].values.tolist()

    dest = route[-1]
    if dest in bridges.index:
        dest = bridges.loc[dest][['lat', 'lng']].values.tolist()

    route_btw = [{'lat': bridges.loc[x, 'lat'], 'lng': bridges.loc[x, 'lng'], 'id': x} for x in route[1:-1]]


    # create edges

    # plot
    """


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))
    trans.plot_graph(cfg.infra['G'], HOME.joinpath('wheat.png'))
    config.plot_graphviz(cfg.infra['G'], outfile=HOME.joinpath('wheat_graph'))


@app.command()
def setup_model(key: Annotated[str, typer.Argument()] = 'Wooroloo-Merredin',
                no_paths: Annotated[int, typer.Argument()] = 5):

    thres = 2 # if takes longer than thres * d_time_itc, we consider the od pair is disconnected
    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]

    #nodes_except_const = list(cfg.infra['nodes'].keys())
    #[nodes_except_const.remove(x) for x in od_pair]

    # variables
    cpms = {}
    varis = {k: variable.Variable(name=k, values = ['f', 's'])
             for k in cfg.infra['nodes'].keys()}

    G = cfg.infra['G']
    d_time_itc = nx.shortest_path_length(G, source=od_pair[0], target=od_pair[1], weight='weight')
    #all_paths = nx.all_simple_paths(G, od_pair[0], od_pair[1])
    selected_paths = k_shortest_paths(G, source=od_pair[0], target=od_pair[1], k=no_paths, weight='weight')

    valid_paths = []
    for path in selected_paths:
        # Calculate the total weight of the path
        path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
        path_weight = sum(G[u][v]['weight'] for u, v in path_edges)
        if path_weight < thres * d_time_itc:
            # Collect the edge names if they exist, otherwise just use the edge tuple
            edge_names = [G[u][v].get('key', (u, v)) for u, v in path_edges]
            valid_paths.append((path, edge_names, path_weight))
        else:
            print(f'one of selected paths is not considered')

    # Sort valid paths by weight
    valid_paths = sorted(valid_paths, key=lambda x: x[2])

    # Print the paths with edge names and weights
    '''
    for path, edge_names, weight in valid_paths:
        print(f"Path (nodes): {path}")
        print(f"Path (edges): {edge_names}")
        print(f"Total weight: {weight}\n")
    '''
    path_names = []
    for idx, (path, edge_names, weight) in enumerate(valid_paths):
        name = '_'.join((*od_pair, str(idx)))
        path_names.append(name)

        varis[name] = variable.Variable(name=name, values = [np.inf, weight])

        n_child = len(path)

        # Event matrix of series system
        # Initialize an array of shape (n+1, n+1) filled with 1
        Cpath = np.ones((n_child + 1, n_child + 1), dtype=int)
        for i in range(1, n_child + 1):
            Cpath[i, 0] = 0
            Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
            Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

        #for i in range(1, n_edge + 1):
        ppath = np.array([1.0]*(n_child + 1))
        cpms[name] = cpm.Cpm(variables = [varis[name]] + [varis[n] for n in path], no_child=1, C=Cpath, p=ppath)

    od_name = '_'.join(od_pair)
    vals = [np.inf] + [varis[p].values[1] for p in path_names[::-1]]

    n_path = len(path_names)
    Csys = np.zeros((n_path+1, n_path+1), dtype=int)
    for i in range(n_path):
        Csys[i, 0] = n_path - i
        Csys[i, i + 1] = 1
        Csys[i, i + 2:] = 2
    psys = np.array([1.0]*(n_path+1))

    varis[od_name] = variable.Variable(name=od_name, values=vals)
    cpms[od_name] = cpm.Cpm(variables = [varis[od_name]] + [varis[p] for p in path_names], no_child=1, C=Csys, p=psys)

    # save 
    dump = {'cpms': cpms,
            'varis': varis,
            'path_names': path_names}

    #dir_path = Path(file_dmg).parent
    file_output = HOME.joinpath(f'model_{key}_{no_paths}.pk')
    with open(file_output, 'wb') as f:
        pickle.dump(dump, f)
    print(f'{file_output} saved')


@app.command()
def main(file_dmg: str,
         key: Annotated[str, typer.Argument()] = 'Wooroloo-Merredin'):

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
    '''
    for path, edge_names, weight in valid_paths:
        print(f"Path (nodes): {path}")
        print(f"Path (edges): {edge_names}")
        print(f"Total weight: {weight}\n")
    '''

    path_names = []
    for idx, (path, edge_names, weight) in enumerate(valid_paths):
        name = '_'.join((*od_pair, str(idx)))
        path_names.append(name)

        varis[name] = variable.Variable(name=name, values = [np.inf, weight])

        n_edge = len(edge_names)

        # Event matrix of series system
        # Initialize an array of shape (n+1, n+1) filled with 1
        Cpath = np.ones((n_edge + 1, n_edge + 1), dtype=int)
        for i in range(1, n_edge + 1):
            Cpath[i, 0] = 0
            Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
            Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

        #for i in range(1, n_edge + 1):
        ppath = np.array([1.0]*(n_edge + 1))
        cpms[name] = cpm.Cpm(variables = [varis[name]] + [varis[e] for e in edge_names], no_child=1, C=Cpath, p=ppath)

    od_name = '_'.join(od_pair)
    vals = [np.inf]
    for p in path_names[::-1]:
        vals.append(varis[p].values[1])

    n_path = len(path_names)
    Csys = np.zeros((n_path+1, n_path+1), dtype=int)
    for i in range(n_path):
        Csys[i, 0] = n_path - i
        Csys[i, i + 1] = 1
        Csys[i, i + 2:] = 2
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
def inference(file_model: str,
              file_dmg: str,
              key: Annotated[str, typer.Argument()] = 'Wooroloo-Merredin'):

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)
        cpms = dump['cpms']
        varis = dump['varis']
        path_names = dump['path_names']

    # assign cpms given scenario
    probs = pd.read_csv(file_dmg, index_col=0)

    probs.index = probs.index.astype('str')
    probs = probs.to_dict('index')

    for k, v in cfg.infra['nodes'].items():
        #varis[k] = variable.Variable(name=k, values = [np.inf, 1.0*v['weight']])
        try:
            pf = probs[k]['Extensive'] + probs[k]['Complete']
        except KeyError:
            pf = 0.0
        finally:
            cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [pf, 1-pf])

    od_pair = cfg.infra['ODs'][key]
    od_name = '_'.join(od_pair)
    VE_ord = list(cfg.infra['nodes'].keys()) + path_names
    vars_inf = operation.get_inf_vars(cpms, od_name, VE_ord)

    Mod = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v != od_name])
    #print(Mod)

    plt.figure()
    p_flat = Mod.p.flatten()
    elapsed_in_mins = [f'{(x/60):.1f}' for x in varis[od_name].values[::-1][:len(p_flat)]]
    plt.bar(range(len(p_flat)), p_flat[::-1], tick_label=elapsed_in_mins)

    plt.xlabel("Travel time")
    plt.ylabel("Probability")
    dir_path = Path(file_dmg).parent
    file_output = dir_path.joinpath(f'{Path(file_dmg).stem}_{key}_travel.png')
    plt.savefig(file_output, dpi=100)
    print(f'{file_output} saved')


@app.command()
def route(file_model: str,
          file_dmg: str,
          key: Annotated[str, typer.Argument()] = 'Wooroloo-Merredin'):

    cfg = config.Config(HOME.joinpath('./config.json'))

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)
        cpms = dump['cpms']
        varis = dump['varis']
        path_names = dump['path_names']

    # assign cpms given scenario
    probs = pd.read_csv(file_dmg, index_col=0)
    probs.index = probs.index.astype('str')
    probs = probs.to_dict('index')

    for k, v in cfg.infra['nodes'].items():
        #varis[k] = variable.Variable(name=k, values = [np.inf, 1.0*v['weight']])
        try:
            pf = probs[k]['Extensive'] + probs[k]['Complete']
        except KeyError:
            pf = 0.0
        finally:
            cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [pf, 1-pf])

    od_pair = cfg.infra['ODs'][key]
    od_name = '_'.join(od_pair)

    paths_name = [v.name for v in cpms[od_name].variables[1:]]

    paths_rel = []
    for path_i in paths_name:
        VE_ord = list(probs.keys()) + path_names
        vars_inf = operation.get_inf_vars(cpms, path_i, VE_ord)

        Mpath = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v!=path_i])
        paths_rel.append( Mpath.p[1][0] )

    plt.figure()
    plt.bar(range(len(paths_rel)), paths_rel, tick_label=range(len(paths_rel)))

    plt.xlabel(f"Route: {od_name}")
    plt.ylabel("Reliability")
    dir_path = Path(file_dmg).parent
    file_output = dir_path.joinpath(f'{Path(file_dmg).stem}_{key}_route.png')
    plt.savefig(file_output, dpi=100)
    print(f'{file_output} saved')


if __name__=='__main__':

    app()

