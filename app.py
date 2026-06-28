import os
import json
import random
import traceback
import colorsys
import pandas as pd
import numpy as np
import geopandas as gpd
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from shapely.geometry import Point
from shapely.ops import voronoi_diagram, unary_union
import matplotlib.colors as mcolors

import sys
import webbrowser
from threading import Timer

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
    base_folder = os.path.dirname(sys.executable)
else:
    app = Flask(__name__)
    base_folder = os.getcwd()

# Kept separate to allow normal clustering (no merging)
pois = [
    {"name": "Indosat KPPTI", "lat": -6.1818, "lon": 106.8223, "city": "JAKARTA PUSAT"},
    {"name": "Semanggi", "lat": -6.2197, "lon": 106.8144, "city": "JAKARTA SELATAN"},
    {"name": "Ciputra World", "lat": -6.2238, "lon": 106.8247, "city": "JAKARTA SELATAN"},
    {"name": "Soekarno Hatta Airport", "lat": -6.1255, "lon": 106.6558, "city": "KOTA TANGERANG"},
    {"name": "Ngurah Rai Airport", "lat": -8.7480, "lon": 115.1675, "city": "BADUNG"}
]

global_df = None
zone_colors = {"West": "#3498db", "South": "#e67e22", "East": "#2ecc71"}
fallback_color_map = {}
nano_color_map = {}

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        lightness = 0.4 + (i % 2) * 0.2
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_col = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_col)
    random.seed(42)
    random.shuffle(colors)
    return colors

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def rebalance_clusters(pts, labels, max_cap):
    """Move farthest sites from over-cap clusters to nearest under-cap neighbor."""
    from scipy.spatial.distance import cdist
    k = int(labels.max()) + 1
    labels = labels.copy()
    
    for _ in range(200):
        counts = np.bincount(labels, minlength=k)
        over = np.where(counts > max_cap)[0]
        if len(over) == 0:
            break
        
        for cid in over:
            while counts[cid] > max_cap:
                mask = labels == cid
                indices = np.where(mask)[0]
                centroid = pts[indices].mean(axis=0)
                dists = cdist(pts[indices], [centroid]).flatten()
                farthest_local = np.argmax(dists)
                farthest_global = indices[farthest_local]
                
                # Find nearest under-cap cluster
                best_j = -1
                best_dist = float('inf')
                for j in range(k):
                    if j != cid and counts[j] < max_cap:
                        jmask = labels == j
                        if jmask.any():
                            j_center = pts[jmask].mean(axis=0)
                            d = np.sqrt(np.sum((pts[farthest_global] - j_center)**2))
                            if d < best_dist:
                                best_dist = d
                                best_j = j
                
                if best_j == -1:
                    break
                
                labels[farthest_global] = best_j
                counts[cid] -= 1
                counts[best_j] += 1
    
    return labels

def greedy_path(df_centroids, start_idx=None):
    if len(df_centroids) <= 1:
        return df_centroids.reset_index(drop=True)
    pts = df_centroids[['LATITUDE', 'LONGITUDE']].values
    visited = [False] * len(pts)
    if start_idx is None:
        start_idx = np.argmin(pts[:, 1])
    path_indices = [start_idx]
    visited[start_idx] = True
    curr = start_idx
    for _ in range(len(pts) - 1):
        min_dist = float('inf')
        next_idx = -1
        for i in range(len(pts)):
            if not visited[i]:
                dist = haversine_dist(pts[curr][0], pts[curr][1], pts[i][0], pts[i][1])
                if dist < min_dist:
                    min_dist = dist
                    next_idx = i
        path_indices.append(next_idx)
        visited[next_idx] = True
        curr = next_idx
    return df_centroids.iloc[path_indices].copy().reset_index(drop=True)

def initialize_clustering():
    global global_df, fallback_color_map, nano_color_map
    autosave_path = os.path.join(base_folder, "autosave_df.pkl")
    if os.path.exists(autosave_path):
        print("Loading autosaved progress...")
        global_df = pd.read_pickle(autosave_path)
        
        fallback_color_map = {}
        for bid, group in global_df.groupby('big_cluster_id'):
            seq = group['big_seq_temp'].iloc[0]
            hue = (seq * 0.618033988749895) % 1.0
            saturation = 0.75 + (seq % 3) * 0.1
            lightness = 0.45 + (seq % 2) * 0.1
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            fallback_color_map[bid] = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        nano_color_map = {nid: color for nid, color in zip(global_df['nano_cluster_id'].unique(), generate_distinct_colors(global_df['nano_cluster_id'].nunique()))}
        print("Ready!")
        return

    print("Loading data and clustering...")
    df_main = pd.read_csv(os.path.join(base_folder, "NR26 (City Based).csv"), encoding='latin1')
    df_main.columns = df_main.columns.str.strip()
    df_main = df_main.dropna(subset=['LATITUDE', 'LONGITUDE', 'CITY', 'REGION']).reset_index(drop=True)
    
    if 'Unique Site ID' in df_main.columns:
        df_main['is_hvc'] = df_main['Unique Site ID'].astype(str).str.strip().str.upper() == 'TRUE'
    else:
        df_main['is_hvc'] = False

    # =========================================================
    # STEP 1: Nano Clusters first (pure geography, per CITY)
    # Standard KMeans naturally produces Voronoi cells = 100%
    # contiguous. No zone splitting at this stage.
    # =========================================================
    df_main['nano_cluster_id'] = ""
    df_main['nano_seq_num'] = -1
    df_main['big_cluster_id'] = ""
    df_main['big_seq_temp'] = -1
    df_main['ZONE'] = "None"
    
    nano_counter = 0
    for city, c_group in df_main.groupby('CITY'):
        c_idx = c_group.index
        pts = c_group[['LATITUDE', 'LONGITUDE']].values
        
        # Nano: target ~30 sites
        k_nano = max(1, int(np.ceil(len(c_group) / 30.0)))
        if k_nano <= 1:
            nano_labels = np.zeros(len(c_group), dtype=int)
        else:
            km_nano = KMeans(n_clusters=k_nano, n_init=10, random_state=42)
            nano_labels = km_nano.fit_predict(pts)
            
            # Enforce max cap of ~35 (tolerance up to 40) by splitting over-cap clusters
            max_nano_cap = 39
            while True:
                counts = np.bincount(nano_labels)
                over_cap = np.where(counts > max_nano_cap)[0]
                if len(over_cap) == 0:
                    break
                
                for cid in over_cap:
                    mask = nano_labels == cid
                    sub_pts = pts[mask]
                    if len(sub_pts) > 1:
                        km_sub = KMeans(n_clusters=2, n_init=10, random_state=42)
                        sub_labels = km_sub.fit_predict(sub_pts)
                        new_cid = nano_labels.max() + 1
                        
                        # Apply new labels to the specific over-cap cluster
                        new_assignments = np.where(sub_labels == 1, new_cid, cid)
                        nano_labels[mask] = new_assignments
        
        # Assign nano cluster IDs (temporary numeric)
        df_main.loc[c_idx, 'nano_cluster_id_num'] = nano_labels + nano_counter
        nano_counter += nano_labels.max() + 1

    # =========================================================
    # STEP 2: Group Nanos into Big Clusters (per CITY)
    # Use agglomerative clustering on nano centroids.
    # Merges geographically adjacent nanos into bigs.
    # =========================================================
    from scipy.cluster.hierarchy import fcluster, linkage
    
    big_counter = 0
    for city, c_group in df_main.groupby('CITY'):
        c_idx = c_group.index
        
        # Calculate nano centroids and sizes
        nano_info = c_group.groupby('nano_cluster_id_num').agg(
            lat=('LATITUDE', 'mean'),
            lon=('LONGITUDE', 'mean'),
            count=('SITE_ID', 'count')
        ).reset_index()
        
        if len(nano_info) <= 1:
            df_main.loc[c_idx, 'big_cluster_id_num'] = big_counter
            big_counter += 1
            continue
        
        total_sites = len(c_group)
        k_big = max(1, int(np.ceil(total_sites / 200.0)))
        
        if k_big <= 1:
            big_labels = np.zeros(len(nano_info), dtype=int)
        else:
            pts_nano = nano_info[['lat', 'lon']].values
            if len(pts_nano) > k_big:
                Z = linkage(pts_nano, method='ward')
                big_labels = fcluster(Z, t=k_big, criterion='maxclust') - 1
            else:
                big_labels = np.arange(len(pts_nano))
            
            # Enforce max cap of ~250 (tolerance up to 255) by splitting over-cap clusters
            max_big_cap = 255
            while True:
                big_counts = {}
                for idx, bid in enumerate(big_labels):
                    big_counts[bid] = big_counts.get(bid, 0) + nano_info.loc[idx, 'count']
                
                over_cap = [bid for bid, count in big_counts.items() if count > max_big_cap]
                if not over_cap:
                    break
                    
                for bid in over_cap:
                    mask = big_labels == bid
                    if sum(mask) <= 1:
                        # Cannot split a single nano cluster
                        continue
                        
                    sub_pts = pts_nano[mask]
                    km_sub = KMeans(n_clusters=2, n_init=10, random_state=42)
                    sub_labels = km_sub.fit_predict(sub_pts)
                    new_bid = big_labels.max() + 1
                    
                    new_assignments = np.where(sub_labels == 1, new_bid, bid)
                    big_labels[mask] = new_assignments
        
        nano_to_big = dict(zip(nano_info['nano_cluster_id_num'], big_labels + big_counter))
        for nid, bid in nano_to_big.items():
            df_main.loc[df_main['nano_cluster_id_num'] == nid, 'big_cluster_id_num'] = bid
        big_counter += big_labels.max() + 1

    # =========================================================
    # STEP 3: Name clusters (CITY_01, CITY_02, etc.)
    # =========================================================
    for city, c_group in df_main.groupby('CITY'):
        c_idx = c_group.index
        city_label = str(city).strip().upper()
        
        # Name big clusters
        big_ids = sorted(c_group['big_cluster_id_num'].unique())
        for idx, bid in enumerate(big_ids):
            big_name = f"{city_label}_{idx+1:02d}"
            mask = (df_main['CITY'] == city) & (df_main['big_cluster_id_num'] == bid)
            df_main.loc[mask, 'big_cluster_id'] = big_name
            
            # Name nano clusters within this big
            nano_ids_in_big = df_main.loc[mask, 'nano_cluster_id_num'].unique()
            # Sort nanos by centroid for consistent naming
            nano_pts = []
            for nid in nano_ids_in_big:
                n_group = df_main[df_main['nano_cluster_id_num'] == nid]
                nano_pts.append((nid, n_group['LATITUDE'].mean(), n_group['LONGITUDE'].mean()))
            
            if len(nano_pts) > 1:
                # Use greedy path for sequential numbering
                nano_df = pd.DataFrame(nano_pts, columns=['nid', 'LATITUDE', 'LONGITUDE'])
                nano_df = greedy_path(nano_df)
                for seq_idx, row in nano_df.iterrows():
                    nano_name = f"{big_name}_N{seq_idx+1:02d}"
                    df_main.loc[df_main['nano_cluster_id_num'] == row['nid'], 'nano_cluster_id'] = nano_name
                    df_main.loc[df_main['nano_cluster_id_num'] == row['nid'], 'nano_seq_num'] = seq_idx + 1
            else:
                nid = nano_pts[0][0]
                df_main.loc[df_main['nano_cluster_id_num'] == nid, 'nano_cluster_id'] = f"{big_name}_N01"
                df_main.loc[df_main['nano_cluster_id_num'] == nid, 'nano_seq_num'] = 1

    # =========================================================
    # STEP 4: Assign Zones (JABO only, based on Big Cluster centroids)
    # Zone never cuts through a cluster because clusters are
    # already fully formed before zone assignment.
    # =========================================================
    jabo_mask = df_main['REGION'].str.upper() == 'JABO'
    if jabo_mask.any():
        jabo_bigs = df_main[jabo_mask].groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'}).reset_index()
        
        if len(jabo_bigs) >= 3:
            km_zone = KMeans(n_clusters=3, n_init=10, random_state=42)
            jabo_bigs['zone_label'] = km_zone.fit_predict(jabo_bigs[['LATITUDE', 'LONGITUDE']])
            
            zone_centroids = jabo_bigs.groupby('zone_label').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'}).sort_values('LONGITUDE')
            west_label = zone_centroids.index[0]
            east_label = zone_centroids.index[-1]
            south_label = [l for l in zone_centroids.index if l not in (west_label, east_label)][0]
            label_to_zone = {west_label: "West", south_label: "South", east_label: "East"}
            
            big_zone_map = dict(zip(jabo_bigs['big_cluster_id'], jabo_bigs['zone_label'].map(label_to_zone)))
            for bid, zone in big_zone_map.items():
                df_main.loc[df_main['big_cluster_id'] == bid, 'ZONE'] = zone

    # =========================================================
    # STEP 5: Assign Big Cluster sequence (greedy path)
    # JABO: per zone. Non-JABO: per region.
    # =========================================================
    for (region, zone), rz_group in df_main.groupby(['REGION', 'ZONE']):
        big_centroids = rz_group.groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'}).reset_index()
        big_centroids = greedy_path(big_centroids)
        big_centroids['big_seq_temp'] = big_centroids.index + 1
        seq_map = dict(zip(big_centroids['big_cluster_id'], big_centroids['big_seq_temp']))
        for bid, seq in seq_map.items():
            df_main.loc[df_main['big_cluster_id'] == bid, 'big_seq_temp'] = seq

    # Clean up temp columns
    if 'nano_cluster_id_num' in df_main.columns:
        df_main.drop(columns=['nano_cluster_id_num'], inplace=True)
    if 'big_cluster_id_num' in df_main.columns:
        df_main.drop(columns=['big_cluster_id_num'], inplace=True)

    global_df = df_main
    fallback_color_map = {}
    for bid, group in global_df.groupby('big_cluster_id'):
        seq = group['big_seq_temp'].iloc[0]
        hue = (seq * 0.618033988749895) % 1.0
        saturation = 0.75 + (seq % 3) * 0.1
        lightness = 0.45 + (seq % 2) * 0.1
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        fallback_color_map[bid] = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    nano_color_map = {nid: color for nid, color in zip(global_df['nano_cluster_id'].unique(), generate_distinct_colors(global_df['nano_cluster_id'].nunique()))}
    
    # Save initial state
    global_df.to_pickle(os.path.join(base_folder, "autosave_df.pkl"))
    print("Ready!")

def create_seamless_voronoi(df, cluster_col):
    geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
    gdf_sites = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    try:
        hulls = [gdf_sites[gdf_sites[cluster_col] == cid].geometry.union_all().convex_hull.buffer(0.007) for cid in gdf_sites[cluster_col].unique()]
        unified_mask = gpd.GeoSeries(hulls).union_all()
        all_coords = gdf_sites.geometry.union_all()
    except AttributeError:
        hulls = [gdf_sites[gdf_sites[cluster_col] == cid].geometry.unary_union.convex_hull.buffer(0.007) for cid in gdf_sites[cluster_col].unique()]
        unified_mask = unary_union(hulls)
        all_coords = gdf_sites.geometry.unary_union
        
    vor_polys = voronoi_diagram(all_coords, envelope=all_coords.envelope.buffer(0.05))
    vor_gdf = gpd.GeoDataFrame(geometry=list(vor_polys.geoms), crs="EPSG:4326")
    joined = gpd.sjoin(vor_gdf, gdf_sites[[cluster_col, 'geometry']], how="inner", predicate="contains")
    seamless_vor = joined.dissolve(by=cluster_col).reset_index()
    seamless_vor['geometry'] = seamless_vor.geometry.intersection(unified_mask)
    return seamless_vor

@app.route('/')
def index():
    if global_df is None:
        initialize_clustering()
    return render_template('index.html')

@app.route('/data')
def get_data():
    if global_df is None:
        initialize_clustering()
    
    sites = global_df[['SITE_ID', 'SITE_NAME', 'SITE_TYPE', 'LATITUDE', 'LONGITUDE', 'is_hvc', 'REGION', 'ZONE', 'big_cluster_id', 'nano_cluster_id', 'big_seq_temp', 'nano_seq_num']].to_dict(orient='records')
    
    big_paths = []
    for (region, zone), group in global_df.groupby(['REGION', 'ZONE']):
        big_centroids = group.groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean', 'big_seq_temp':'first'}).sort_values('big_seq_temp').reset_index()
        pts = big_centroids[['LATITUDE', 'LONGITUDE', 'big_seq_temp', 'big_cluster_id']].to_dict(orient='records')
        color = zone_colors.get(zone, "black") if region.upper() == 'JABO' else "black"
        big_paths.append({'region': region, 'zone': zone, 'color': color, 'points': pts})
        
    nano_paths = []
    for b_id in global_df['big_cluster_id'].unique():
        b_group = global_df[global_df['big_cluster_id'] == b_id]
        nano_centroids = b_group.groupby('nano_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean', 'nano_seq_num':'first'}).sort_values('nano_seq_num').reset_index()
        pts = nano_centroids[['LATITUDE', 'LONGITUDE', 'nano_seq_num', 'nano_cluster_id']].to_dict(orient='records')
        nano_paths.append({'big_id': b_id, 'color': fallback_color_map[b_id], 'points': pts})
        
    return jsonify({
        'sites': sites,
        'big_paths': big_paths,
        'nano_paths': nano_paths,
        'zone_colors': zone_colors,
        'fallback_colors': fallback_color_map,
        'nano_colors': nano_color_map
    })

@app.route('/autosave', methods=['POST'])
def autosave():
    global global_df
    data = request.json
    
    site_edits = data.get('site_edits', {})
    for site_id, edits in site_edits.items():
        for field, new_val in edits.items():
            if field in ['REGION', 'ZONE', 'big_cluster_id', 'nano_cluster_id']:
                global_df.loc[global_df['SITE_ID'] == site_id, field] = new_val
                
    big_seq = data.get('big_seq', {})
    for bid, seq in big_seq.items():
        global_df.loc[global_df['big_cluster_id'] == bid, 'big_seq_temp'] = int(seq)
        
    nano_seq = data.get('nano_seq', {})
    for nid, seq in nano_seq.items():
        global_df.loc[global_df['nano_cluster_id'] == nid, 'nano_seq_num'] = int(seq)
        
    global_df.to_pickle(os.path.join(base_folder, "autosave_df.pkl"))
    return jsonify({"status": "success"})

@app.route('/reset', methods=['POST'])
def reset_progress():
    global global_df
    autosave_path = os.path.join(base_folder, "autosave_df.pkl")
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    global_df = None
    initialize_clustering()
    return jsonify({"status": "success"})

@app.route('/export', methods=['POST'])
def export_data():
    global global_df
    try:
        print("Generating revised exports...")
        
        revised_folder = os.path.join(base_folder, "Revised")
        if not os.path.exists(revised_folder):
            os.makedirs(revised_folder)
            
        summary_big = global_df.groupby(['REGION', 'ZONE', 'CITY', 'big_cluster_id', 'big_seq_temp']).agg(total_sites=('SITE_ID', 'count'), hvc_count=('is_hvc', 'sum')).reset_index()
        summary_nano = global_df.groupby(['REGION', 'ZONE', 'CITY', 'big_cluster_id', 'big_seq_temp', 'nano_cluster_id', 'nano_seq_num']).agg(total_sites=('SITE_ID', 'count')).reset_index()

        with pd.ExcelWriter(os.path.join(revised_folder, "5G_Final_Revised_Report.xlsx")) as writer:
            summary_big.to_excel(writer, sheet_name='Summary', index=False)
            global_df[['SITE_ID', 'SITE_NAME', 'Unique Site ID', 'SITE_TYPE', 'LONGITUDE', 'LATITUDE', 'REGION', 'ZONE', 'CITY', 'big_seq_temp', 'big_cluster_id', 'nano_seq_num', 'nano_cluster_id']].to_excel(writer, sheet_name='Site_Details', index=False)
            summary_nano.to_excel(writer, sheet_name='Nano_Cluster_Summary', index=False)

        try:
            big_gdf = create_seamless_voronoi(global_df, 'big_cluster_id')
            # Merge Big Cluster attributes (Sequence, Region, Zone, City)
            big_attrs = global_df[['big_cluster_id', 'REGION', 'ZONE', 'CITY', 'big_seq_temp']].drop_duplicates(subset=['big_cluster_id'])
            big_gdf = big_gdf.merge(big_attrs, on='big_cluster_id', how='left')
            big_gdf.to_file(os.path.join(revised_folder, "Big_Cluster_Border.tab"), driver="MapInfo File")
            
            nano_gdf = create_seamless_voronoi(global_df, 'nano_cluster_id')
            # Merge Nano and Big Cluster attributes into the Nano polygons
            nano_attrs = global_df[['nano_cluster_id', 'big_cluster_id', 'REGION', 'ZONE', 'CITY', 'big_seq_temp', 'nano_seq_num']].drop_duplicates(subset=['nano_cluster_id'])
            nano_gdf = nano_gdf.merge(nano_attrs, on='nano_cluster_id', how='left')
            nano_gdf.to_file(os.path.join(revised_folder, "Nano_Cluster_Border.tab"), driver="MapInfo File")
            
            # The Nano_Cluster_Border essentially IS the combined polygon (Nano level geometry + Big/Nano attributes)
            # We'll save a copy named clearly as Combined_Polygons
            nano_gdf.to_file(os.path.join(revised_folder, "All_Clusters_Combined_Polygons.tab"), driver="MapInfo File")
            
        except Exception as e:
            print(f"Voronoi/TAB export failed: {e}")

        import folium
        from branca.element import Template, MacroElement
        m = folium.Map(location=[global_df['LATITUDE'].mean(), global_df['LONGITUDE'].mean()], zoom_start=11, tiles='CartoDB Positron')

        fg_points = folium.FeatureGroup(name="Sites")
        for _, row in global_df.iterrows():
            color = fallback_color_map[row['big_cluster_id']]
            popup_html = f"Site ID: {row['SITE_ID']}<br>Name: {row['SITE_NAME']}<br>Type: {row['SITE_TYPE']}<br>Big Cluster: {row['big_cluster_id']}<br>Nano Cluster: {row['nano_cluster_id']}"
            folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']], radius=4, color=color, fill=True, fill_opacity=0.6, popup=popup_html).add_to(fg_points)
        fg_points.add_to(m)

        fg_path_big = folium.FeatureGroup(name="Big Cluster Path", show=True)
        for (region, zone), group in global_df.groupby(['REGION', 'ZONE']):
            big_centroids = group.groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean', 'big_seq_temp':'first'}).sort_values('big_seq_temp')
            pts = big_centroids[['LATITUDE', 'LONGITUDE']].values.tolist()
            
            color = "black"
            if region.upper() == 'JABO' and zone in zone_colors:
                color = zone_colors[zone]
                
            if region.upper() == 'JABO':
                folium.PolyLine(pts, color="black", weight=9, opacity=0.9).add_to(fg_path_big)
                folium.PolyLine(pts, color=color, weight=5, opacity=1.0).add_to(fg_path_big)
            else:
                folium.PolyLine(pts, color=color, weight=5, opacity=0.8).add_to(fg_path_big)
                
            for _, p in big_centroids.iterrows():
                folium.Marker(location=[p['LATITUDE'], p['LONGITUDE']], icon=folium.DivIcon(html=f'''<div style="font-size: 8pt; color: white; background: {color}; border-radius: 4px; width: 22px; height: 16px; text-align: center; line-height: 16px; border: 1px solid white; font-weight: bold;">{int(p["big_seq_temp"])}</div>''')).add_to(fg_path_big)
        fg_path_big.add_to(m)

        m.save(os.path.join(revised_folder, "5G_Final_Revised_Map.html"))
        return jsonify({"status": "success", "msg": f"Exported successfully to Revised folder"})
    except Exception as e:
        print(f"Export Error: {traceback.format_exc()}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")
    Timer(1.5, open_browser).start()
    initialize_clustering()
    app.run(host='0.0.0.0', port=5000)
