import pandas as pd
import numpy as np
import os
import folium
import geopandas as gpd
from folium.features import RegularPolygonMarker
from sklearn.cluster import KMeans
from shapely.geometry import Point
from shapely.ops import voronoi_diagram, unary_union
from branca.element import Template, MacroElement
import warnings
warnings.filterwarnings('ignore')

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def greedy_path(df_centroids, start_idx=None):
    if len(df_centroids) <= 1:
        return df_centroids.reset_index(drop=True)
    
    pts = df_centroids[['LATITUDE', 'LONGITUDE']].values
    visited = [False] * len(pts)
    
    if start_idx is None:
        start_idx = np.argmin(pts[:, 1]) # start from westernmost
        
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
        
    res = df_centroids.iloc[path_indices].copy().reset_index(drop=True)
    return res

base_folder = "C:/Request/NR26 Cluster"
result_folder = os.path.join(base_folder, "Output")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

df_main = pd.read_csv(os.path.join(base_folder, "NR26 (City Based).csv"), encoding='latin1')
df_main.columns = df_main.columns.str.strip()
df_main = df_main.dropna(subset=['LATITUDE', 'LONGITUDE', 'CITY', 'REGION']).reset_index(drop=True)

if 'Unique Site ID' in df_main.columns:
    df_main['is_hvc'] = df_main['Unique Site ID'].astype(str).str.strip().str.upper() == 'TRUE'
else:
    df_main['is_hvc'] = False



df_main['ZONE'] = "None"
df_main['big_cluster_id_temp'] = -1
df_main['nano_cluster_id_temp'] = -1

# Identify JABO 3 Zones
jabo_mask = df_main['REGION'].str.upper() == 'JABO'
jabo_centroid = (0,0)
zone_colors = {}
if jabo_mask.any():
    jabo_sites = df_main[jabo_mask].copy()
    jabo_centroid = (jabo_sites['LATITUDE'].mean(), jabo_sites['LONGITUDE'].mean())
    
    km_jabo = KMeans(n_clusters=3, n_init=10, random_state=42)
    jabo_sites['zone_label'] = km_jabo.fit_predict(jabo_sites[['LATITUDE', 'LONGITUDE']])
    
    centroids = jabo_sites.groupby('zone_label').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'})
    # Sort centroids by Longitude
    centroids = centroids.sort_values('LONGITUDE')
    west_label = centroids.index[0] # Smallest Longitude
    east_label = centroids.index[-1] # Largest Longitude
    south_label = [l for l in centroids.index if l not in (west_label, east_label)][0] # Middle Longitude, usually South in JABO context
    
    label_to_zone = {west_label: "West", south_label: "South", east_label: "East"}
    jabo_sites['ZONE'] = jabo_sites['zone_label'].map(label_to_zone)
    df_main.loc[jabo_mask, 'ZONE'] = jabo_sites['ZONE']
    
    zone_colors = {"West": "#3498db", "South": "#e67e22", "East": "#2ecc71"} # Blue, Orange, Green

big_cluster_counter_global = 0
all_results = []

for region, r_group in df_main.groupby('REGION'):
    for zone, z_group in r_group.groupby('ZONE'):
        for city, c_group in z_group.groupby('CITY'):
            c_group = c_group.copy()
            unassigned = c_group.index.tolist()
            
            # B) Remaining Sites K-Means for Big Clusters
            if unassigned:
                rem_sites = c_group.loc[unassigned]
                num_big = int(np.ceil(len(rem_sites) / 200.0))
                if num_big == 1:
                    c_group.loc[unassigned, 'big_cluster_id_temp'] = big_cluster_counter_global
                    big_cluster_counter_global += 1
                else:
                    km = KMeans(n_clusters=num_big, n_init=10, random_state=42)
                    labels = km.fit_predict(rem_sites[['LATITUDE', 'LONGITUDE']])
                    for i in range(num_big):
                        sub_idx = rem_sites[labels == i].index
                        c_group.loc[sub_idx, 'big_cluster_id_temp'] = big_cluster_counter_global
                        big_cluster_counter_global += 1
                        
            # Map to CITY_XX
            c_bigs = c_group['big_cluster_id_temp'].unique()
            big_id_map = {}
            for idx, b_id in enumerate(c_bigs):
                big_id_map[b_id] = f"{str(city).strip().upper()}_{idx+1:02d}"
            c_group['big_cluster_id'] = c_group['big_cluster_id_temp'].map(big_id_map)
            all_results.append(c_group)

df_all = pd.concat(all_results).reset_index(drop=True)
df_all['big_seq_temp'] = -1
df_all['nano_seq_num'] = -1
df_all['nano_cluster_id'] = ""

# C) Sequence Big Clusters per Zone/Region
final_chunks = []
for (region, zone), rz_group in df_all.groupby(['REGION', 'ZONE']):
    rz_group = rz_group.copy()
    big_centroids = rz_group.groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'}).reset_index()
    
    if region.upper() == 'JABO':
        # Sort by distance to JABO centroid (descending) so they converge
        big_centroids['dist_to_center'] = haversine_dist(big_centroids['LATITUDE'], big_centroids['LONGITUDE'], jabo_centroid[0], jabo_centroid[1])
        big_centroids = big_centroids.sort_values('dist_to_center', ascending=False).reset_index(drop=True)
        big_centroids['big_seq_temp'] = big_centroids.index + 1
    else:
        # Greedy TSP path
        big_centroids = greedy_path(big_centroids)
        big_centroids['big_seq_temp'] = big_centroids.index + 1
        
    seq_map = dict(zip(big_centroids['big_cluster_id'], big_centroids['big_seq_temp']))
    rz_group['big_seq_temp'] = rz_group['big_cluster_id'].map(seq_map)
    
    # Nano Clusters
    nano_cluster_counter = 0
    for b_id in rz_group['big_cluster_id'].unique():
        b_group = rz_group[rz_group['big_cluster_id'] == b_id].copy()
        num_nano = int(np.ceil(len(b_group) / 30.0))
        
        if num_nano == 1:
            b_group['nano_cluster_id_temp'] = nano_cluster_counter
            nano_cluster_counter += 1
        else:
            km = KMeans(n_clusters=num_nano, n_init=10, random_state=42)
            labels = km.fit_predict(b_group[['LATITUDE', 'LONGITUDE']])
            b_group['nano_cluster_id_temp'] = labels + nano_cluster_counter
            nano_cluster_counter += num_nano
            
        nano_centroids = b_group.groupby('nano_cluster_id_temp').agg({'LATITUDE':'mean', 'LONGITUDE':'mean'}).reset_index()
        
        if region.upper() == 'JABO':
            nano_centroids['dist_to_center'] = haversine_dist(nano_centroids['LATITUDE'], nano_centroids['LONGITUDE'], jabo_centroid[0], jabo_centroid[1])
            nano_centroids = nano_centroids.sort_values('dist_to_center', ascending=False).reset_index(drop=True)
        else:
            nano_centroids = greedy_path(nano_centroids)
            
        nano_centroids['nano_seq_num'] = nano_centroids.index + 1
        
        nano_id_map = {}
        for _, n_row in nano_centroids.iterrows():
            n_idx = int(n_row['nano_seq_num'])
            nano_id_map[n_row['nano_cluster_id_temp']] = f"{b_id}_N{n_idx:02d}"
            
        b_group['nano_cluster_id'] = b_group['nano_cluster_id_temp'].map(nano_id_map)
        n_seq_map = dict(zip(nano_centroids['nano_cluster_id_temp'], nano_centroids['nano_seq_num']))
        b_group['nano_seq_num'] = b_group['nano_cluster_id_temp'].map(n_seq_map)
        
        rz_group.update(b_group[['nano_cluster_id', 'nano_seq_num']])
    
    final_chunks.append(rz_group)

df_final = pd.concat(final_chunks).reset_index(drop=True)

# 3. EXPORTS
summary_big = df_final.groupby(['REGION', 'ZONE', 'CITY', 'big_cluster_id']).agg(total_sites=('SITE_ID', 'count'), hvc_count=('is_hvc', 'sum')).reset_index()
summary_nano = df_final.groupby(['REGION', 'ZONE', 'CITY', 'big_cluster_id', 'nano_cluster_id', 'nano_seq_num']).agg(total_sites=('SITE_ID', 'count')).reset_index()

with pd.ExcelWriter(os.path.join(result_folder, "5G_Final_Revised_Report.xlsx"), engine='xlsxwriter') as writer:
    summary_big.to_excel(writer, sheet_name='Summary', index=False)
    df_final[['SITE_ID', 'SITE_NAME', 'Unique Site ID', 'SITE_TYPE', 'LONGITUDE', 'LATITUDE', 'REGION', 'ZONE', 'CITY', 'big_seq_temp', 'big_cluster_id', 'nano_seq_num', 'nano_cluster_id']].to_excel(writer, sheet_name='Site_Details', index=False)
    summary_nano.to_excel(writer, sheet_name='Nano_Cluster_Summary', index=False)

# SHAPEFILES (Seamless Voronoi)
def create_seamless_voronoi(df, cluster_col):
    geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
    gdf_sites = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Unified convex hull mask
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

try:
    big_gdf = create_seamless_voronoi(df_final, 'big_cluster_id')
    nano_gdf = create_seamless_voronoi(df_final, 'nano_cluster_id')
    big_gdf.to_file(os.path.join(result_folder, "Big_Cluster_Border.tab"), driver="MapInfo File")
    nano_gdf.to_file(os.path.join(result_folder, "Nano_Cluster_Border.tab"), driver="MapInfo File")
except Exception as e:
    print(f"Voronoi failed or MapInfo export failed: {e}")

# FOLIUM MAP
legend_html = """
{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 110px; border:2px solid grey; z-index:9999; 
    font-size:14px; background-color: white; opacity: 0.85; padding: 10px; border-radius: 10px; font-family: 'Arial';">
    <b>Site Type Legend</b><br>
    <i style="color: black;">&#9650;</i> P1 / HVC Site<br>
    <i style="color: black;">&#9679;</i> Normal Site<br>
    <hr style="margin: 5px 0;">
    <small>Line: Converging Path</small>
</div>
{% endmacro %}
"""
legend = MacroElement()
legend._template = Template(legend_html)

map_center = [df_final['LATITUDE'].mean(), df_final['LONGITUDE'].mean()]
m = folium.Map(location=map_center, zoom_start=11, tiles='CartoDB Positron')
m.add_child(legend)

import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
fallback_color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(df_final['big_cluster_id'].unique())}

# Points
fg_points = folium.FeatureGroup(name="Sites")
for _, row in df_final.iterrows():
    color = fallback_color_map[row['big_cluster_id']]
        
    if row['is_hvc']:
        RegularPolygonMarker(location=[row['LATITUDE'], row['LONGITUDE']], number_of_sides=3, radius=7, color=color, fill=True, fill_opacity=0.6, popup=row['nano_cluster_id']).add_to(fg_points)
    else:
        folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']], radius=4, color=color, fill=True, fill_opacity=0.6, popup=row['nano_cluster_id']).add_to(fg_points)
fg_points.add_to(m)

# Snake Path Big
fg_path_big = folium.FeatureGroup(name="Big Cluster Path", show=False)
for (region, zone), group in df_final.groupby(['REGION', 'ZONE']):
    big_centroids = group.groupby('big_cluster_id').agg({'LATITUDE':'mean', 'LONGITUDE':'mean', 'big_seq_temp':'first'}).sort_values('big_seq_temp')
    pts = big_centroids[['LATITUDE', 'LONGITUDE']].values.tolist()
    
    color = "black"
    if region.upper() == 'JABO' and zone in zone_colors:
        color = zone_colors[zone]
        
    if region.upper() == 'JABO':
        # Draw black outline first
        folium.PolyLine(pts, color="black", weight=9, opacity=0.9).add_to(fg_path_big)
        folium.PolyLine(pts, color=color, weight=5, opacity=1.0).add_to(fg_path_big)
    else:
        folium.PolyLine(pts, color=color, weight=5, opacity=0.8).add_to(fg_path_big)
        
    for _, p in big_centroids.iterrows():
        folium.Marker(location=[p['LATITUDE'], p['LONGITUDE']], icon=folium.DivIcon(html=f'''
            <div style="font-size: 8pt; color: white; background: {color}; border-radius: 4px; width: 22px; height: 16px; 
            text-align: center; line-height: 16px; border: 1px solid white; font-weight: bold;">{int(p["big_seq_temp"])}</div>''')).add_to(fg_path_big)
fg_path_big.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save(os.path.join(result_folder, "5G_Final_Revised_Map.html"))

print("Script execution completed successfully.")
