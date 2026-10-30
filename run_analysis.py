from pathlib import Path
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio,tarfile,io,json
from rasterio.io import MemoryFile
from scipy.ndimage import minimum_filter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
P=Path(__file__).resolve().parent; SRC=P.parent
raw=pd.read_csv(SRC/'nyc311_infra_2025.csv',parse_dates=['created_date','closed_date'],low_memory=False)
df=raw.drop_duplicates('unique_key');df=df[df.borough.ne('Unspecified')].copy()
CRITICAL=['Cave-in','Sidewalk Collapsed','Manhole Cover Missing','Catch Basin Grating Missing','Plate Condition - Open','Street Flooding','Highway Flooding','Manhole Overflow','Sewer Backup','Cellar Door Open/Unprotected','Guard Rail - Street','Crash Cushion Defect','Unsafe Worksite','Sidewalk Grating - Missing','Metal Protruding - Sign Stump']
HIGH=['Pothole','Catch Basin Clogged/Flooding','Broken Sidewalk','Failed Street Repair','Plate Condition - Shifted','Catch Basin Sunken/Damaged/Raised','Pedestrian Ramp Defective','Culvert Blocked/Needs Cleaning','Defective Hardware','Rough, Pitted or Cracked Roads','Sidewalk Grating - Defective','Sidewalk Staircase','Defective/Missing Curb Piece','Affecting Sewer or Foundation','Depression Maintenance','Hummock','Cellar Door Defective']
FLOOD=['Sewer Backup','Catch Basin Clogged/Flooding','Street Flooding','Manhole Overflow','Highway Flooding','Culvert Blocked/Needs Cleaning']
df['issue']=df.descriptor.fillna('Unknown').str.replace(r'\s*\((Use Comments|Emergency)\)','',regex=True).str.replace(r'\s*\([A-Z0-9]+\)$','',regex=True).str.strip()
df['tier']=np.select([df.issue.isin(CRITICAL),df.issue.isin(HIGH)],['Critical','High'],'Routine');df['flood']=df.issue.isin(FLOOD)
df['date']=df.created_date.dt.normalize();df['month']=df.created_date.dt.month;df['dow']=df.created_date.dt.dayofweek
w=pd.read_csv(SRC/'nyc_centralpark_2025(1).csv',parse_dates=['DATE']).rename(columns={'DATE':'date','PRCP':'rain_mm'})
df['res_days']=(df.closed_date-df.created_date).dt.total_seconds()/86400;df.loc[df.res_days<0,'res_days']=np.nan
# NYC rain chart
byday=df.groupby('date').agg(flood=('flood','sum')).reset_index().merge(w[['date','rain_mm']],on='date',how='left')
byday['rain_bin']=pd.cut(byday.rain_mm,[-.1,0,5,20,10000],labels=['0','0–5','5–20','>20'])
rain_stats=byday.groupby('rain_bin',observed=True).flood.agg(['mean','size'])
fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(rain_stats.index.astype(str),rain_stats['mean'],color='#2877ad');ax.set(xlabel='Daily precipitation (mm), Central Park',ylabel='Mean daily flood complaints',title='NYC: rainfall and flood-related complaints');fig.tight_layout();fig.savefig(P/'nyc_rain_complaints.png',dpi=160);plt.close(fig)
# NYC recurrence evaluation: avoid censoring December and ensure positive target known by end of dataset
q=df.dropna(subset=['latitude','longitude']).copy();q['cell']=q.latitude.round(3).astype(str)+'_'+q.longitude.round(3).astype(str);q=q.sort_values('created_date');q['next']=q.groupby('cell').created_date.shift(-1);q['recur30']=((q['next']-q.created_date).dt.total_seconds()/86400<=30).astype(int)
q=q.merge(w[['date','rain_mm']],on='date',how='left');q['rain_mm']=q.rain_mm.fillna(0)
q['rain_3d']=q.date.map(w.set_index('date').rain_mm.rolling(3,min_periods=1).sum()).fillna(0)
q['_one']=1;q=q.set_index('created_date');q['prior_90d']=q.groupby('cell')['_one'].transform(lambda s:s.rolling('90D').count()-1);q=q.reset_index();q['prior_any']=q.groupby('cell').cumcount();q=q[q.created_date<'2025-12-01'].copy()
cat=['complaint_type','issue','tier','borough','location_type'];num=['month','dow','rain_mm','rain_3d','prior_90d','prior_any'];q[cat]=q[cat].fillna('Unknown');q[num]=q[num].fillna(0)
train=q[q.created_date<'2025-09-01'];test=q[q.created_date>='2025-09-01']
pre=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore',min_frequency=20,sparse_output=False),cat),('n','passthrough',num)])
model=make_pipeline(pre,HistGradientBoostingClassifier(max_iter=120,learning_rate=.08,max_leaf_nodes=21,random_state=42))
model.fit(train[cat+num],train.recur30);prob=model.predict_proba(test[cat+num])[:,1];threshold=np.quantile(prob,.9)
metrics={'raw_records':len(raw),'clean_records':len(df),'train_records':len(train),'test_records':len(test),'test_positive_rate':float(test.recur30.mean()),'roc_auc':float(roc_auc_score(test.recur30,prob)),'pr_auc':float(average_precision_score(test.recur30,prob)),'f1_at_05':float(f1_score(test.recur30,prob>=.5)),'precision_top_decile':float(test.recur30[prob>=threshold].mean()),'nyc_rain_bins':{str(k):{'avg_flood_complaints':float(v['mean']),'days':int(v['size'])} for k,v in rain_stats.iterrows()}}
# Istanbul precipitation from already-prepared export, traceable to supplied Open-Meteo workbook
ist=pd.read_csv(SRC/'istanbul_daily_rain_oct2026.csv');ist.to_csv(P/'istanbul_rain_oct2026.csv',index=False)
fig,ax=plt.subplots(figsize=(8,3.5));ax.bar(pd.to_datetime(ist.date),ist.rain_mm,color='#2877ad');ax.set(xlabel='Date (October 2026)',ylabel='Daily precipitation (mm)',title='Istanbul Open-Meteo, single coordinate (41.0193N, 28.8873E)');fig.autofmt_xdate();fig.tight_layout();fig.savefig(P/'istanbul_rain.png',dpi=160);plt.close(fig)
metrics['istanbul_oct14_mm']=float(ist.loc[ist.date=='2026-10-14','rain_mm'].iloc[0]);metrics['istanbul_max_daily_mm']=float(ist.rain_mm.max())
# COP30 DEM exploratory screening (NOT flood probability)
with tarfile.open(SRC/'rasters_COP30.tar') as tar: buf=tar.extractfile('output_hh.tif').read()
with MemoryFile(buf) as mem,mem.open() as ds:
 z=ds.read(1).astype(float);bounds=ds.bounds;trans=ds.transform;crs=str(ds.crs);nodata=ds.nodata
 if nodata is not None:z[z==nodata]=np.nan
valid=np.isfinite(z);a=np.where(valid,z,np.nanmedian(z));lat=(bounds.top+bounds.bottom)/2
if crs=='EPSG:4326':dx=abs(trans.a)*111320*np.cos(np.radians(lat));dy=abs(trans.e)*111320
else:dx=abs(trans.a);dy=abs(trans.e)
gy,gx=np.gradient(a,dy,dx);slope=np.degrees(np.arctan(np.hypot(gx,gy)));rel=a-minimum_filter(a,size=15,mode='nearest');proxy=100*(.55*np.clip((15-rel)/15,0,1)+.45*np.clip((12-slope)/12,0,1));proxy[~valid]=np.nan
ext=[bounds.left,bounds.right,bounds.bottom,bounds.top];fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for ax,data,title in zip(axs,[z,slope,proxy],['Elevation (m)','Slope (degrees)','Exploratory terrain index (0–100)']):
 im=ax.imshow(data,extent=ext,origin='upper',cmap='viridis');ax.set_title(title);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');fig.colorbar(im,ax=ax,shrink=.75)
fig.suptitle('Istanbul COP30 pilot — terrain index is NOT validated flood probability');fig.savefig(P/'istanbul_terrain.png',dpi=150);plt.close(fig)
metrics['dem_bounds_lonlat']=[bounds.left,bounds.bottom,bounds.right,bounds.top];metrics['dem_shape']=list(z.shape);metrics['terrain_index_median']=float(np.nanmedian(proxy))
(P/'results.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False));print(json.dumps(metrics,indent=2))
