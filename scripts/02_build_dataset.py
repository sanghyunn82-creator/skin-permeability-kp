"""Build modelling dataset: HuskinDB + SkinPiX, structure-standardised, RDKit descriptors."""
import json, warnings
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
D='/workspace/icbmehs_c3_skin/data/'

HK_COLS=['name','smiles','smiles2','logkp','skin_source','site','layer','preparation',
         'storage_temp','storage_days','flag','donor_temp','donor_ph','donor_vehicle',
         'acceptor_temp','acceptor_ph','acceptor_medium','cell_type','notes1','notes2','ref','doi']
hk=pd.DataFrame(json.load(open(D+'huskindb_full.json')), columns=HK_COLS); hk['src']='HuskinDB'
hk=hk[['src','name','smiles','logkp','site','layer','preparation','donor_temp','donor_ph',
       'acceptor_temp','acceptor_ph','cell_type','ref','doi']]
sp=pd.read_excel(D+'skinpix_20230620_cleanedDB.xlsx')
sp=pd.DataFrame({'src':'SkinPiX','name':sp['compound name'].astype(str).str.strip(),
    'smiles':sp['SMILES'],'logkp':sp['logKp (cm/s) (converted)'],
    'site':sp['skin source site'],'layer':sp['used layer'],'preparation':sp['skin preparation'],
    'donor_temp':sp['donor/skin surface temperature (°C)'],'donor_ph':sp['donor pH'],
    'acceptor_temp':sp['acceptor temperature (°C)'],'acceptor_ph':sp['acceptor pH'],
    'cell_type':sp['cell type'],'ref':sp['author'].astype(str)+', '+sp['date'].astype(str),'doi':sp['doi']})
df=pd.concat([hk,sp],ignore_index=True)
df['logkp']=pd.to_numeric(df['logkp'],errors='coerce')
for c in ['donor_temp','donor_ph','acceptor_temp','acceptor_ph']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
n0=len(df); df=df[df['logkp'].notna()].copy()
print('STEP0 pooled records: %d -> with logKp: %d'%(n0,len(df)))

# ---- structure standardisation: desalt, neutralise, strip isotopes ----
lfc=rdMolStandardize.LargestFragmentChooser(); unch=rdMolStandardize.Uncharger()
def std(smi):
    if not isinstance(smi,str): return None
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        m=rdMolStandardize.Cleanup(m); m=lfc.choose(m); m=unch.uncharge(m)
        for a in m.GetAtoms(): a.SetIsotope(0)
        return Chem.MolToSmiles(m)
    except Exception: return None
df['csmiles']=df['smiles'].map(std)
df=df[df['csmiles'].notna()].copy()
print('STEP1 valid+standardised structures: %d records / %d compounds'%(len(df),df.csmiles.nunique()))
# ---- exclude carbon-free (inorganic) solutes: Crippen/TPSA not parameterised ----
has_c=df.csmiles.map(lambda s: any(a.GetSymbol()=='C' for a in Chem.MolFromSmiles(s).GetAtoms()))
drop=sorted(set(df.loc[~has_c,'name']))
df=df[has_c].copy()
print('STEP2 excluded carbon-free solutes (%d compounds): %s'%(len(drop),'; '.join(drop)))
print('        remaining: %d records / %d compounds'%(len(df),df.csmiles.nunique()))


# ---- unify same-substance records that differ only by stereo annotation ----
df["_nm"]=df["name"].str.lower().str.strip()
best=(df.groupby(["_nm","csmiles"]).size().rename("n").reset_index()
        .sort_values(["_nm","n"],ascending=[True,False]).drop_duplicates("_nm"))
nmap=dict(zip(best["_nm"],best["csmiles"]))
multi=[k for k,g in df.groupby("_nm")["csmiles"] if g.nunique()>1]
df["csmiles"]=df["_nm"].map(nmap)
print("STEP2b unified %d substance(s) recorded under >1 structure: %s"%(len(multi),"; ".join(multi)))
print("        remaining: %d records / %d compounds"%(len(df),df.csmiles.nunique()))
df=df.drop(columns=["_nm"])

DESC={'MW':Descriptors.MolWt,'logP':Crippen.MolLogP,'MR':Crippen.MolMR,'TPSA':Descriptors.TPSA,
 'HBD':rdMolDescriptors.CalcNumHBD,'HBA':rdMolDescriptors.CalcNumHBA,
 'RotB':rdMolDescriptors.CalcNumRotatableBonds,'AromRings':rdMolDescriptors.CalcNumAromaticRings,
 'Rings':rdMolDescriptors.CalcNumRings,'HeavyAtoms':Descriptors.HeavyAtomCount,
 'FracCSP3':rdMolDescriptors.CalcFractionCSP3,'LabuteASA':rdMolDescriptors.CalcLabuteASA,
 'BalabanJ':Descriptors.BalabanJ,'BertzCT':Descriptors.BertzCT,
 'HeteroAtoms':rdMolDescriptors.CalcNumHeteroatoms}
rows=[]
for s in df.csmiles.unique():
    m=Chem.MolFromSmiles(s); d={'csmiles':s}
    for k,f in DESC.items():
        try: d[k]=f(m)
        except Exception: d[k]=np.nan
    rows.append(d)
desc=pd.DataFrame(rows)
print('STEP3 descriptors: %d structures x %d descriptors'%(len(desc),len(DESC)))

# ---- analgesic flag (ATC N02A / N02B / M01A / N01B), name-matched, audited list ----
STRICT=['morphine','codeine','fentanyl','sufentanil','oxycodone','oxymorphone','hydrocodone',
 'hydromorphone','meperidine','pethidine','etorphine','buprenorphine','tramadol','ibuprofen',
 'ketoprofen','naproxen','diclofenac','indomethacin','indometacin','flurbiprofen','ketorolac',
 'etodolac','nimesulide','flufenamic','pranoprofen','aminopyrine','acetylsalicylic',
 'salicylic acid','methyl salicylate','lidocaine','aminobenzoate','benzocaine']
EXCLUDE=['papaverine','naltrexone','naloxone']
def is_analg(n):
    l=str(n).lower()
    return (not any(e in l for e in EXCLUDE)) and any(t in l for t in STRICT)
df['analgesic']=df['name'].map(is_analg)

df=df.merge(desc,on='csmiles',how='left')
df.to_csv(D+'records_with_descriptors.csv',index=False)

agg=df.groupby('csmiles').agg(name=('name','first'), n_rec=('logkp','size'),
    logkp=('logkp','median'), logkp_min=('logkp','min'), logkp_max=('logkp','max'),
    n_src=('src','nunique'), n_ref=('ref','nunique'), analgesic=('analgesic','max')).reset_index()
agg=agg.merge(desc,on='csmiles',how='left')
agg.to_csv(D+'compounds_modelling.csv',index=False)
rep=agg[agg.n_rec>=2]
print()
print('=== FINAL MODELLING TABLE: %d compounds, %d records ==='%(len(agg),len(df)))
print('logKp (cm/s): median %.2f, IQR %.2f to %.2f, range %.2f to %.2f'%(
    agg.logkp.median(),agg.logkp.quantile(.25),agg.logkp.quantile(.75),agg.logkp.min(),agg.logkp.max()))
print('MW %.1f-%.1f | logP %.2f-%.2f | TPSA %.1f-%.1f'%(agg.MW.min(),agg.MW.max(),agg.logP.min(),agg.logP.max(),agg.TPSA.min(),agg.TPSA.max()))
print('replicated compounds (>=2 records): %d; median within-compound logKp range %.2f (IQR %.2f-%.2f)'%(
    len(rep), (rep.logkp_max-rep.logkp_min).median(),
    (rep.logkp_max-rep.logkp_min).quantile(.25),(rep.logkp_max-rep.logkp_min).quantile(.75)))
print('ANALGESIC subgroup: %d compounds / %d records'%(agg.analgesic.sum(),df.analgesic.sum()))
print('  ->', '; '.join(sorted(agg.loc[agg.analgesic,'name'])))
