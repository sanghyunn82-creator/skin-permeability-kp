import json, math, warnings, re
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
D='/workspace/icbmehs_c3_skin/data/'
LOG3600 = math.log10(3600.0)

def canon(s):
    if not s or not isinstance(s,str): return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

rows=[]
# --- HuskinDB (logKp already cm/s) ---
hk=json.load(open(D+'huskindb_full.json'))
for r in hk:
    rows.append(dict(src='HuskinDB', name=(r[0] or '').strip(), smiles=r[1], cas=None, logkp_cms=r[3]))
# --- SkinPiX (logKp cm/s converted) ---
sp=pd.read_excel(D+'skinpix_20230620_cleanedDB.xlsx')
for _,r in sp.iterrows():
    rows.append(dict(src='SkinPiX', name=str(r['compound name']).strip(), smiles=r['SMILES'],
                     cas=str(r['CAS number']).strip(), logkp_cms=r['logKp (cm/s) (converted)']))
# --- Cheruvu 2022 (logkp in cm/h -> cm/s) ---
ch=pd.read_excel(D+'cheruvu2022_Table1.xlsx')
for _,r in ch.iterrows():
    v=r['logkpl']
    rows.append(dict(src='Cheruvu2022', name=str(r['Compound']).strip(), smiles=None,
                     cas=str(r['CAS No']).strip(), logkp_cms=(v-LOG3600) if pd.notna(v) else None))

df=pd.DataFrame(rows)
df['csmiles']=df['smiles'].map(canon)
df['key']=df['csmiles'].fillna(df['cas']).fillna(df['name'].str.lower())
df=df[df['logkp_cms'].notna()]

print('=== RECORDS WITH logKp (cm/s) ===')
print(df.groupby('src').agg(records=('logkp_cms','size'), unique_key=('key','nunique'),
                            kp_min=('logkp_cms','min'), kp_max=('logkp_cms','max')).round(2))
print('TOTAL records:', len(df), '| UNION unique entities (canonical SMILES > CAS > name):', df['key'].nunique())
print('SMILES resolved:', df['csmiles'].notna().sum(), '/', len(df))

# overlap between the two SMILES-bearing sets
hs=set(df[df.src=="HuskinDB"]['csmiles'].dropna()); ss=set(df[df.src=="SkinPiX"]['csmiles'].dropna())
print('HuskinDB unique SMILES:',len(hs),'| SkinPiX unique SMILES:',len(ss),'| shared:',len(hs&ss),'| SkinPiX-only:',len(ss-hs))

# --- analgesic subgroup, explicit auditable term list (ATC N02 / M01A / N01B) ---
TERMS = {
 'opioid (ATC N02A)': ['morphine','codeine','fentanyl','sufentanil','alfentanil','remifentanil','oxycodone',
    'oxymorphone','hydrocodone','hydromorphone','buprenorphine','methadone','tramadol','tapentadol',
    'meperidine','pethidine','etorphine','naltrexone','naloxone','papaverine'],
 'NSAID / non-opioid (ATC M01A, N02B)': ['ibuprofen','ketoprofen','naproxen','diclofenac','indomethacin','indometacin',
    'flurbiprofen','ketorolac','piroxicam','meloxicam','celecoxib','etodolac','nimesulide','flufenamic',
    'mefenamic','salicyl','aspirin','acetylsalicylic','paracetamol','acetaminophen','aminopyrine',
    'antipyrine','pranoprofen','loxoprofen','aceclofenac'],
 'local anaesthetic (ATC N01B)': ['lidocaine','lignocaine','prilocaine','bupivacaine','ropivacaine','tetracaine',
    'benzocaine','aminobenzoate','procaine','mepivacaine','articaine'],
 'analgesic adjuvant': ['clonidine','capsaicin','gabapentin','pregabalin','amitriptyline','nortriptyline'],
}
def klass(n):
    l=n.lower()
    for k,ts in TERMS.items():
        if any(t in l for t in ts): return k
    return None
df['klass']=df['name'].map(klass)
an=df[df['klass'].notna()]
print()
print('=== ANALGESIC / PAIN-RELATED SUBGROUP (provisional, name-matched) ===')
print('records:',len(an),'| unique entities:',an['key'].nunique())
print(an.groupby('klass').agg(records=('logkp_cms','size'),unique=('key','nunique')))
print()
for k in TERMS:
    sub=an[an.klass==k]
    if len(sub)==0: continue
    print(f'-- {k}:')
    for n,g in sub.groupby(sub['name'].str.lower()):
        print(f'   {g["name"].iloc[0]:42s} n={len(g):2d}  logKp {g["logkp_cms"].min():.2f}..{g["logkp_cms"].max():.2f}  [{",".join(sorted(set(g["src"])))}]')
df.to_csv('/workspace/icbmehs_c3_skin/data/merged_inventory.csv',index=False)
print('\nwrote merged_inventory.csv')
