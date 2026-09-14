import pandas as pd
df=pd.read_csv('/workspace/icbmehs_c3_skin/data/merged_inventory.csv')
u=df[df.src.isin(['HuskinDB','SkinPiX'])].copy()          # both carry SMILES
u=u[u.csmiles.notna()]
print('=== SMILES-verified union: HuskinDB + SkinPiX ===')
print('records:',len(u),'| unique compounds (canonical SMILES):',u.csmiles.nunique())
print('logKp(cm/s) range: %.2f .. %.2f'%(u.logkp_cms.min(),u.logkp_cms.max()))
print('compounds with >=2 records (replicate spread available):',(u.groupby("csmiles").size()>=2).sum())
print()
# STRICT = drugs whose primary indication is pain (ATC N02A, N02B, M01A, N01B)
STRICT=['morphine','codeine','fentanyl','sufentanil','oxycodone','oxymorphone','hydrocodone','hydromorphone',
 'meperidine','etorphine','buprenorphine','tramadol','ibuprofen','ketoprofen','naproxen','diclofenac',
 'indomethacin','indometacin','flurbiprofen','ketorolac','etodolac','nimesulide','flufenamic','pranoprofen',
 'aminopyrine','acetylsalicylic','salicylic acid','methyl salicylate','lidocaine','aminobenzoate','benzocaine']
EXCLUDE=['papaverine','naltrexone','naloxone']   # opium alkaloid / antagonists: not analgesics
def strict(n):
    l=str(n).lower()
    if any(e in l for e in EXCLUDE): return False
    return any(t in l for t in STRICT)
u['analg']=u['name'].map(strict)
a=u[u.analg]
print('=== ANALGESIC SUBGROUP (strict: ATC N02A/N02B/M01A/N01B) ===')
print('records:',len(a),'| unique compounds:',a.csmiles.nunique())
print('logKp range: %.2f .. %.2f'%(a.logkp_cms.min(),a.logkp_cms.max()))
print()
for n,g in sorted(a.groupby(a['name']), key=lambda x:-len(x[1])):
    print('  %-40s n=%2d  logKp %6.2f..%6.2f  [%s]'%(n,len(g),g.logkp_cms.min(),g.logkp_cms.max(),','.join(sorted(set(g.src)))))
print()
print('parent-drug count excluding glycoside prodrug conjugates:',
      a[~a.name.str.lower().str.contains('glucoside|mannoside')].csmiles.nunique())
u.to_csv('/workspace/icbmehs_c3_skin/data/union_smiles_verified.csv',index=False)
print('wrote union_smiles_verified.csv (n=%d)'%len(u))
