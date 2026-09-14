"""Harvest verified citations from PubMed E-utilities. Nothing written from memory."""
import json, time, urllib.parse, urllib.request, sys
E='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(u):
    for _ in range(3):
        try: return json.loads(urllib.request.urlopen(u,timeout=30).read())
        except Exception as e: time.sleep(2)
    return None
def search(term,n=4):
    u=E+'esearch.fcgi?db=pubmed&retmode=json&retmax=%d&sort=relevance&term=%s'%(n,urllib.parse.quote(term))
    r=get(u); return r['esearchresult']['idlist'] if r else []
def summ(ids):
    if not ids: return []
    r=get(E+'esummary.fcgi?db=pubmed&retmode=json&id=%s'%','.join(ids))
    if not r: return []
    out=[]
    for i in ids:
        d=r['result'].get(i)
        if not d or 'error' in d: continue
        doi=next((x['value'] for x in d.get('articleids',[]) if x['idtype']=='doi'),'')
        out.append(dict(pmid=i,title=d.get('title','').rstrip('.'),
            authors=[a['name'] for a in d.get('authors',[]) if a.get('authtype')=='Author'],
            journal=d.get('source',''),year=(d.get('pubdate','') or '')[:4],
            vol=d.get('volume',''),issue=d.get('issue',''),pages=d.get('pages',''),doi=doi))
    return out
QUERIES=[
 ('pain-burden','global burden chronic pain prevalence disability adults'),
 ('tdd-overview','transdermal drug delivery systems review advantages skin'),
 ('fentanyl-patch','transdermal fentanyl patch chronic pain management efficacy'),
 ('buprenorphine-patch','transdermal buprenorphine patch chronic pain randomised'),
 ('lidocaine-patch','lidocaine 5% medicated plaster localised neuropathic pain'),
 ('opioid-safety','transdermal fentanyl patch safety adverse events respiratory depression'),
 ('stratum-corneum','stratum corneum barrier function lipid organisation percutaneous absorption'),
 ('ivpt','in vitro permeation testing human skin Franz cell standardisation variability'),
 ('ivpt-interlab','interlaboratory variability in vitro skin permeation human skin'),
 ('oecd428','OECD 428 skin absorption in vitro method guidance dermal risk assessment'),
 ('flynn','Flynn physicochemical determinants skin permeability database quantitative'),
 ('qspr-skin','quantitative structure permeability relationship skin QSPR model review'),
 ('ml-skin','machine learning prediction skin permeability random forest deep learning'),
 ('mitragotri','mathematical models skin permeability overview mechanisms'),
 ('ad-qsar','applicability domain QSAR model validation OECD principles'),
 ('rdkit-desc','RDKit open source cheminformatics descriptor calculation'),
 ('sklearn','scikit-learn machine learning in Python'),
 ('xgboost','XGBoost scalable tree boosting system'),
 ('crippen','Crippen atom contribution molar refractivity logP calculation'),
 ('rf','random forests Breiman machine learning ensemble'),
 ('permeant-ionisation','effect of pH ionisation on skin permeation of drugs'),
 ('temperature','effect of temperature on percutaneous absorption human skin'),
 ('skin-site','regional variation skin permeability body site human'),
 ('nsaid-topical','topical NSAID ibuprofen diclofenac musculoskeletal pain efficacy'),
 ('prodrug','prodrug approach improve skin permeation glucoside ester'),
 ('data-quality','data quality curation chemical databases QSAR modelling impact'),
 ('molecular-size','molecular size lipophilicity determinants percutaneous penetration'),
]
allrefs={}
for k,q in QUERIES:
    hits=summ(search(q,4)); allrefs[k]=hits
    print('== %s (%s)'%(k,q))
    for h in hits:
        print('   PMID %s | %s | %s %s;%s(%s):%s | doi:%s'%(h['pmid'],h['title'][:95],h['journal'],h['year'],h['vol'],h['issue'],h['pages'],h['doi']))
    time.sleep(0.4)
json.dump(allrefs,open('/workspace/icbmehs_c3_skin/data/ref_candidates.json','w'),indent=1)
print('\nwrote ref_candidates.json')
