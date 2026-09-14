import json, time, urllib.parse, urllib.request
E='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(u):
    for _ in range(3):
        try: return json.loads(urllib.request.urlopen(u,timeout=30).read())
        except Exception: time.sleep(2)
    return None
def search(t,n=5):
    r=get(E+'esearch.fcgi?db=pubmed&retmode=json&retmax=%d&sort=relevance&term=%s'%(n,urllib.parse.quote(t)))
    return r['esearchresult']['idlist'] if r else []
def summ(ids):
    if not ids: return []
    r=get(E+'esummary.fcgi?db=pubmed&retmode=json&id=%s'%','.join(ids)); out=[]
    if not r: return out
    for i in ids:
        d=r['result'].get(i)
        if not d or 'error' in d: continue
        doi=next((x['value'] for x in d.get('articleids',[]) if x['idtype']=='doi'),'')
        out.append(dict(pmid=i,title=d.get('title','').rstrip('.'),
          authors=[a['name'] for a in d.get('authors',[]) if a.get('authtype')=='Author'],
          journal=d.get('source',''),year=(d.get('pubdate','') or '')[:4],vol=d.get('volume',''),
          issue=d.get('issue',''),pages=d.get('pages',''),doi=doi))
    return out
Q=[('crippen','Wildman Crippen prediction physicochemical parameters atomic contributions'),
   ('edetox','EDETOX database percutaneous penetration occupational exposure'),
   ('ml-skin2','artificial neural network prediction human skin permeability coefficient QSPR'),
   ('ml-skin3','Gaussian process machine learning skin permeability prediction comparison'),
   ('site-var','regional anatomical site differences percutaneous penetration hydrocortisone human'),
   ('pain-prev','prevalence chronic pain adults national survey burden'),
   ('lowbackpain','global burden low back pain 1990 2020 projections'),
   ('fent-pk','transdermal fentanyl pharmacokinetics patch systemic delivery rate'),
   ('flynn2','Flynn physicochemical determinants percutaneous absorption principles route'),
   ('skinperm-rev','skin permeability prediction models comparison evaluation mathematical'),
   ('opioid-transdermal','transdermal opioid delivery physicochemical requirements patch design'),
   ('sc-lipid','stratum corneum intercellular lipid lamellae permeability pathway'),
   ('vecchia-bunge','skin permeability coefficient database evaluation aqueous solutions Bunge'),
   ('doseresp','infinite dose finite dose skin permeation experimental design effect'),
   ('curation','chemical structure curation standardisation QSAR best practice'),
   ('crossval','cross-validation model validation external test set QSAR reliability'),
  ]
out={}
for k,q in Q:
    h=summ(search(q,5)); out[k]=h
    print('== %s'%k)
    for x in h:
        print('   PMID %s | %s | %s %s;%s(%s):%s | doi:%s'%(x['pmid'],x['title'][:92],x['journal'],x['year'],x['vol'],x['issue'],x['pages'],x['doi']))
    time.sleep(0.4)
json.dump(out,open('/workspace/icbmehs_c3_skin/data/ref_candidates2.json','w'),indent=1)
print('done')
