"""Format the chosen PubMed records into Vancouver style. Nothing typed from memory."""
import json, time, urllib.parse, urllib.request, sys
E='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(u):
    for _ in range(3):
        try: return json.loads(urllib.request.urlopen(u,timeout=30).read())
        except Exception: time.sleep(2)
    return None
def find(term):
    r=get(E+'esearch.fcgi?db=pubmed&retmode=json&retmax=3&sort=relevance&term=%s'%urllib.parse.quote(term))
    return r['esearchresult']['idlist'] if r else []
NEED={'huskindb':'HuskinDB a database for skin permeation of xenobiotics',
      'waters2022':'Predicting skin permeability using huskinDB Waters Quah',
      'waters2023':'Fragment contribution models for predicting skin permeability using huskinDB',
      'skinpix':'An update of skin permeability data based on a systematic review of recent research',
      'crippen':'Prediction of physicochemical parameters by atomic contributions Wildman Crippen'}
for k,q in NEED.items():
    ids=find(q); print('%-12s -> %s'%(k,ids))
    if ids:
        r=get(E+'esummary.fcgi?db=pubmed&retmode=json&id=%s'%ids[0])
        d=r['result'][ids[0]]
        print('             %s | %s %s;%s(%s):%s'%(d['title'][:80],d['source'],d['pubdate'][:4],d['volume'],d.get('issue',''),d.get('pages','')))
