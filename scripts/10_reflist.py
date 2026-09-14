"""Emit the Vancouver reference list in citation order from verified PubMed records."""
import json, time, urllib.parse, urllib.request
E='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(u):
    for _ in range(4):
        try: return json.loads(urllib.request.urlopen(u,timeout=30).read())
        except Exception: time.sleep(2)
    return None
ORDER=[33990113,37273833,28497473,27103611,1517637,22141389,26929664,34813879,37109736,
       34403750,37666282,12467638,1608900,12020604,28645577,30658035,38258521,21988422,
       22420662,33262341,38383523,36151144,37996523,14629737,29934891,11858635,41772705,
       25506400,22634139,39437978,29468312,14757507,8378261,22580335,38351144,31539567,
       27885862,26896663]
ids=[str(x) for x in ORDER]
recs={}
for i in range(0,len(ids),20):
    r=get(E+'esummary.fcgi?db=pubmed&retmode=json&id=%s'%','.join(ids[i:i+20]))
    if r: recs.update(r['result']); time.sleep(0.4)
def vanc(d):
    au=[a['name'] for a in d.get('authors',[]) if a.get('authtype')=='Author']
    if not au:
        col=[a['name'] for a in d.get('authors',[])]
        astr=col[0] if col else 'Anonymous'
    elif len(au)>6: astr=', '.join(au[:6])+', et al'
    else: astr=', '.join(au)
    doi=next((x['value'] for x in d.get('articleids',[]) if x['idtype']=='doi'),'')
    ti=d.get('title','').rstrip('.').replace('[','').replace(']','')
    yr=(d.get('pubdate','') or '')[:4]; vol=d.get('volume',''); iss=d.get('issue',''); pg=d.get('pages','')
    loc=vol+(('('+iss+')') if iss else '')+((':'+pg) if pg else '')
    s='%s. %s. %s. %s;%s.'%(astr,ti,d.get('source',''),yr,loc)
    if doi: s+=' doi: %s.'%doi
    return s
print('=== VANCOUVER REFERENCE LIST (PubMed-verified) ===')
miss=[]
n=0
for i in ids:
    d=recs.get(i)
    if not d or 'error' in d: miss.append(i); continue
    n+=1; print('%d. %s'%(n,vanc(d)))
print('\nresolved %d / %d ; missing: %s'%(n,len(ids),miss))
