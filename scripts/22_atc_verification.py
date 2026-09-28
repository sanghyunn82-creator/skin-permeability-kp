"""Reviewer 2, Minor 1.

"Name-matching against a term list is acknowledged as a limitation, but mapping
to actual ATC codes is straightforward and would remove the caveat rather than
footnote it."

The 40 analgesics were assigned by matching compound names against a term list
drawn from ATC classes N02A, N02B, M01A and N01B. Here each name is looked up
against real ATC codes through the RxClass API of the US National Library of
Medicine, and the resulting codes are compared with the published assignment.

Not every entry can be resolved this way, and that is itself worth reporting:
the dataset contains glucoside and mannoside conjugates of several
anti-inflammatory drugs, which are research compounds with no ATC code of their
own. For those the parent drug is looked up and the inheritance is recorded
explicitly rather than silently assumed.

Writes revision/minor1/. Nothing published is overwritten.
"""
import json, os, re, time, urllib.parse, urllib.request
import pandas as pd

ROOT = '/workspace/icbmehs_c3_skin/'
OUT = ROOT + 'revision/minor1/'
os.makedirs(OUT, exist_ok=True)
API = 'https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json?drugName=%s&relaSource=ATC'

TARGET = {'N02A': 'Opioid', 'N02B': 'NSAID / non-opioid',
          'M01A': 'NSAID / non-opioid', 'N01B': 'Local anaesthetic'}

an = pd.read_csv(ROOT + 'results/analgesic_predictions_classed.csv')
print('%d analgesics to verify' % len(an))

def lookup(name):
    try:
        url = API % urllib.parse.quote(name)
        with urllib.request.urlopen(url, timeout=25) as r:
            d = json.load(r)
        items = d.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
        return sorted({(i['rxclassMinConceptItem']['classId'],
                        i['rxclassMinConceptItem']['className']) for i in items})
    except Exception:
        return []

# Some entries are indexed under a common name rather than the systematic one.
# These are the same substance, not a substitution, and each is verified below.
SYNONYM = {
    'acetylsalicylic acid': 'Aspirin',            # same substance, ATC N02BA01
    'ethyl p-aminobenzoate': 'Benzocaine',        # benzocaine is ethyl p-aminobenzoate
}

def parent_of(name):
    """Strip conjugate suffixes to reach the parent drug."""
    n = re.sub(r'\s+(mannoside|glucoside|glucuronide|ester|sodium|hydrochloride|HCl)$',
               '', name.strip(), flags=re.I)
    return n if n.lower() != name.strip().lower() else None

rows = []
for _, r in an.iterrows():
    name = r['name']
    codes = lookup(name)
    via_parent = None
    if not codes and name.strip().lower() in SYNONYM:
        syn = SYNONYM[name.strip().lower()]
        codes = lookup(syn)
        via_parent = 'synonym: %s' % syn if codes else None
    if not codes:
        p = parent_of(name)
        if p:
            codes = lookup(p)
            via_parent = p if codes else None
    time.sleep(0.25)
    matched = sorted({c for c, _ in codes if c[:4] in TARGET})
    implied = sorted({TARGET[c[:4]] for c in matched})
    rows.append(dict(
        name=name, published_class=r['klass'],
        atc_codes_all='; '.join(c for c, _ in codes) or '',
        atc_codes_in_scope='; '.join(matched) or '',
        atc_implied_class='; '.join(implied) or '',
        resolved_via_parent=via_parent or '',
        agrees=(r['klass'] in implied) if implied else None))
    print('  %-28s %-20s -> %s' % (name[:28], r['klass'],
                                   '; '.join(matched) or ('(no ATC)' if not codes else '(out of scope)')))

df = pd.DataFrame(rows)
df.to_csv(OUT + 'atc_verification.csv', index=False)

n_any = int((df.atc_codes_all != '').sum())
n_scope = int((df.atc_codes_in_scope != '').sum())
n_parent = int((df.resolved_via_parent != '').sum())
agree = df[df.agrees.notna()]
n_agree = int(agree.agrees.sum())
disagree = agree[~agree.agrees.astype(bool)]

summary = dict(n=len(df), with_any_atc=n_any, with_in_scope_atc=n_scope,
               resolved_via_parent=n_parent, comparable=int(len(agree)),
               agreeing=n_agree, disagreeing=int(len(disagree)),
               disagreements=disagree[['name', 'published_class',
                                       'atc_implied_class']].to_dict('records'),
               no_atc=df[df.atc_codes_all == '']['name'].tolist())
json.dump(summary, open(OUT + 'atc_summary.json', 'w'), indent=1, ensure_ascii=False)

print('\n=== summary ===')
print('  any ATC code found        : %d / %d' % (n_any, len(df)))
print('  code inside N02A/N02B/M01A/N01B : %d' % n_scope)
print('  resolved only via parent drug   : %d' % n_parent)
print('  comparable with published class : %d, agreeing %d, disagreeing %d'
      % (len(agree), n_agree, len(disagree)))
if len(disagree):
    print('  disagreements:')
    for _, d in disagree.iterrows():
        print('    %-28s published %-20s ATC implies %s'
              % (d['name'][:28], d['published_class'], d['atc_implied_class']))
if summary['no_atc']:
    print('  no ATC code at all: %s' % ', '.join(summary['no_atc']))
print('\nwrote', OUT)
