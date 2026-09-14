"""Three-way self-plagiarism screen across the C1/C2/C3 submission manuscripts.

The three papers were produced by the same pipeline for the same client, and
go out to different journals at about the same time, so shared boilerplate is
a real self-plagiarism exposure rather than a theoretical one.

Method (as specified): difflib.SequenceMatcher ratio, paragraph-by-paragraph
across all three pairs, reporting every pair at ratio >= 0.30, plus a
sentence-level pass so that a short reused frame inside an otherwise different
paragraph is not averaged away.

Read-only. Writes a JSON of hits for the report; touches no manuscript.
"""
import json, os, re, unicodedata
from difflib import SequenceMatcher
from itertools import combinations
from docx import Document

DOCS = {
    'C1': '/workspace/icbmehs_c1_elderly/submission/01_Manuscript.docx',
    'C2': '/workspace/icbmehs_c2_organ/manuscript/antioxidants_draft/Antioxidants_Manuscript.docx',
    'C3': '/workspace/icbmehs_c3_skin/manuscript/medicina_draft/Medicina_Manuscript.docx',
}
OUT = '/tmp/claude-0/-workspace-icbmehs-c3-skin/62441292-5d2b-4e53-9573-ea6ed7682810/scratchpad/sim3.json'

PARA_MIN = 40        # ignore headings and stubs
SENT_MIN = 45
PARA_T = 0.30
SENT_T = 0.55        # sentences are short; a lower bar would be all noise

# Section attribution: the heading a paragraph sits under, normalised across
# the three papers' different heading conventions.
SECTION_PATTERNS = [
    (r'ethic|institutional review|informed consent',      'Ethics / IRB'),
    (r'data availability',                                'Data Availability'),
    (r'author contribution',                              'Author Contributions'),
    (r'generative artificial intelligence|genai',         'GenAI disclosure'),
    (r'acknowledg',                                       'Acknowledgments'),
    (r'conflict',                                         'Conflicts of Interest'),
    (r'funding',                                          'Funding'),
    (r'statistic',                                        'Statistics'),
    (r'limitation',                                       'Limitations'),
    (r'data source|data collection|dataset|study populat|participants|cohort',
                                                          'Data sources'),
    (r'software|implementation',                          'Software'),
    (r'introduction|background',                          'Introduction'),
    (r'material|method',                                  'Methods (other)'),
    (r'result',                                           'Results'),
    (r'discussion',                                       'Discussion'),
    (r'conclusion',                                       'Conclusions'),
    (r'abstract',                                         'Abstract'),
    (r'reference',                                        'References'),
]

def section_of(heading):
    h = (heading or '').lower()
    for pat, name in SECTION_PATTERNS:
        if re.search(pat, h):
            return name
    return heading or '(front matter)'

def norm(s):
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('–', '-').replace('—', '-').replace('−', '-')
    s = s.replace('‘', "'").replace('’', "'")
    s = s.replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', s).strip()

def is_heading(p):
    st = (p.style.name or '').lower()
    return ('head' in st or 'title' in st
            or bool(re.match(r'^\d+(\.\d+)*\.?\s+\S', p.text.strip())))

def load(path):
    """Paragraphs with the section they belong to. Back-matter paragraphs in the
    MDPI layout are one paragraph each labelled 'Funding: ...' etc., so their
    own leading label is used as the heading."""
    doc = Document(path)
    out, cur = [], None
    for p in doc.paragraphs:
        t = norm(p.text)
        if not t:
            continue
        if is_heading(p) and len(t) < 120:
            cur = t
            continue
        m = re.match(r'^([A-Z][A-Za-z /\-]{3,45}):\s', t)
        head = m.group(1) if m else cur
        body = t[m.end():] if m else t
        if len(body) >= PARA_MIN:
            out.append({'section': section_of(head), 'heading': head or '', 'text': body})
    return out

def sentences(t):
    parts = re.split(r'(?<=[.;])\s+(?=[A-Z(])', t)
    return [s.strip() for s in parts if len(s.strip()) >= SENT_MIN]

corpora = {k: load(v) for k, v in DOCS.items()}
for k, v in corpora.items():
    print('%s: %d comparable paragraphs' % (k, len(v)))

hits = []
for a, b in combinations(['C1', 'C2', 'C3'], 2):
    for i, pa in enumerate(corpora[a]):
        for j, pb in enumerate(corpora[b]):
            r = SequenceMatcher(None, pa['text'], pb['text'], autojunk=False).ratio()
            if r >= PARA_T:
                sm = SequenceMatcher(None, pa['text'], pb['text'], autojunk=False)
                blocks = sorted((x for x in sm.get_matching_blocks() if x.size > 25),
                                key=lambda x: -x.size)[:3]
                hits.append({
                    'pair': '%s-%s' % (a, b), 'level': 'paragraph', 'ratio': round(r, 3),
                    'section_a': pa['section'], 'section_b': pb['section'],
                    'a': pa['text'], 'b': pb['text'],
                    'shared': [pa['text'][bl.a:bl.a + bl.size] for bl in blocks],
                })
    # sentence level
    sa = [(p['section'], s) for p in corpora[a] for s in sentences(p['text'])]
    sb = [(p['section'], s) for p in corpora[b] for s in sentences(p['text'])]
    for seca, x in sa:
        for secb, y in sb:
            r = SequenceMatcher(None, x, y, autojunk=False).ratio()
            if r >= SENT_T:
                hits.append({
                    'pair': '%s-%s' % (a, b), 'level': 'sentence', 'ratio': round(r, 3),
                    'section_a': seca, 'section_b': secb, 'a': x, 'b': y, 'shared': [],
                })

hits.sort(key=lambda h: -h['ratio'])
json.dump({'docs': DOCS, 'hits': hits}, open(OUT, 'w'), ensure_ascii=False, indent=1)

print('\n%d hits (paragraph >= %.2f, sentence >= %.2f)' % (len(hits), PARA_T, SENT_T))
for pair in ['C1-C2', 'C1-C3', 'C2-C3']:
    ph = [h for h in hits if h['pair'] == pair]
    top = max((h['ratio'] for h in ph), default=0)
    print('  %s: %3d hits, max ratio %.3f' % (pair, len(ph), top))
print('\nwrote', OUT)
