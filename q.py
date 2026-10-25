import sys,urllib.request,urllib.parse,xml.etree.ElementTree as ET,time
ns={'a':'http://www.w3.org/2005/Atom'}
def go(q,n=40):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
        {'search_query':q,'start':0,'max_results':n,'sortBy':'submittedDate','sortOrder':'descending'})
    d=urllib.request.urlopen(url,timeout=60).read()
    r=ET.fromstring(d)
    es=r.findall('a:entry',ns)
    print(f"===== {q}  -> {len(es)} hits =====")
    for e in es:
        i=e.find('a:id',ns).text.split('/abs/')[-1]
        t=' '.join(e.find('a:title',ns).text.split())
        dt=e.find('a:published',ns).text[:10]
        au=', '.join(x.find('a:name',ns).text for x in e.findall('a:author',ns))
        print(f"{i} | {dt} | {t}\n     AUTHORS: {au}")
    print()
for q in sys.argv[1:]:
    try: go(q)
    except Exception as ex: print(f"ERR {q}: {ex}")
    time.sleep(3)
