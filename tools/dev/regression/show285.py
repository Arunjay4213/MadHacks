import json,sys
d={c['id']:c for c in json.load(open(sys.argv[1]))}
want=[(4,3),(31,2),(22,4),(16,9),(1016,9),(2016,9)]+[(9001,i) for i in range(6)]
for cid,i in want:
    if cid not in d or i>=len(d[cid]['steps']): continue
    s=d[cid]['steps'][i]
    print(f"[{cid}:{i}] {s['step']} path={s.get('path')} ms={s.get('ms')} tok={s.get('tokens')} chips={[c['label'] for c in s.get('chips',[])]}")
    print("    ", (s.get('prompt') or '')[:int(sys.argv[2]) if len(sys.argv)>2 else 700])
