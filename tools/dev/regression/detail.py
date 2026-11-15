import json,sys
m={c['id']:c for c in json.load(open(sys.argv[1]))}; r={c['id']:c for c in json.load(open(sys.argv[2]))}
for spec in sys.argv[3:]:
    cid,i=map(int,spec.split(':'))
    for tag,d in (('MAIN',m),('PR',r)):
        s=d[cid]['steps'][i]
        print(f"--- {tag} chain {cid} step {i} {s['step']} und={s.get('understood')} unm={s.get('unmatched')!r} path={s.get('path')} ms={s.get('ms')} tok={s.get('tokens')} delta={s.get('stats_delta')}")
        print("   chips", [c['label'] for c in s['chips']])
        print("   before:", (s.get('prompt_before') or '')[:400])
        print("   after: ", (s.get('prompt') or '')[:700])
        for l in s.get('logs',[]): print("   log:", l[:400])
